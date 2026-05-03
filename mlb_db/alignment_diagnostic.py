#!/usr/bin/env python3
"""
Alignment diagnostic: check whether hip_shoulder_separation and wrist_height_rel_hip
peaks fall at consistent relative positions across all pitchers.

Usage: python alignment_diagnostic.py
Output: profiles/alignment_diagnostic.png  +  terminal report
"""

import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
PROFILES_DIR = BASE_DIR / "profiles"

# Feature indices (from features.py)
IDX_HIP_SHOULDER_SEP = 5   # hip_shoulder_separation  (weight 2.5 — highest)
IDX_WRIST_HEIGHT     = 9   # wrist_height_rel_hip

FEATURE_NAMES = [
    "elbow_angle_throw",        # 0
    "shoulder_abduction_throw", # 1
    "stride_knee_flexion",      # 2
    "back_knee_flexion",        # 3
    "stride_length_norm",       # 4
    "hip_shoulder_separation",  # 5  ← primary diagnostic
    "hip_rotation_frame",       # 6
    "shoulder_rotation_frame",  # 7
    "trunk_tilt_lateral",       # 8
    "wrist_height_rel_hip",     # 9  ← secondary diagnostic
    "elbow_angle_glove",        # 10
]

ALIGNED_MIN = 35.0   # % — expected release window
ALIGNED_MAX = 75.0


# ── Load profiles ──────────────────────────────────────────────────────────────
def load_all_profiles():
    profiles = {}
    for path in sorted(PROFILES_DIR.glob("*.json")):
        if path.stem in ("anchors", "similarity_matrix",
                         "similarity_matrix_features", "similarity_matrix_keypoints"):
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  [skip] {path.stem}: cannot read — {e}")
            continue

        fp = data.get("feature_profile", {})
        mf = fp.get("mean_features")
        if not mf:
            print(f"  [skip] {path.stem}: no mean_features")
            continue

        arr = np.array(mf, dtype=np.float32)   # (T, F)
        T, F = arr.shape
        if F < 11:
            print(f"  [skip] {path.stem}: only {F} features (expected 11)")
            continue

        # Degraded check: >50% of frames NaN in hip_shoulder_separation
        hip_col = arr[:, IDX_HIP_SHOULDER_SEP]
        nan_frac = np.isnan(hip_col).mean()
        if nan_frac > 0.5:
            print(f"  [skip] {path.stem}: {nan_frac*100:.0f}% NaN in hip_shoulder_sep — degraded")
            continue

        name = data.get("pitcher_name", path.stem)
        profiles[name] = arr
        print(f"  [ok]   {name:<26}  T={T:4d}  hip_sep NaN={nan_frac*100:.0f}%")

    return profiles


# ── Smooth helper ──────────────────────────────────────────────────────────────
def smooth(x, w=7):
    """Simple uniform moving-average, NaN-aware."""
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i - w // 2)
        hi = min(len(x), i + w // 2 + 1)
        chunk = x[lo:hi]
        valid = chunk[np.isfinite(chunk)]
        if valid.size:
            out[i] = valid.mean()
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("\nLoading profiles …")
    profiles = load_all_profiles()

    if not profiles:
        print("No clean profiles found.")
        sys.exit(1)

    print(f"\n{len(profiles)} pitchers included.\n")

    # Assign colors
    cmap   = plt.get_cmap("tab10")
    names  = sorted(profiles.keys())
    colors = {n: cmap(i % 10) for i, n in enumerate(names)}

    # ── Peak analysis ─────────────────────────────────────────────────────────
    print(f"{'Pitcher':<26}  {'T':>5}  {'hip_sep peak':>13}  {'wrist_ht peak':>14}  verdict")
    print("-" * 80)

    hip_peak_pcts   = {}
    wrist_peak_pcts = {}
    all_aligned     = True

    for name in names:
        arr = profiles[name]
        T   = arr.shape[0]
        pct = np.linspace(0, 100, T)

        hip_col   = smooth(arr[:, IDX_HIP_SHOULDER_SEP])
        wrist_col = smooth(arr[:, IDX_WRIST_HEIGHT])

        # NaN-safe argmax
        def nanargmax(col):
            valid = np.where(np.isfinite(col))[0]
            if not valid.size:
                return None
            return valid[np.argmax(col[valid])]

        hip_idx   = nanargmax(hip_col)
        wrist_idx = nanargmax(wrist_col)

        hip_pct   = pct[hip_idx]   if hip_idx   is not None else float("nan")
        wrist_pct = pct[wrist_idx] if wrist_idx is not None else float("nan")

        hip_peak_pcts[name]   = hip_pct
        wrist_peak_pcts[name] = wrist_pct

        in_window = ALIGNED_MIN <= hip_pct <= ALIGNED_MAX
        if not in_window:
            all_aligned = False

        flag = "" if in_window else "  ← OUTSIDE WINDOW"
        print(
            f"{name:<26}  {T:>5}  "
            f"frame {hip_idx:>4} ({hip_pct:>5.1f}%)  "
            f"frame {(wrist_idx if wrist_idx is not None else -1):>4} ({wrist_pct:>5.1f}%)"
            f"{flag}"
        )

    print()
    verdict = "ALIGNED" if all_aligned else "MISALIGNED"
    print(f"Verdict: {verdict}")
    print(f"  (window = {ALIGNED_MIN:.0f}–{ALIGNED_MAX:.0f}% of sequence)")
    print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Delivery Alignment Diagnostic\n"
                 "Shaded band = expected release window (35–75% of sequence)",
                 fontsize=13)

    feature_pairs = [
        (0, IDX_HIP_SHOULDER_SEP, "hip_shoulder_separation",  "Hip–Shoulder Separation"),
        (1, IDX_WRIST_HEIGHT,     "wrist_height_rel_hip",     "Wrist Height (rel. hip)"),
    ]

    for ax_idx, feat_idx, feat_key, feat_label in feature_pairs:
        ax = axes[ax_idx]

        # Release window band
        ax.axvspan(ALIGNED_MIN, ALIGNED_MAX, alpha=0.08, color="green",
                   label="expected release window")
        ax.axvline(ALIGNED_MIN, color="green", lw=0.8, ls="--", alpha=0.4)
        ax.axvline(ALIGNED_MAX, color="green", lw=0.8, ls="--", alpha=0.4)

        for name in names:
            arr    = profiles[name]
            T      = arr.shape[0]
            pct    = np.linspace(0, 100, T)
            col    = smooth(arr[:, feat_idx], w=9)
            color  = colors[name]
            lname  = name.split()[-1]   # last name for legend

            ax.plot(pct, col, color=color, lw=1.6, label=lname, alpha=0.85)

            # Mark peak with a dot
            peak_pct = (hip_peak_pcts if feat_key == "hip_shoulder_separation"
                        else wrist_peak_pcts)[name]
            if np.isfinite(peak_pct):
                peak_val_idx = int(round(peak_pct / 100 * (T - 1)))
                peak_val     = col[peak_val_idx] if np.isfinite(col[peak_val_idx]) else np.nan
                if np.isfinite(peak_val):
                    ax.scatter([peak_pct], [peak_val], color=color, s=60, zorder=5)

        ax.set_ylabel(feat_label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=3, loc="upper left")

    axes[1].set_xlabel("Sequence position (% of total frames)", fontsize=10)
    plt.tight_layout()

    out_path = PROFILES_DIR / "alignment_diagnostic.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
