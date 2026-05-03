"""
Tests for features.py.

Run with:
    cd mlb_db
    .venv/bin/python3 -m pytest tests/test_features.py -v
or just:
    .venv/bin/python3 tests/test_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Allow `python tests/test_features.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features import (   # noqa: E402
    FEATURE_NAMES,
    N_FEATURES,
    ROTATION_INVARIANT_FEATURES,
    add_velocity_channels,
    compute_features,
    mirror_pose,
)


# ─── Synthetic pose builder ──────────────────────────────────────────────────

def _synthetic_pitcher(T: int = 60) -> np.ndarray:
    """
    Build a plausible (T, 12, 3) keypoint sequence for a right-handed pitcher
    in a fixed upright stance. All points in normalized [0, 1] image coords
    with confidence = 0.9.

    Landmark order (matches pipeline.LANDMARK_NAMES):
      0 L_sh  1 R_sh   2 L_el  3 R_el   4 L_wr  5 R_wr
      6 L_hi  7 R_hi   8 L_kn  9 R_kn  10 L_an 11 R_an

    Over T frames the throwing (right) wrist sweeps overhead to simulate the
    arm action, so features like elbow angle + wrist-rel-hip have real dynamics.
    """
    kp = np.zeros((T, 12, 3), dtype=np.float32)

    # Static skeleton in image coords (y grows downward).
    # Center of body at (0.5, 0.5). Torso length ≈ 0.20.
    L_sh = np.array([0.42, 0.30]); R_sh = np.array([0.58, 0.30])
    L_el = np.array([0.38, 0.45]); R_el = np.array([0.62, 0.45])   # overwritten per-frame
    L_wr = np.array([0.36, 0.55]); R_wr = np.array([0.64, 0.55])   # overwritten per-frame
    L_hi = np.array([0.45, 0.50]); R_hi = np.array([0.55, 0.50])
    L_kn = np.array([0.43, 0.70]); R_kn = np.array([0.57, 0.70])
    L_an = np.array([0.38, 0.90]); R_an = np.array([0.62, 0.90])

    kp[:, 0, :2] = L_sh; kp[:, 1, :2] = R_sh
    kp[:, 6, :2] = L_hi; kp[:, 7, :2] = R_hi
    kp[:, 8, :2] = L_kn; kp[:, 9, :2] = R_kn
    kp[:, 10, :2] = L_an; kp[:, 11, :2] = R_an

    # Throwing arm sweeps: elbow goes up, wrist goes up+back over T frames.
    t_norm = np.linspace(0, 1, T)   # 0 → 1
    # Right elbow: from low-outside to high-alongside-head
    R_el_x = 0.62 - 0.10 * t_norm
    R_el_y = 0.45 - 0.25 * t_norm
    # Right wrist: from low to overhead
    R_wr_x = 0.64 - 0.14 * t_norm
    R_wr_y = 0.55 - 0.40 * t_norm
    kp[:, 3, 0] = R_el_x; kp[:, 3, 1] = R_el_y
    kp[:, 5, 0] = R_wr_x; kp[:, 5, 1] = R_wr_y

    # Glove arm stays roughly static.
    kp[:, 2, :2] = L_el
    kp[:, 4, :2] = L_wr

    # Confidence — set all to 0.9
    kp[..., 2] = 0.9
    return kp


def _rotate_around(kp: np.ndarray, center: np.ndarray, deg: float) -> np.ndarray:
    """Rotate (T, 12, 3) xy around `center` by `deg` degrees, leaving confidence alone."""
    out = kp.copy()
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = out[..., :2] - center
    rotated = xy @ R.T
    out[..., :2] = rotated + center
    return out


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_compute_features_shape():
    kp = _synthetic_pitcher(T=50)
    f = compute_features(kp, handedness="R")
    assert f.shape == (50, N_FEATURES), f"got {f.shape}"
    assert f.dtype == np.float32


def test_all_features_finite_on_clean_input():
    kp = _synthetic_pitcher(T=30)
    f = compute_features(kp, handedness="R")
    # Every feature should have at least some finite values
    for fi, fname in enumerate(FEATURE_NAMES):
        assert np.isfinite(f[:, fi]).any(), f"{fname} is all NaN"


def test_velocity_channel_doubles_dim():
    kp = _synthetic_pitcher(T=20)
    f = compute_features(kp, handedness="R")
    fv = add_velocity_channels(f)
    assert fv.shape == (20, 2 * N_FEATURES)


def test_rotation_invariance():
    """
    Rotating the whole body 30° around the hip midpoint should leave all
    rotation-invariant features (joint angles and angle-differences) unchanged
    within a small tolerance.
    """
    T = 40
    kp = _synthetic_pitcher(T=T)
    # Hip midpoint in image coords (using the static hips we defined)
    hip_mid = np.array([0.5, 0.5], dtype=np.float32)
    kp_rot = _rotate_around(kp, hip_mid, deg=30.0)

    f_orig = compute_features(kp,     handedness="R")
    f_rot  = compute_features(kp_rot, handedness="R")

    failures = []
    tol = 1.0   # degrees, accounting for float32 round-trip
    for fi, fname in enumerate(FEATURE_NAMES):
        if fname not in ROTATION_INVARIANT_FEATURES:
            continue
        diff = np.abs(f_orig[:, fi] - f_rot[:, fi])
        mask = np.isfinite(diff)
        if not mask.any():
            continue
        max_d = float(diff[mask].max())
        if max_d > tol:
            failures.append((fname, max_d))

    assert not failures, (
        "Rotation-invariant features changed under rotation:\n"
        + "\n".join(f"  {n}: max diff {d:.4f}°" for n, d in failures)
    )


def test_non_invariant_features_shift_with_rotation():
    """Sanity check — frame-relative features (hip_rotation_frame,
    shoulder_rotation_frame, trunk_tilt_lateral) SHOULD change by ~30°."""
    kp = _synthetic_pitcher(T=10)
    hip_mid = np.array([0.5, 0.5], dtype=np.float32)
    kp_rot = _rotate_around(kp, hip_mid, deg=30.0)

    f_orig = compute_features(kp,     handedness="R")
    f_rot  = compute_features(kp_rot, handedness="R")

    for fname in ("hip_rotation_frame", "shoulder_rotation_frame", "trunk_tilt_lateral"):
        fi = FEATURE_NAMES.index(fname)
        diff = np.abs(f_orig[:, fi] - f_rot[:, fi])
        # Accept wrap-around: if the diff is ~330° that's also a 30° rotation.
        diff = np.minimum(diff, 360 - diff)
        mask = np.isfinite(diff)
        assert mask.any(), fname
        median = float(np.median(diff[mask]))
        assert abs(median - 30.0) < 1.5, f"{fname}: expected ~30° shift, got {median:.2f}°"


def test_lhp_mirror_places_throwing_arm_on_right():
    """A LHP with arm action on their physical LEFT should, after mirroring,
    show the throwing elbow angle at index 0 (right/throwing), not index 1."""
    rhp = _synthetic_pitcher(T=30)

    # Build an LHP by swapping the animated throwing arm to the left side.
    lhp = rhp.copy()
    # Move the arm animation from right landmarks to left landmarks,
    # and make the left side's static arm live on the right.
    lhp[:, 2, :2] = rhp[:, 3, :2]   # L_elbow  ← R_elbow animation
    lhp[:, 4, :2] = rhp[:, 5, :2]   # L_wrist  ← R_wrist animation
    # Mirror x so the left-side motion lives on the left side of the image
    lhp[..., 0] = 1.0 - lhp[..., 0]
    lhp[:, 3, :2] = 1.0 - rhp[:, 2, :2] * np.array([1.0, 0.0]) + rhp[:, 2, :2] * np.array([0.0, 1.0])
    # (simpler: just use mirror_pose on the prepared LHP body for correctness)
    # Reset and use mirror_pose to build a clean LHP:
    lhp = mirror_pose(rhp)

    f_rhp = compute_features(rhp, handedness="R")
    f_lhp = compute_features(lhp, handedness="L")

    # After handedness resolution, throwing-side elbow angles should match
    # within a small tolerance (same underlying pose).
    fi = FEATURE_NAMES.index("elbow_angle_throw")
    diff = np.abs(f_rhp[:, fi] - f_lhp[:, fi])
    mask = np.isfinite(diff)
    assert mask.any()
    assert float(diff[mask].max()) < 1.0, (
        f"elbow_angle_throw differs between matched RHP and mirrored LHP: "
        f"max diff {float(diff[mask].max()):.3f}°"
    )


# ─── Self-running entry ──────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_compute_features_shape,
        test_all_features_finite_on_clean_input,
        test_velocity_channel_doubles_dim,
        test_rotation_invariance,
        test_non_invariant_features_shift_with_rotation,
        test_lhp_mirror_places_throwing_arm_on_right,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ! {t.__name__}: unexpected {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
