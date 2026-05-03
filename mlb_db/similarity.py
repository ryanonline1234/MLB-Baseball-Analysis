"""
DTW similarity engine — pitcher-vs-pitcher mechanical comparison.

Two modes:

1. `features` (default, recommended) — DTW on biomechanical features from
   features.py (joint angles, inter-segment separations, normalized lengths),
   with first-derivative velocity channels concatenated. Z-scored per pair.
   Rotation / scale tolerant; handles low-confidence frames via NaN masking.

2. `keypoints` (legacy) — DTW on hip-centered, torso-scaled, LHP-mirrored
   (T, 12, 2) raw keypoint sequences. Kept for A/B comparison.

CLI entry point: `python run.py compare [--mode features|keypoints]`.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from pipeline import (
    LANDMARK_NAMES,
    N_LANDMARKS,
    PROFILES_DIR,
    _safe_name,
    load_pitchers_config,
)
from features import (
    FEATURE_GROUPS,
    FEATURE_NAMES,
    FEATURE_WEIGHTS,
    N_FEATURES,
    VELOCITY_WEIGHT_FACTOR,
    add_velocity_channels,
    build_weight_vector,
    compute_features,
    group_column_indices,
    mirror_pose,
)

logger = logging.getLogger(__name__)

# Landmark indices (see pipeline.LANDMARK_NAMES)
_L_SHOULDER, _R_SHOULDER = 0, 1
_L_HIP,      _R_HIP      = 6, 7


# ─── Load keypoints (legacy path) ────────────────────────────────────────────

def load_pitcher_sequence(name: str) -> Optional[np.ndarray]:
    """
    Load mean_keypoints from a pitcher's profile JSON.

    Returns (T, 12, 3) float32 array or None if the profile is missing /
    malformed.
    """
    path = PROFILES_DIR / f"{_safe_name(name)}.json"
    if not path.exists():
        logger.warning(f"[similarity] profile not found: {path}")
        return None

    try:
        data = json.loads(path.read_text())
        mean_kp = data.get("keypoint_profile", {}).get("mean_keypoints")
        if mean_kp is None:
            logger.warning(f"[similarity] {name}: no mean_keypoints in profile")
            return None
        seq = np.asarray(mean_kp, dtype=np.float32)
        if seq.ndim != 3 or seq.shape[1:] != (N_LANDMARKS, 3):
            logger.warning(
                f"[similarity] {name}: unexpected shape {seq.shape}, expected (T, {N_LANDMARKS}, 3)"
            )
            return None
        return seq
    except Exception as e:
        logger.error(f"[similarity] failed to load {name}: {e}")
        return None


# ─── Load features (new path) ────────────────────────────────────────────────

def load_pitcher_features(name: str) -> Optional[np.ndarray]:
    """
    Load mean_features from profile JSON. Returns (T, N_FEATURES) float32 or
    None. If the profile predates the feature upgrade (no feature_profile key),
    compute features on the fly from mean_keypoints as a fallback.
    """
    path = PROFILES_DIR / f"{_safe_name(name)}.json"
    if not path.exists():
        logger.warning(f"[similarity] profile not found: {path}")
        return None

    try:
        data = json.loads(path.read_text())
        fp = data.get("feature_profile")
        if fp and fp.get("mean_features") is not None:
            raw = fp["mean_features"]
            # JSON doesn't carry NaN; we stored NaN as None — convert back.
            def _to_float(v):
                return float("nan") if v is None else float(v)
            feats = np.asarray(
                [[_to_float(v) for v in row] for row in raw],
                dtype=np.float32,
            )
            if feats.ndim == 2 and feats.shape[1] == N_FEATURES:
                return feats
            logger.warning(
                f"[similarity] {name}: unexpected feature shape {feats.shape}"
            )
        # Fallback: compute from mean_keypoints using RHP handedness
        # (mean_keypoints are already stored raw — we don't know handedness
        # from the profile alone — so this is best-effort).
        logger.info(f"[similarity] {name}: recomputing features from keypoints")
        kp = load_pitcher_sequence(name)
        if kp is None:
            return None
        return compute_features(kp, handedness="R")
    except Exception as e:
        logger.error(f"[similarity] failed to load features for {name}: {e}")
        return None


# ─── Normalize keypoints (legacy path) ───────────────────────────────────────

# Landmark pairs swapped when mirroring L↔R
_MIRROR_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]


def normalize_pose_sequence(seq: np.ndarray, handedness: str) -> np.ndarray:
    """
    (T, 12, 3) → (T, 12, 2) body-relative, direction-normalized xy.

    1. Drop confidence channel → xy only
    2. Translate so hip midpoint = origin each frame
    3. Scale so torso height (hip-mid → shoulder-mid) = 1.0
    4. If LHP, mirror x AND swap L/R landmark pairs so the throwing arm always
       lives at the RIGHT indices. (A naive x-flip without the swap was the
       previous bug — it moved LHP points correctly but left throwing-arm
       landmarks mis-labeled, making every LHP↔RHP distance misleading.)
    """
    xy = seq[..., :2].astype(np.float32, copy=True)   # (T, 12, 2)

    # 2. hip midpoint per frame → origin
    hip_mid = (xy[:, _L_HIP] + xy[:, _R_HIP]) * 0.5
    xy -= hip_mid[:, None, :]

    # 3. torso-height scale per frame
    shoulder_mid = (xy[:, _L_SHOULDER] + xy[:, _R_SHOULDER]) * 0.5
    torso = np.linalg.norm(shoulder_mid, axis=1)
    torso = np.where(torso < 1e-6, 1.0, torso).astype(np.float32)
    xy /= torso[:, None, None]

    # 4. LHP mirror — flip x AND swap paired landmarks
    if handedness.upper() == "L":
        xy[..., 0] = -xy[..., 0]
        for li, ri in _MIRROR_PAIRS:
            tmp = xy[:, li, :].copy()
            xy[:, li, :] = xy[:, ri, :]
            xy[:, ri, :] = tmp

    return xy   # (T, 12, 2)


def _flatten_frames(xy: np.ndarray) -> np.ndarray:
    """(T, 12, 2) → (T, 24) float32 — frame vector for cdist."""
    return xy.reshape(xy.shape[0], -1).astype(np.float32, copy=False)


# ─── DTW (plain, no NaN) — legacy keypoint path ──────────────────────────────

def dtw_distance(
    seq1: np.ndarray,
    seq2: np.ndarray,
    band_frac: float = 0.10,
) -> Tuple[float, int]:
    """
    DTW with a Sakoe-Chiba band. Returns (total_cost, path_length).

    seq1, seq2 : (T, D) float32. Frame-distance matrix via cdist; DP recurrence
    is a scalar Python loop constrained to the band.
    """
    T1, _ = seq1.shape
    T2    = seq2.shape[0]
    band  = max(10, int(max(T1, T2) * band_frac), abs(T1 - T2) + 2)

    frame_d = cdist(seq1, seq2, metric="euclidean").astype(np.float32)

    cost     = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    path_len = np.zeros((T1 + 1, T2 + 1), dtype=np.int32)
    cost[0, 0] = 0.0

    for i in range(1, T1 + 1):
        j_min = max(1, i - band)
        j_max = min(T2, i + band)
        for j in range(j_min, j_max + 1):
            c_diag = cost[i - 1, j - 1]
            c_up   = cost[i - 1, j    ]
            c_left = cost[i,     j - 1]
            if c_diag <= c_up and c_diag <= c_left:
                prev_cost, prev_len = c_diag, path_len[i - 1, j - 1]
            elif c_up <= c_left:
                prev_cost, prev_len = c_up,   path_len[i - 1, j    ]
            else:
                prev_cost, prev_len = c_left, path_len[i,     j - 1]
            cost[i, j]     = frame_d[i - 1, j - 1] + prev_cost
            path_len[i, j] = prev_len + 1

    total = float(cost[T1, T2])
    plen  = int(path_len[T1, T2])
    if not np.isfinite(total) or plen == 0:
        logger.warning(
            f"[DTW] band={band} did not reach corner for T1={T1}, T2={T2}; "
            f"retrying unconstrained"
        )
        return dtw_distance(seq1, seq2, band_frac=1.0)
    return total, plen


# ─── Masked DTW — NaN-aware, used for feature mode ───────────────────────────

def dtw_distance_masked(
    seq1: np.ndarray,
    seq2: np.ndarray,
    band_frac: float = 0.10,
    min_valid_dim_frac: float = 0.5,
) -> Tuple[float, int]:
    """
    DTW with Sakoe-Chiba band and NaN masking.

    A frame is considered "valid" if at least `min_valid_dim_frac` of its
    feature dimensions are finite. A frame *pair* contributes to both cost and
    path length only when BOTH frames are valid. Invalid frame pairs pass
    cost and path length through unchanged — effectively skipping them in the
    alignment.

    NaN in valid frame-pairs is replaced by 0 for the distance computation
    (matches zero-mean post z-score), so residual NaN dims contribute zero.
    """
    T1, _ = seq1.shape
    T2    = seq2.shape[0]
    band  = max(10, int(max(T1, T2) * band_frac), abs(T1 - T2) + 2)

    v1 = np.isfinite(seq1).mean(axis=1) >= min_valid_dim_frac
    v2 = np.isfinite(seq2).mean(axis=1) >= min_valid_dim_frac
    valid_pair = v1[:, None] & v2[None, :]        # (T1, T2)

    s1 = np.nan_to_num(seq1, nan=0.0).astype(np.float32, copy=False)
    s2 = np.nan_to_num(seq2, nan=0.0).astype(np.float32, copy=False)
    frame_d = cdist(s1, s2, metric="euclidean").astype(np.float32)

    cost     = np.full((T1 + 1, T2 + 1), np.inf, dtype=np.float64)
    path_len = np.zeros((T1 + 1, T2 + 1), dtype=np.int32)
    cost[0, 0] = 0.0

    for i in range(1, T1 + 1):
        j_min = max(1, i - band)
        j_max = min(T2, i + band)
        for j in range(j_min, j_max + 1):
            c_diag = cost[i - 1, j - 1]
            c_up   = cost[i - 1, j    ]
            c_left = cost[i,     j - 1]
            if c_diag <= c_up and c_diag <= c_left:
                prev_cost, prev_len = c_diag, path_len[i - 1, j - 1]
            elif c_up <= c_left:
                prev_cost, prev_len = c_up,   path_len[i - 1, j    ]
            else:
                prev_cost, prev_len = c_left, path_len[i,     j - 1]

            if valid_pair[i - 1, j - 1]:
                cost[i, j]     = frame_d[i - 1, j - 1] + prev_cost
                path_len[i, j] = prev_len + 1
            else:
                # Invalid pair: skip contribution but propagate DP state.
                cost[i, j]     = prev_cost
                path_len[i, j] = prev_len

    total = float(cost[T1, T2])
    plen  = int(path_len[T1, T2])
    if not np.isfinite(total) or plen == 0:
        if band_frac < 1.0:
            logger.warning(
                f"[DTW masked] band={band} did not reach corner for T1={T1}, T2={T2}; "
                f"retrying unconstrained"
            )
            return dtw_distance_masked(seq1, seq2, band_frac=1.0)
        # Masking blocks every path — fall back to plain DTW on NaN-filled seqs.
        # We still report path_len as max(T1, T2) so the normalized distance is
        # comparable to normal pairs.
        logger.warning(
            f"[DTW masked] no valid alignment for T1={T1}, T2={T2}; "
            f"falling back to unmasked plain DTW (features likely sparse)"
        )
        return dtw_distance(s1, s2, band_frac=0.10)
    return total, plen


# ─── Helpers for feature mode ────────────────────────────────────────────────

def _zscore_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z-score each feature column using stats computed jointly on the concatenation
    of a and b. NaN-aware. Returns (a_z, b_z) with NaN preserved.
    """
    stacked = np.concatenate([a, b], axis=0)
    mean = np.nanmean(stacked, axis=0)
    std  = np.nanstd(stacked, axis=0)
    std  = np.where(std > 1e-9, std, 1.0)
    return ((a - mean) / std).astype(np.float32), ((b - mean) / std).astype(np.float32)


def _apply_weights(z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Scale each z-scored column by its importance weight. Broadcasting: (T, D) × (D,).
    Columns weighted to zero are effectively excluded from the DTW distance.
    """
    if z.shape[1] != weights.shape[0]:
        raise ValueError(
            f"weight vector dim {weights.shape[0]} ≠ feature dim {z.shape[1]}"
        )
    return (z * weights[None, :]).astype(np.float32)


def _minmax_similarity_matrix(
    dist: np.ndarray,
    valid_pair: Optional[np.ndarray] = None,
    anchor_min: Optional[float] = None,
    anchor_max: Optional[float] = None,
) -> np.ndarray:
    """
    Convert an (N, N) distance matrix into an (N, N) similarity % matrix using
    min-max rescaling over off-diagonal distances. Diagonal stays NaN.

    Fixed-anchor mode (preferred):
      If `anchor_min` and `anchor_max` are both provided, they are used directly
      as d_min / d_max instead of computing from the pool.
        anchor_min = anchor_100_dist  (same-delivery DTW — the "100%" ceiling)
        anchor_max = anchor_0_dist    (maximally-dissimilar DTW — the "0%" floor)
      Scores outside [0, 100] are clipped; this is expected for pairs that are
      more similar than the intra-pitcher anchor or less similar than the
      empirical floor.

    Pool-based mode (fallback / keypoints mode):
      If anchors are not provided, min/max are computed from the pool of valid
      off-diagonal distances.  If `valid_pair` is provided (N, N bool),
      degraded pairs don't set the scale but are still scored on it.
    """
    N = dist.shape[0]
    off_mask = ~np.eye(N, dtype=bool)

    if anchor_min is not None and anchor_max is not None:
        d_min, d_max = anchor_min, anchor_max
    else:
        # Pool-based fallback (existing behavior)
        if valid_pair is None:
            pool_mask = off_mask
        else:
            pool_mask = off_mask & valid_pair
        pool = dist[pool_mask]
        if pool.size:
            d_min, d_max = float(pool.min()), float(pool.max())
        else:
            # No valid pairs → fall back to all off-diagonal entries
            fb = dist[off_mask]
            d_min = float(fb.min()) if fb.size else 0.0
            d_max = float(fb.max()) if fb.size else 1.0

    span = max(d_max - d_min, 1e-9)
    sim = 100.0 * (d_max - dist) / span
    sim = np.clip(sim, 0.0, 100.0)
    sim[~off_mask] = np.nan
    return sim


def _degraded_mask(
    sequences: Dict[str, np.ndarray],
    names: List[str],
    min_valid_frame_frac: float = 0.5,
) -> Tuple[np.ndarray, List[str]]:
    """
    Identify pitchers whose feature sequences are too NaN-heavy to trust — any
    pair involving such a pitcher is marked degraded and excluded from the
    min-max endpoint pool.

    Returns (valid_pair (N, N) bool, degraded_names list).
    """
    N = len(names)
    pitcher_ok = np.ones(N, dtype=bool)
    degraded: List[str] = []
    for idx, name in enumerate(names):
        seq = sequences[name]
        # Frame is "valid" if at least one feature dim is finite.
        frac = float(np.isfinite(seq).any(axis=1).mean())
        if frac < min_valid_frame_frac:
            pitcher_ok[idx] = False
            degraded.append(name)
    valid_pair = pitcher_ok[:, None] & pitcher_ok[None, :]
    return valid_pair, degraded


# ─── Fixed-anchor computation ────────────────────────────────────────────────

def compute_anchors(kershaw_name: str = "Clayton Kershaw") -> Dict:
    """
    Compute and persist fixed DTW anchor distances to ``profiles/anchors.json``.

    Anchors are computed separately for each scoring scope:

      overall — full 22-feature (11 pos + 11 vel) z-scored + weighted sequence.
      per component — same Kershaw split-half and empirical-max-pair approach,
          but sliced to that component's column indices so the distances live on
          the same numeric scale as the per-component DTW used at compare time.

    Two anchors per scope:
      anchor_100_dist — DTW between first and second half of Kershaw's canonical
          sequence (same delivery measured twice → "100% similar" ceiling).
      anchor_0_dist   — Maximum pairwise DTW across all available clean pitcher
          profiles (provisional floor until a real non-pitching video is added,
          see ROADMAP-008).

    Scoring formula: sim% = 100 * (anchor_0 - d) / (anchor_0 - anchor_100),
    clipped to [0, 100]. Fixed constants → stable across DB changes.
    """
    logger.info(f"[Anchors] Loading {kershaw_name!r} feature sequence …")

    feats = load_pitcher_features(kershaw_name)
    if feats is None:
        raise ValueError(
            f"[Anchors] No feature profile for {kershaw_name!r}. "
            f"Run add-pitcher first."
        )

    weights = build_weight_vector(include_velocity=True)   # (22,)
    seq     = add_velocity_channels(feats)                 # (T, 22)
    T       = seq.shape[0]
    half    = T // 2

    # ── Load all clean pitcher sequences ──────────────────────────────────────
    pitchers    = load_pitchers_config()
    all_names:  List[str]             = []
    all_seqs:   Dict[str, np.ndarray] = {}

    for entry in pitchers:
        name    = entry["name"]
        feats_i = load_pitcher_features(name)
        if feats_i is None:
            continue
        seq_i = add_velocity_channels(feats_i)
        frac  = float(np.isfinite(seq_i).any(axis=1).mean())
        if frac < 0.5:
            logger.info(f"[Anchors] Skipping degraded profile: {name}")
            continue
        all_names.append(name)
        all_seqs[name]  = seq_i

    if len(all_names) < 2:
        raise RuntimeError(
            "[Anchors] Need ≥ 2 clean profiles to compute anchor_0. "
            "Run add-pitcher for more pitchers first."
        )

    n_pairs = len(all_names) * (len(all_names) - 1) // 2

    def _anchors_for_cols(cols: Optional[np.ndarray]) -> Tuple[float, float, List[str]]:
        """
        Compute (anchor_100, anchor_0, worst_pair) for a given column slice.
        cols=None means use all columns (full sequence).
        """
        def _slice(s: np.ndarray) -> np.ndarray:
            return s if cols is None else s[:, cols]

        # ── anchor_100: Kershaw split-half ──
        s1_raw = _slice(seq[:half])
        s2_raw = _slice(seq[half:])
        z1, z2 = _zscore_pair(s1_raw, s2_raw)
        # weights already baked into the full seq above; for sliced cols,
        # the z-scored values are already unit-variance → just apply weights
        w_vec  = weights if cols is None else weights[cols]
        w1 = _apply_weights(z1, w_vec)
        w2 = _apply_weights(z2, w_vec)
        cost, plen = dtw_distance_masked(w1, w2)
        a100 = cost / max(plen, 1)

        # ── anchor_0: empirical max across clean pairs ──
        max_d    = 0.0
        max_pair: List[str] = [all_names[0], all_names[1]]
        for i in range(len(all_names)):
            for j in range(i + 1, len(all_names)):
                si = _slice(all_seqs[all_names[i]])
                sj = _slice(all_seqs[all_names[j]])
                zi, zj = _zscore_pair(si, sj)
                wi = _apply_weights(zi, w_vec)
                wj = _apply_weights(zj, w_vec)
                c, p = dtw_distance_masked(wi, wj)
                d    = c / max(p, 1)
                if d > max_d:
                    max_d    = d
                    max_pair = [all_names[i], all_names[j]]
        return a100, max_d, max_pair

    # ── Overall anchors ────────────────────────────────────────────────────────
    logger.info(f"[Anchors] Overall ({n_pairs} pairs, T={T}, half={half}) …")
    overall_100, overall_0, overall_pair = _anchors_for_cols(None)
    logger.info(
        f"[Anchors] overall: 100%={overall_100:.6f}, "
        f"0%={overall_0:.6f}  (worst: {overall_pair[0]} ↔ {overall_pair[1]})"
    )

    # ── Per-component anchors ─────────────────────────────────────────────────
    comp_anchors: Dict[str, Dict] = {}
    for g in FEATURE_GROUPS:
        cols = group_column_indices(g, include_velocity=True)
        logger.info(f"[Anchors] {g} ({n_pairs} pairs, {len(cols)} cols) …")
        a100, a0, wp = _anchors_for_cols(cols)
        comp_anchors[g] = {
            "anchor_100_dist": a100,
            "anchor_0_dist":   a0,
            "anchor_0_pair":   wp,
        }
        logger.info(
            f"[Anchors]   {g}: 100%={a100:.6f}, 0%={a0:.6f}  "
            f"(worst: {wp[0]} ↔ {wp[1]})"
        )

    result: Dict = {
        "anchor_100_dist":    overall_100,
        "anchor_100_source":  "kershaw_split_half",
        "anchor_0_dist":      overall_0,
        "anchor_0_source":    "empirical_max_provisional",
        "anchor_0_pair":      overall_pair,
        "components":         comp_anchors,
        "computed_at":        datetime.now().isoformat(timespec="seconds"),
    }

    anchors_path = PROFILES_DIR / "anchors.json"
    anchors_path.write_text(json.dumps(result, indent=2))
    logger.info(f"[Anchors] Saved → {anchors_path}")
    return result


# ─── Matrix ──────────────────────────────────────────────────────────────────

def compute_similarity_matrix(
    pitcher_entries: List[Dict],
    mode: str = "features",
) -> Dict:
    """
    All-pairs normalized DTW across every pitcher with a profile.

    mode : "features" — biomech features + velocity + z-score + masked DTW (default).
           "keypoints" — legacy raw-keypoint DTW (pre-upgrade baseline).
    """
    mode = mode.lower()
    if mode not in ("features", "keypoints"):
        raise ValueError(f"mode must be 'features' or 'keypoints', got {mode!r}")

    names      : List[str]         = []
    handedness : Dict[str, str]    = {}
    sequences  : Dict[str, np.ndarray] = {}
    seq_lengths: Dict[str, int]    = {}

    for entry in pitcher_entries:
        name = entry["name"]
        hand = entry.get("handedness", "R")

        if mode == "keypoints":
            raw = load_pitcher_sequence(name)
            if raw is None:
                logger.info(f"[similarity] skipping {name} (no profile)")
                continue
            norm = normalize_pose_sequence(raw, hand)
            seq = _flatten_frames(norm)
        else:
            feats = load_pitcher_features(name)
            if feats is None:
                logger.info(f"[similarity] skipping {name} (no profile)")
                continue
            seq = add_velocity_channels(feats)   # (T, 2F)

        names.append(name)
        handedness[name]  = hand
        sequences[name]   = seq
        seq_lengths[name] = seq.shape[0]

    N = len(names)
    if N < 2:
        logger.error(f"[similarity] need ≥2 profiles, got {N}")
        return {
            "mode":            mode,
            "pitchers":        names,
            "handedness":      handedness,
            "distance_matrix": [],
            "rankings":        {},
            "computed_at":     datetime.now().isoformat(timespec="seconds"),
        }

    # ── Load fixed anchors (features mode only) ───────────────────────────────
    # profiles/anchors.json holds per-component anchor distances calibrated on
    # the Kershaw split-half (100% ceiling) and the empirical max pair (0% floor).
    # Each component has its own anchors because component DTW distances live on
    # a different numeric scale than the full-sequence DTW (fewer columns → smaller
    # Euclidean distances).  Without the file, fall back to pool-based rescaling.
    comp_anchor_100: Dict[str, Optional[float]] = {g: None for g in FEATURE_GROUPS}
    comp_anchor_0:   Dict[str, Optional[float]] = {g: None for g in FEATURE_GROUPS}
    if mode == "features":
        anchors_path = PROFILES_DIR / "anchors.json"
        if anchors_path.exists():
            try:
                _anchors     = json.loads(anchors_path.read_text())
                _comp_data   = _anchors.get("components", {})
                for g in FEATURE_GROUPS:
                    gd = _comp_data.get(g, {})
                    if gd.get("anchor_100_dist") is not None:
                        comp_anchor_100[g] = float(gd["anchor_100_dist"])
                        comp_anchor_0[g]   = float(gd["anchor_0_dist"])
                loaded_any = any(v is not None for v in comp_anchor_100.values())
                if loaded_any:
                    logger.info(
                        f"[similarity] Fixed per-component anchors loaded from anchors.json"
                    )
                else:
                    logger.warning(
                        "[similarity] anchors.json has no 'components' block — "
                        "falling back to pool-based normalization. "
                        "Re-run compute_anchors() to regenerate."
                    )
            except Exception as _e:
                logger.warning(
                    f"[similarity] Failed to parse anchors.json ({_e}) "
                    f"— falling back to pool-based normalization."
                )
        else:
            print(
                "WARNING: profiles/anchors.json not found — falling back to "
                "pool-based normalization. Run compute_anchors() to fix."
            )

    # ── Pairwise DTW
    # Features mode: z-score → weight → DTW on full sequence (overall),
    # then z-score → weight → slice → DTW per component. We z-score ONCE per
    # pair on the full feature set so component scores share the same per-pair
    # normalization as the overall (keeps scales comparable).
    weights = build_weight_vector(include_velocity=True) if mode == "features" else None
    component_indices: Dict[str, np.ndarray] = (
        {g: group_column_indices(g, include_velocity=True) for g in FEATURE_GROUPS}
        if mode == "features" else {}
    )

    dist_matrix = np.zeros((N, N), dtype=np.float64)
    component_dist: Dict[str, np.ndarray] = {
        g: np.zeros((N, N), dtype=np.float64) for g in component_indices
    }
    t0 = time.time()
    for i in range(N):
        for j in range(i + 1, N):
            s1 = sequences[names[i]]
            s2 = sequences[names[j]]
            if mode == "features":
                z1, z2 = _zscore_pair(s1, s2)
                w1 = _apply_weights(z1, weights)
                w2 = _apply_weights(z2, weights)
                cost, plen = dtw_distance_masked(w1, w2)
                # Per-component DTW on the same z-scored+weighted sequences
                for g, cols in component_indices.items():
                    cs1 = w1[:, cols]
                    cs2 = w2[:, cols]
                    c_cost, c_plen = dtw_distance_masked(cs1, cs2)
                    cd = c_cost / max(c_plen, 1)
                    component_dist[g][i, j] = cd
                    component_dist[g][j, i] = cd
            else:
                cost, plen = dtw_distance(s1, s2)
            d = cost / max(plen, 1)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            logger.info(
                f"[DTW:{mode}] {names[i]:>14s} ↔ {names[j]:<14s}  "
                f"d={d:.4f}  (len {seq_lengths[names[i]]}, {seq_lengths[names[j]]})"
            )
    logger.info(f"[DTW:{mode}] {N*(N-1)//2} pairs computed in {time.time()-t0:.1f}s")

    # ── Degraded-pitcher detection (features mode only).
    # A pitcher whose feature sequence is mostly NaN almost certainly hit the
    # DTW NaN fallback, which produces distances on a different numeric scale.
    # We still compute and display their similarities, but exclude their pairs
    # from setting the min-max endpoints used to rescale everyone else.
    if mode == "features":
        valid_pair, degraded_names = _degraded_mask(sequences, names)
        if degraded_names:
            logger.warning(
                f"[similarity] degraded pitcher(s) excluded from min-max "
                f"endpoints: {degraded_names}"
            )
    else:
        valid_pair, degraded_names = np.ones((N, N), dtype=bool), []

    # ── Per-component similarity % (features mode)
    # Fixed-anchor mode: each component uses its own calibrated anchor pair
    # (Kershaw split-half and empirical-max for that component's column slice).
    # Pool-based fallback when anchors.json is absent or has no components block.
    component_sim: Dict[str, np.ndarray] = {
        g: _minmax_similarity_matrix(
            cm,
            valid_pair=valid_pair,
            anchor_min=comp_anchor_100[g],
            anchor_max=comp_anchor_0[g],
        )
        for g, cm in component_dist.items()
    }

    # ── Overall similarity %
    # Features mode: weighted mean of component similarity %'s (weights =
    #   summed FEATURE_WEIGHTS within each group, so heavier-weighted features
    #   pull the overall). Decouples the displayed overall from the raw full-
    #   sequence DTW scale, which is what the Skenes NaN fallback was poisoning.
    # Keypoints mode: legacy min-max on the full DTW distance.
    if mode == "features":
        group_weight: Dict[str, float] = {
            g: float(sum(FEATURE_WEIGHTS[f] for f in FEATURE_GROUPS[g]))
            for g in component_indices
        }
        total_w = sum(group_weight.values()) or 1.0
        # Weighted mean over the (N, N) component similarity matrices.
        overall_sim = np.zeros((N, N), dtype=np.float64)
        for g in component_indices:
            overall_sim += component_sim[g] * group_weight[g]
        overall_sim = overall_sim / total_w
        # Preserve NaN on the diagonal
        np.fill_diagonal(overall_sim, np.nan)
    else:
        overall_sim = _minmax_similarity_matrix(dist_matrix, valid_pair=valid_pair)

    rankings: Dict[str, List[Dict]] = {}
    for i, name in enumerate(names):
        others = [
            (names[j], j, float(dist_matrix[i, j]), float(overall_sim[i, j]))
            for j in range(N) if j != i
        ]
        # Sort by similarity DESC (highest-similarity first) in feature mode;
        # keypoint mode still sorts by raw distance (no components to weight).
        if mode == "features":
            others.sort(key=lambda x: -x[3])
        else:
            others.sort(key=lambda x: x[2])
        entries = []
        for other_name, j, d, sim in others:
            entry = {
                "pitcher":        other_name,
                "distance":       round(d, 6),
                "similarity_pct": round(sim, 1),
            }
            if mode == "features":
                entry["degraded"] = bool(
                    name in degraded_names or other_name in degraded_names
                )
                entry["components"] = {
                    g: {
                        "distance":       round(float(component_dist[g][i, j]), 6),
                        "similarity_pct": round(float(component_sim[g][i, j]), 1),
                    }
                    for g in component_indices
                }
            entries.append(entry)
        rankings[name] = entries

    result = {
        "mode":            mode,
        "pitchers":        names,
        "handedness":      handedness,
        "sequence_length": seq_lengths,
        "distance_matrix": [[round(v, 6) for v in row] for row in dist_matrix.tolist()],
        "rankings":        rankings,
        "computed_at":     datetime.now().isoformat(timespec="seconds"),
    }
    if mode == "features":
        result["components"] = {
            g: {
                "features":        FEATURE_GROUPS[g],
                "weight":          round(group_weight[g], 4),
                "distance_matrix": [[round(v, 6) for v in row] for row in component_dist[g].tolist()],
            }
            for g in component_indices
        }
        result["feature_weights"]        = FEATURE_WEIGHTS
        result["velocity_weight_factor"] = VELOCITY_WEIGHT_FACTOR
        result["degraded_pitchers"]      = degraded_names
        result["overall_scoring"]        = (
            "weighted mean of component similarity %'s; "
            "weights = sum of FEATURE_WEIGHTS within each group"
        )
    return result


# ─── Report ──────────────────────────────────────────────────────────────────

def _last_name(full: str) -> str:
    parts = full.strip().split()
    return parts[-1] if parts else full


def _bar(pct: float, width: int = 20) -> str:
    filled = int(round(pct / 100.0 * width))
    return "█" * filled + "·" * (width - filled)


def _mode_label(mode: str) -> str:
    return {
        "features":  "biomech features + velocity, z-scored, masked DTW",
        "keypoints": "hip-centered + torso-scaled keypoints, plain DTW (legacy)",
    }.get(mode, mode)


def print_similarity_report(
    result: Dict,
    filter_name: Optional[str] = None,
) -> None:
    """Pretty-print distance matrix and ranked lists."""
    names = result["pitchers"]
    matrix = result["distance_matrix"]
    rankings = result["rankings"]
    mode = result.get("mode", "features")

    if not names:
        print("No pitcher profiles available. Run `add-pitcher` first.")
        return

    if filter_name is None:
        print()
        print("=" * 72)
        print(f"  PITCHER MECHANICAL SIMILARITY  —  mode: {mode}")
        print(f"  ({_mode_label(mode)})")
        print("=" * 72)

        col_w = 12
        header = " " * 20 + "".join(f"{_last_name(n):>{col_w}}" for n in names)
        print()
        print(header)
        print("-" * len(header))
        for i, n in enumerate(names):
            row_label = f"{n:<20}"
            cells = ""
            for j in range(len(names)):
                if i == j:
                    cells += f"{'—':>{col_w}}"
                else:
                    cells += f"{matrix[i][j]:>{col_w}.4f}"
            print(row_label + cells)
        print()

    # Component columns (only meaningful in features mode)
    has_components = mode == "features" and any(
        "components" in r for r_list in rankings.values() for r in r_list
    )
    comp_order = (
        list(next(iter(rankings.values()))[0]["components"].keys())
        if has_components and rankings and rankings[next(iter(rankings))]
        else []
    )

    to_show = [filter_name] if filter_name else names
    for name in to_show:
        if name not in rankings:
            print(f"\n  {name}: no profile / ranking available")
            continue
        print(f"\n  {name}  ({result['handedness'].get(name, '?')})")

        if has_components:
            # Header: pitcher | overall | each component
            col_w = 10
            header = (
                f"    {'':<26} {'overall':>9}  "
                + "  ".join(f"{g.replace('_', ' '):>{col_w}}" for g in comp_order)
            )
            print(header)
            print("    " + "-" * (len(header) - 4))
            for i, r in enumerate(rankings[name], 1):
                flag = " *" if r.get("degraded") else ""
                label = f"{i}. {r['pitcher']}{flag}"
                row = f"    {label:<26} {r['similarity_pct']:>8.1f}%  "
                row += "  ".join(
                    f"{r['components'][g]['similarity_pct']:>{col_w-1}.1f}%"
                    for g in comp_order
                )
                print(row)
            # Legend
            print()
            print(f"    {'overall':<12} = all 11 features, z-scored, importance-weighted")
            for g in comp_order:
                feats = ", ".join(FEATURE_GROUPS[g])
                print(f"    {g.replace('_', ' '):<12} = {feats}")
            deg = result.get("degraded_pitchers") or []
            if deg:
                print()
                print(
                    f"    * = pair involves a degraded profile "
                    f"({', '.join(deg)}); excluded from min-max endpoints. "
                    f"Component %'s are computed against the clean-pair scale "
                    f"and may read as 0% or 100% — treat as unreliable."
                )
        else:
            for i, r in enumerate(rankings[name], 1):
                print(
                    f"    {i}. {r['pitcher']:<22} "
                    f"{r['similarity_pct']:>5.1f}%  {_bar(r['similarity_pct'])}  "
                    f"(d={r['distance']:.4f})"
                )
    print()


def print_side_by_side(features_result: Dict, keypoints_result: Dict) -> None:
    """
    Print both matrices and a rank-difference table so we can visually verify
    whether feature-mode meaningfully changes the ordering.
    """
    names = features_result["pitchers"]
    if keypoints_result["pitchers"] != names:
        logger.warning("[compare] pitcher lists differ; falling back to separate prints")
        print_similarity_report(features_result)
        print_similarity_report(keypoints_result)
        return

    print()
    print("=" * 72)
    print("  A/B COMPARISON — feature mode vs legacy keypoint mode")
    print("=" * 72)

    for name in names:
        feat_rank = features_result["rankings"][name]
        kp_rank   = keypoints_result["rankings"][name]
        kp_order  = {r["pitcher"]: idx for idx, r in enumerate(kp_rank, 1)}
        print(f"\n  {name}  ({features_result['handedness'].get(name, '?')})")
        print(f"    {'features':<34} {'keypoints (legacy)':<34}")
        for i, fr in enumerate(feat_rank, 1):
            kp_pos = kp_order.get(fr["pitcher"], "?")
            arrow  = "↑" if isinstance(kp_pos, int) and kp_pos > i else \
                     "↓" if isinstance(kp_pos, int) and kp_pos < i else "="
            kr = next((r for r in kp_rank if r["pitcher"] == fr["pitcher"]), None)
            kp_text = f"{kp_pos}. {fr['pitcher']:<18} {kr['similarity_pct']:>5.1f}%" \
                      if kr else f"?. {fr['pitcher']}"
            print(
                f"    {i}. {fr['pitcher']:<18} {fr['similarity_pct']:>5.1f}%  {arrow}  "
                f"{kp_text}"
            )
    print()
