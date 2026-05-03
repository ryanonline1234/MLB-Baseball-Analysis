"""
features.py — biomechanical feature extraction from 2D pose sequences.

Converts (T, 12, 3) keypoint arrays [x, y, confidence] into (T, F) sequences
of derived biomechanical features. Unlike raw keypoint positions, most of
these features are invariant to body size (normalized by torso length) and
many are invariant to body orientation in the camera frame (joint angles and
inter-segment angle *differences*). This makes pitcher-to-pitcher comparison
more robust to camera-angle drift between source videos.

Handedness is resolved internally: LHP keypoints are mirrored (x-flip + L/R
landmark swap) so all "throwing-side" features read from the RIGHT landmarks.
The caller supplies handedness; features.py handles the mirror.

Feature list (F = 11)
---------------------
  0  elbow_angle_throw         — rotation + scale invariant (interior joint angle)
  1  elbow_angle_glove         — rotation + scale invariant
  2  shoulder_abduction_throw  — rotation invariant (torso-axis ↔ upper-arm angle)
  3  hip_rotation_frame        — NOT rotation invariant (hip line ↔ +x axis)
  4  shoulder_rotation_frame   — NOT rotation invariant
  5  hip_shoulder_separation   — rotation invariant (difference of 3 and 4)
  6  stride_knee_flexion       — rotation + scale invariant
  7  back_knee_flexion         — rotation + scale invariant
  8  trunk_tilt_lateral        — NOT rotation invariant (torso axis ↔ vertical)
  9  wrist_height_rel_hip      — scale invariant (wrist.y − hip.y) / torso_len
 10  stride_length_norm        — scale invariant, |Δankle_x| / torso_len

Note on rotation invariance: "rotation" here means a 2D rotation of the pitcher
in the camera plane — the closest analog of mild camera-angle drift.
Interior-joint angles are unchanged by any rigid transform of the points.
Frame-relative angles (3, 4, 8) rotate with the body. Feature 5 is constructed
as a difference of 3 and 4 and is therefore rotation invariant by construction.

NaN handling
------------
If any required keypoint for a feature has confidence < `conf_threshold`
(default 0.3), that feature is set to NaN for that frame. NaN runs of
≤ `ffill_max_gap` frames (default 3) are forward-filled from the last valid
value; longer gaps remain NaN for downstream masking in the DTW layer.

Feature forward-declaration dimensionality check in tests/test_features.py.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# ─── Landmark indices (must match pipeline.LANDMARK_NAMES order) ─────────────
L_SHOULDER, R_SHOULDER = 0, 1
L_ELBOW,    R_ELBOW    = 2, 3
L_WRIST,    R_WRIST    = 4, 5
L_HIP,      R_HIP      = 6, 7
L_KNEE,     R_KNEE     = 8, 9
L_ANKLE,    R_ANKLE    = 10, 11

# ─── Feature registry ────────────────────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    "elbow_angle_throw",
    "elbow_angle_glove",
    "shoulder_abduction_throw",
    "hip_rotation_frame",
    "shoulder_rotation_frame",
    "hip_shoulder_separation",
    "stride_knee_flexion",
    "back_knee_flexion",
    "trunk_tilt_lateral",
    "wrist_height_rel_hip",
    "stride_length_norm",
]
N_FEATURES: int = len(FEATURE_NAMES)

# Subset guaranteed to be invariant under a 2D rigid rotation of the body in
# the camera frame. Used by tests/test_features.py.
ROTATION_INVARIANT_FEATURES: List[str] = [
    "elbow_angle_throw",
    "elbow_angle_glove",
    "shoulder_abduction_throw",
    "hip_shoulder_separation",
    "stride_knee_flexion",
    "back_knee_flexion",
]

# Scale-invariant in the sense that they are angles (unitless) OR are explicitly
# divided by torso length before returning.
SCALE_INVARIANT_FEATURES: List[str] = list(FEATURE_NAMES)  # all of them

VELOCITY_FEATURE_NAMES: List[str] = FEATURE_NAMES + [n + "_vel" for n in FEATURE_NAMES]

CONF_THRESHOLD: float = 0.3


# ─── Coaching-meaningful feature groups ──────────────────────────────────────
# Each feature appears in exactly one group. The groups are chosen so that a
# per-component similarity score maps to a category a pitching coach would
# actually give feedback on. "overall" is computed as the full 11-feature DTW
# and is NOT a weighted sum of the components (they use different subsets and
# different per-pair z-score stats).
FEATURE_GROUPS: "dict[str, List[str]]" = {
    "arm_action": [
        "elbow_angle_throw",
        "shoulder_abduction_throw",
        "wrist_height_rel_hip",
    ],
    "lower_body": [
        "stride_knee_flexion",
        "back_knee_flexion",
        "stride_length_norm",
    ],
    "rotation_timing": [
        "hip_rotation_frame",
        "shoulder_rotation_frame",
        "hip_shoulder_separation",
    ],
    "posture": [
        "trunk_tilt_lateral",
        "elbow_angle_glove",
    ],
}

# Sanity: groups partition FEATURE_NAMES.
_grouped = [f for g in FEATURE_GROUPS.values() for f in g]
assert sorted(_grouped) == sorted(FEATURE_NAMES), (
    f"FEATURE_GROUPS must partition FEATURE_NAMES; got {sorted(_grouped)}"
)


# ─── Feature importance weights (applied AFTER per-pair z-score) ─────────────
# Rationale: every feature is unit-variance post-z-score, so these weights
# directly scale each column's contribution to the DTW frame distance.
# Tuned from pitching-biomech priors:
#   - hip-shoulder separation ("X-factor") and shoulder abduction / arm slot
#     are the most discriminative between pitcher archetypes → weighted up.
#   - back-knee flexion and absolute hip/shoulder frame rotations are either
#     redundant or camera-dependent → weighted down.
# Velocity channels inherit their base feature's weight, scaled by
# VELOCITY_WEIGHT_FACTOR (velocities are noisier than positions).
FEATURE_WEIGHTS: "dict[str, float]" = {
    "elbow_angle_throw":        1.5,
    "elbow_angle_glove":        0.7,
    "shoulder_abduction_throw": 2.0,
    # hip_rotation_frame / shoulder_rotation_frame are measured in the CAMERA
    # frame, not the body frame — they rotate 1:1 with any camera-angle drift
    # between source videos. Their only mechanically-meaningful information is
    # their DIFFERENCE, which is already captured by hip_shoulder_separation
    # (a rotation-invariant derived feature, weighted 2.5). Zeroed out to stop
    # them injecting camera noise into the rotation_timing component.
    "hip_rotation_frame":       0.0,
    "shoulder_rotation_frame":  0.0,
    "hip_shoulder_separation":  2.5,
    "stride_knee_flexion":      0.8,
    "back_knee_flexion":        0.4,
    # trunk_tilt_lateral is also camera-frame-relative but still useful as a
    # proxy for posture when source videos are all roughly side-on. Keep it
    # with a moderate weight; revisit if it becomes the dominant signal.
    "trunk_tilt_lateral":       1.2,
    "wrist_height_rel_hip":     2.0,
    "stride_length_norm":       1.5,
}
assert set(FEATURE_WEIGHTS.keys()) == set(FEATURE_NAMES), (
    "FEATURE_WEIGHTS must cover every feature in FEATURE_NAMES"
)

VELOCITY_WEIGHT_FACTOR: float = 0.5


def build_weight_vector(include_velocity: bool = True) -> np.ndarray:
    """
    Return a (F,) or (2F,) float32 weight vector aligned to FEATURE_NAMES
    (and then the velocity block if include_velocity).
    """
    base = np.array([FEATURE_WEIGHTS[n] for n in FEATURE_NAMES], dtype=np.float32)
    if not include_velocity:
        return base
    return np.concatenate([base, base * VELOCITY_WEIGHT_FACTOR]).astype(np.float32)


def group_column_indices(
    group_name: str,
    include_velocity: bool = True,
) -> np.ndarray:
    """
    Column indices into a (T, F) or (T, 2F) feature+velocity array for the
    features in `group_name`. Used to slice out a component-specific subsequence
    for per-component DTW.
    """
    names = FEATURE_GROUPS[group_name]
    pos_idx = [FEATURE_NAMES.index(n) for n in names]
    if not include_velocity:
        return np.array(pos_idx, dtype=np.int32)
    vel_idx = [N_FEATURES + i for i in pos_idx]
    return np.array(pos_idx + vel_idx, dtype=np.int32)


# ─── Mirror (LHP → RHP frame) ────────────────────────────────────────────────
_MIRROR_PAIRS: List[Tuple[int, int]] = [
    (L_SHOULDER, R_SHOULDER), (L_ELBOW, R_ELBOW), (L_WRIST, R_WRIST),
    (L_HIP, R_HIP),           (L_KNEE,  R_KNEE),  (L_ANKLE, R_ANKLE),
]


def mirror_pose(keypoints: np.ndarray) -> np.ndarray:
    """
    Mirror a (T, 12, 3) keypoint sequence: flip x around 0.5 (normalized coords)
    AND swap each left/right landmark pair.

    Flipping x alone is insufficient — it moves the points correctly but leaves
    landmark-index labels wrong (e.g. a LHP's left_wrist, the throwing wrist,
    would still be stored at index 4 after a naked x-flip, but for an RHP the
    throwing wrist is at index 5). We swap the pairs so that after mirroring,
    throwing-side landmarks always live at the RIGHT indices.
    """
    out = keypoints.copy()
    out[..., 0] = 1.0 - out[..., 0]
    for li, ri in _MIRROR_PAIRS:
        # copy needed — swap via intermediate to avoid aliasing
        tmp = out[:, li, :].copy()
        out[:, li, :] = out[:, ri, :]
        out[:, ri, :] = tmp
    return out


# ─── Geometry helpers ────────────────────────────────────────────────────────
def _angle_at_joint(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle at b formed by rays b→a and b→c. Returns degrees in [0, 180]."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=-1) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _angle_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Unsigned angle between two 2-D vectors. Returns degrees in [0, 180]."""
    nu = np.linalg.norm(u, axis=-1)
    nv = np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(u * v, axis=-1) / (nu * nv)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _line_angle_horizontal(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Signed angle of vector p1→p2 relative to +x axis, degrees in [-180, 180]."""
    d = p2 - p1
    return np.degrees(np.arctan2(d[..., 1], d[..., 0]))


def _ffill_nan_1d(col: np.ndarray, max_gap: int) -> np.ndarray:
    """Forward-fill NaN runs of length ≤ max_gap along a 1-D array."""
    out = col.copy()
    T = len(out)
    i = 0
    while i < T:
        if np.isnan(out[i]):
            start = i
            while i < T and np.isnan(out[i]):
                i += 1
            end = i
            run = end - start
            if start > 0 and run <= max_gap:
                out[start:end] = out[start - 1]
        else:
            i += 1
    return out


def _ffill_nan(arr: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Forward-fill NaN along axis 0, per column, up to max_gap consecutive frames."""
    out = arr.copy()
    for j in range(arr.shape[1]):
        out[:, j] = _ffill_nan_1d(out[:, j], max_gap)
    return out


# ─── Main entry point ────────────────────────────────────────────────────────
def compute_features(
    keypoints: np.ndarray,
    handedness: str,
    conf_threshold: float = CONF_THRESHOLD,
    ffill_max_gap: int = 3,
) -> np.ndarray:
    """
    Compute the biomechanical feature sequence for a single pitcher video.

    Parameters
    ----------
    keypoints      : (T, 12, 3) array of [x, y, confidence] in [0, 1] coords.
    handedness     : 'R' or 'L'. LHP input is mirrored so throwing arm = right.
    conf_threshold : landmarks below this confidence produce NaN for any
                     feature that requires them (default 0.3).
    ffill_max_gap  : forward-fill short NaN runs of this length (default 3).

    Returns
    -------
    (T, N_FEATURES) float32 array. Units: degrees for all angles,
    torso-length ratios for wrist_height_rel_hip and stride_length_norm.
    """
    if keypoints.ndim != 3 or keypoints.shape[1:] != (12, 3):
        raise ValueError(f"expected (T, 12, 3), got {keypoints.shape}")

    kp = keypoints.astype(np.float32, copy=True)
    if handedness.upper() == "L":
        kp = mirror_pose(kp)

    T = kp.shape[0]
    xy = kp[..., :2]    # (T, 12, 2)
    conf = kp[..., 2]   # (T, 12)

    def p(idx: int) -> np.ndarray:
        """Fetch landmark idx, replacing low-confidence frames with NaN."""
        q = xy[:, idx, :].copy()
        q[conf[:, idx] < conf_threshold] = np.nan
        return q

    l_sh, r_sh = p(L_SHOULDER), p(R_SHOULDER)
    l_el, r_el = p(L_ELBOW),    p(R_ELBOW)
    l_wr, r_wr = p(L_WRIST),    p(R_WRIST)
    l_hi, r_hi = p(L_HIP),      p(R_HIP)
    l_kn, r_kn = p(L_KNEE),     p(R_KNEE)
    l_an, r_an = p(L_ANKLE),    p(R_ANKLE)

    hip_mid = (l_hi + r_hi) * 0.5            # (T, 2)
    sh_mid  = (l_sh + r_sh) * 0.5            # (T, 2)
    torso_vec = sh_mid - hip_mid             # (T, 2), head direction
    torso_len = np.linalg.norm(torso_vec, axis=-1)   # (T,)

    feats = np.full((T, N_FEATURES), np.nan, dtype=np.float32)

    # 0. Elbow angle (throwing arm = right, post-mirror)
    feats[:, 0] = _angle_at_joint(r_sh, r_el, r_wr)

    # 1. Elbow angle (glove arm = left)
    feats[:, 1] = _angle_at_joint(l_sh, l_el, l_wr)

    # 2. Shoulder abduction throwing side: angle between torso axis and upper arm.
    #    Rotation invariant because both vectors rotate together with the body.
    feats[:, 2] = _angle_between(torso_vec, r_el - r_sh)

    # 3. Hip rotation in camera frame: orientation of the hip line L→R.
    feats[:, 3] = _line_angle_horizontal(l_hi, r_hi)

    # 4. Shoulder rotation in camera frame: orientation of the shoulder line L→R.
    feats[:, 4] = _line_angle_horizontal(l_sh, r_sh)

    # 5. Hip-shoulder separation ("X-factor"): wrapped signed difference.
    #    Rotation invariant by construction (the body's rotation cancels).
    feats[:, 5] = ((feats[:, 4] - feats[:, 3] + 180.0) % 360.0) - 180.0

    # 6. Stride knee flexion (stride leg = LEFT post-mirror, i.e. glove-side foot
    #    strides toward the plate for an RHP).
    feats[:, 6] = _angle_at_joint(l_hi, l_kn, l_an)

    # 7. Back knee flexion (push-off leg = RIGHT post-mirror).
    feats[:, 7] = _angle_at_joint(r_hi, r_kn, r_an)

    # 8. Lateral trunk tilt: angle of torso vs. image vertical.
    #    Image y grows downward; "up" is -y. atan2(dx, -dy) puts 0° at vertical.
    feats[:, 8] = np.degrees(np.arctan2(torso_vec[..., 0], -torso_vec[..., 1]))

    # 9. Wrist height relative to hip, projected along image vertical and
    #    normalized by torso length. Positive = wrist above hip in image.
    #    Scale invariant via torso_len division.
    with np.errstate(invalid="ignore", divide="ignore"):
        safe_torso = np.where(torso_len > 1e-6, torso_len, np.nan)
        feats[:, 9] = -(r_wr[:, 1] - hip_mid[:, 1]) / safe_torso
        # 10. Stride length normalized: |right_ankle_x − left_ankle_x| / torso.
        feats[:, 10] = np.abs(r_an[:, 0] - l_an[:, 0]) / safe_torso

    feats = _ffill_nan(feats, max_gap=ffill_max_gap)
    return feats.astype(np.float32)


# ─── Velocity channels ───────────────────────────────────────────────────────
def add_velocity_channels(features: np.ndarray) -> np.ndarray:
    """
    Append first-derivative (time-gradient) channels.

    (T, F) → (T, 2F) float32. Velocity for frame t uses np.gradient (central
    differences in the interior, forward/backward at endpoints). NaN entries
    propagate — any feature NaN at frame t produces a NaN velocity at t-1..t+1.
    """
    if features.ndim != 2:
        raise ValueError(f"expected (T, F), got {features.shape}")
    vel = np.gradient(features, axis=0)
    return np.concatenate([features, vel], axis=1).astype(np.float32)
