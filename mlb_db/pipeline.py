"""
MLB Pitcher Reference Database — Core Pipeline

Pose backends (in priority order):
  1. MMPose  — ViTPose-B, GPU via CUDA (preferred on DGX Spark)
  2. MediaPipe — CPU fallback for local dev machines

Keypoint output format (consistent across both backends):
  shape : (T, 12, 3)  float32
  axis 2: [x, y, confidence]   — x/y normalised to [0, 1]
  order : left_shoulder, right_shoulder, left_elbow, right_elbow,
          left_wrist, right_wrist, left_hip, right_hip,
          left_knee, right_knee, left_ankle, right_ankle
"""

import json
import logging
import urllib.request
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from datetime import datetime
from scipy.interpolate import interp1d

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Landmark config ──────────────────────────────────────────────────────────

LANDMARK_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]
N_LANDMARKS = len(LANDMARK_NAMES)  # 12

# MediaPipe landmark indices (LANDMARK_NAMES order)
_MP_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

# COCO-17 landmark indices (same body parts, MMPose output order)
_COCO_INDICES = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).parent
RAW_VIDEO_DIR = BASE_DIR / "raw_video"
KEYPOINTS_DIR = BASE_DIR / "keypoints"
SAVANT_DIR    = BASE_DIR / "savant"
PROFILES_DIR  = BASE_DIR / "profiles"

_MP_MODEL_PATH = BASE_DIR / "pose_landmarker_full.task"
_MP_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)


def ensure_dirs():
    for d in [RAW_VIDEO_DIR, KEYPOINTS_DIR, SAVANT_DIR, PROFILES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


def load_pitchers_config() -> List[Dict]:
    config_path = BASE_DIR / "pitchers.yaml"
    if not config_path.exists():
        logger.error(f"pitchers.yaml not found at {config_path}")
        return []
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data.get("pitchers", [])


# ─── Step 2: Baseball Savant ──────────────────────────────────────────────────

def fetch_savant_stats(pitcher_name: str, savant_id) -> Dict:
    """Pull pitch metrics from Baseball Savant via pybaseball and cache as JSON."""
    if savant_id is None:
        logger.info(f"[Savant] Skipping {pitcher_name} — no savant_id")
        return {}
    logger.info(f"[Savant] Fetching stats for {pitcher_name} (id={savant_id})")
    try:
        import pybaseball

        current_year = datetime.now().year
        df = None
        for year_offset in range(0, 3):
            end_year   = current_year - year_offset
            start_year = end_year - 1
            try:
                df = pybaseball.statcast_pitcher(
                    f"{start_year}-01-01", f"{end_year}-12-31",
                    player_id=savant_id,
                )
                if df is not None and not df.empty:
                    logger.info(f"[Savant] Got {len(df)} rows for {start_year}–{end_year}")
                    break
            except Exception:
                continue

        if df is None or df.empty:
            logger.error(f"[Savant] No Statcast data found for {pitcher_name}")
            return {}

        stats: Dict[str, Dict] = {}
        for pitch_type, group in df.groupby("pitch_type"):
            if not pitch_type or str(pitch_type) == "nan":
                continue

            def safe_mean(col):
                if col not in group.columns:
                    return None
                val = group[col].dropna()
                return round(float(val.mean()), 4) if len(val) else None

            pfx_x = safe_mean("pfx_x")
            pfx_z = safe_mean("pfx_z")
            entry = {
                "count":            int(len(group)),
                "avg_velocity":     safe_mean("release_speed"),
                "avg_spin_rate":    safe_mean("release_spin_rate"),
                "avg_extension":    safe_mean("release_extension"),
                "release_pos_x":    safe_mean("release_pos_x"),
                "release_pos_z":    safe_mean("release_pos_z"),
                "horizontal_break": round(pfx_x * 12, 2) if pfx_x is not None else None,
                "vertical_break":   round(pfx_z * 12, 2) if pfx_z is not None else None,
            }
            stats[str(pitch_type)] = {k: v for k, v in entry.items() if v is not None}

        result = {
            "pitcher_name":  pitcher_name,
            "savant_id":     savant_id,
            "total_pitches": int(len(df)),
            "pitch_stats":   stats,
            "fetched_at":    datetime.now().isoformat(),
        }

        out_path = SAVANT_DIR / f"{_safe_name(pitcher_name)}.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(f"[Savant] Saved {len(stats)} pitch types → {out_path.name}")
        return result

    except ImportError:
        logger.error("[Savant] pybaseball not installed. Run: pip install pybaseball")
        return {}
    except Exception as e:
        logger.error(f"[Savant] Failed for {pitcher_name}: {e}")
        return {}


# ─── Step 4: Derived biomechanics ────────────────────────────────────────────

def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle (degrees) at point b given three 2-D or 3-D points."""
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _compute_derived(frame_kps: np.ndarray, handedness: str = "R") -> Dict:
    """
    Compute per-frame biomechanical features.

    frame_kps : (12, 3) — [x, y, confidence], normalised [0,1]
    handedness: 'R' (RHP) or 'L' (LHP) — selects throwing arm
    """
    l_shoulder, r_shoulder = frame_kps[0], frame_kps[1]
    l_elbow,    r_elbow    = frame_kps[2], frame_kps[3]
    l_wrist,    r_wrist    = frame_kps[4], frame_kps[5]
    l_hip,      r_hip      = frame_kps[6], frame_kps[7]

    if handedness == "L":
        shoulder, elbow, wrist = l_shoulder, l_elbow, l_wrist
    else:
        shoulder, elbow, wrist = r_shoulder, r_elbow, r_wrist

    elbow_angle = _angle_between(shoulder, elbow, wrist)

    # Hip–shoulder separation (angle between axis vectors in xy plane)
    hip_vec      = r_hip[:2]      - l_hip[:2]
    shoulder_vec = r_shoulder[:2] - l_shoulder[:2]
    cos_sep = np.dot(hip_vec, shoulder_vec) / (
        np.linalg.norm(hip_vec) * np.linalg.norm(shoulder_vec) + 1e-8
    )
    hip_shoulder_sep = float(np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))))

    # Wrist height relative to hip midpoint (y; smaller y = higher in image)
    hip_y         = float((l_hip[1] + r_hip[1]) / 2)
    wrist_rel_hip = float(wrist[1] - hip_y)

    return {
        "elbow_angle":      round(elbow_angle,      3),
        "hip_shoulder_sep": round(hip_shoulder_sep, 3),
        "wrist_rel_hip":    round(wrist_rel_hip,    4),
    }


# ─── Step 4: Pose backends ────────────────────────────────────────────────────

def _detect_backend() -> str:
    """
    Auto-select pose backend in priority order:
      1. mmpose   — ViTPose-B, requires CUDA + mmpose installed (DGX)
      2. yolo     — YOLOv8x-pose, COCO-17 kps, GPU or CPU (local dev)
      3. mediapipe — CPU-only fallback
    """
    try:
        import torch
        import mmpose  # noqa: F401
        if torch.cuda.is_available():
            logger.info("[Pose] Backend: MMPose (CUDA)")
            return "mmpose"
    except ImportError:
        pass

    try:
        from ultralytics import YOLO  # noqa: F401
        logger.info("[Pose] Backend: YOLO-Pose (ultralytics)")
        return "yolo"
    except ImportError:
        pass

    logger.info("[Pose] Backend: MediaPipe (CPU fallback)")
    return "mediapipe"


def _open_video(
    video_path: Path,
    start_sec: Optional[float] = None,
    end_sec:   Optional[float] = None,
) -> Tuple[cv2.VideoCapture, float, int, int, int, int]:
    """
    Open a video and return (cap, fps, total_frames, frame_interval,
                              start_frame, end_frame).

    start_sec / end_sec: optional trim bounds in seconds.
    The caller should only read frames in [start_frame, end_frame).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps) if start_sec is not None else 0
    end_frame   = int(end_sec   * fps) if end_sec   is not None else total_frames
    end_frame   = min(end_frame, total_frames)

    frame_interval = max(1, round(fps / 30))   # sample at ~30 fps

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    active_frames = end_frame - start_frame
    return cap, fps, active_frames, frame_interval, start_frame, end_frame


# ── MMPose backend ────────────────────────────────────────────────────────────

def _run_mmpose(
    video_path: Path, handedness: str,
    start_sec: Optional[float] = None, end_sec: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Extract keypoints with MMPose ViTPose-B (GPU).
    Returns (keypoint_sequence, derived_sequence).
    """
    import torch
    from mmpose.apis import MMPoseInferencer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"[MMPose] Loading ViTPose-B on {device}")
    inferencer = MMPoseInferencer(pose2d="vitpose-b", device=device)

    cap, fps, active_frames, frame_interval, start_frame, end_frame = \
        _open_video(video_path, start_sec, end_sec)
    logger.info(
        f"[MMPose] {video_path.name}  "
        f"({active_frames} frames @ {fps:.1f} fps, interval={frame_interval}"
        + (f", trim={start_sec}s–{end_sec}s" if start_sec or end_sec else "") + ")"
    )

    keypoint_seq: List[np.ndarray] = []
    derived_seq:  List[Dict]       = []
    n_ok = n_fail = 0
    frame_idx = start_frame
    prev_cx = prev_cy = -1.0

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= end_frame:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        try:
            result_gen = inferencer(frame, show=False, return_vis=False)
            result     = next(result_gen)
            preds      = result.get("predictions", [[]])[0]

            if preds:
                # Build boxes from bbox field, use _select_subject
                boxes_np = np.array(
                    [[p["bbox"][0][0], p["bbox"][0][1],
                      p["bbox"][0][2], p["bbox"][0][3]] for p in preds],
                    dtype=np.float32,
                )
                best_idx = _select_subject(boxes_np, prev_cx, prev_cy,
                                           frame_w=w, frame_h=h)
                inst     = preds[best_idx]
                bx       = boxes_np[best_idx]
                prev_cx  = float((bx[0] + bx[2]) / 2 / w)
                prev_cy  = float((bx[1] + bx[3]) / 2 / h)

                kp_px  = np.array(inst["keypoints"],       dtype=np.float32)   # (17, 2)
                scores = np.array(inst["keypoint_scores"], dtype=np.float32)   # (17,)

                # Select our 12 landmarks from COCO-17, normalise xy to [0,1]
                kps = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
                for dst, src in enumerate(_COCO_INDICES):
                    kps[dst, 0] = kp_px[src, 0] / w       # x norm
                    kps[dst, 1] = kp_px[src, 1] / h       # y norm
                    kps[dst, 2] = scores[src]              # confidence

                keypoint_seq.append(kps)
                derived_seq.append(_compute_derived(kps, handedness))
                n_ok += 1
            else:
                n_fail += 1

        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                logger.debug(f"[MMPose] Frame {frame_idx} skipped: {e}")

        frame_idx += 1

    cap.release()
    logger.info(f"[MMPose] {video_path.name}: {n_ok} frames kept, {n_fail} skipped")
    return keypoint_seq, derived_seq


# ── YOLO-Pose backend (ultralytics, local GPU/CPU) ───────────────────────────

# YOLOv8 outputs COCO-17 keypoints; we map the same 12 body indices
_YOLO_BODY_INDICES = _COCO_INDICES   # [5..16]

def _select_subject(
    boxes_xyxy: np.ndarray,
    prev_cx: float,
    prev_cy: float,
    frame_w: int = 1,
    frame_h: int = 1,
) -> int:
    """
    Pick the correct person from multi-detection frames.

    Strategy (in priority order):
      1. Largest bounding-box area  — subject is always closer to camera
      2. Position continuity guard  — if the largest box has jumped >30% of
         frame width from the previous frame's centre, prefer the box whose
         centre is closest to the previous centre instead.

    boxes_xyxy : (N, 4) pixel coords  [x1, y1, x2, y2]
    prev_cx/cy : normalised [0,1] centre from last accepted frame
                 (-1 if no previous frame yet → use area only)
    frame_w/h  : actual frame pixel dimensions (must be passed for correct
                 normalisation — the old boxes_xyxy[:, 2].max() estimate
                 misfires when the pitcher strides to the far left and their
                 bbox right-edge becomes much smaller than the real frame width)
    Returns the chosen index into boxes_xyxy.
    """
    if len(boxes_xyxy) == 1:
        return 0

    areas  = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    cx     = ((boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2)
    cy     = ((boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2)
    best_area = int(areas.argmax())

    if prev_cx < 0:          # first frame — trust area alone
        return best_area

    # Frame-normalised centre of the largest-area box
    jump_x = abs(cx[best_area] / frame_w - prev_cx)

    if jump_x < 0.30:        # reasonable movement — accept the largest box
        return best_area

    # Large jump detected (follow-through clipping to background person).
    # Fall back to whoever is closest to the previous centre.
    dists = np.hypot(cx / frame_w - prev_cx, cy / frame_h - prev_cy)
    logger.debug(
        f"[YOLO] Subject jump {jump_x:.2f} > 0.30 — "
        f"using nearest-to-prev (dist={dists.min():.3f})"
    )
    return int(dists.argmin())


def _run_yolo(
    video_path: Path, handedness: str,
    start_sec: Optional[float] = None, end_sec: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Extract keypoints with YOLOv8x-pose (ultralytics).

    Subject selection per frame:
      - Primary:   largest bounding box (subject is always closer to camera)
      - Guard:     if box centre jumps >30% of frame width vs previous frame,
                   fall back to nearest-to-previous (catches follow-through
                   moments where a background person briefly clips in)

    Auto-uses GPU if available, falls back to CPU.
    Returns (keypoint_sequence, derived_sequence).
    """
    import torch
    from ultralytics import YOLO

    device = "0" if torch.cuda.is_available() else "cpu"
    logger.info(f"[YOLO] Loading yolov8x-pose on device={device!r}")
    model = YOLO("yolov8x-pose.pt")

    cap, fps, active_frames, frame_interval, start_frame, end_frame = \
        _open_video(video_path, start_sec, end_sec)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1
    logger.info(
        f"[YOLO] {video_path.name}  "
        f"({active_frames} frames @ {fps:.1f} fps, interval={frame_interval}"
        + (f", trim={start_sec}s–{end_sec}s" if start_sec or end_sec else "") + ")"
    )

    keypoint_seq: List[np.ndarray] = []
    derived_seq:  List[Dict]       = []
    n_ok = n_fail = 0
    frame_idx = start_frame
    prev_cx = prev_cy = -1.0   # normalised centre of last accepted detection

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= end_frame:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        try:
            results  = model(frame, verbose=False, device=device)
            r0       = results[0]
            kps_all  = r0.keypoints
            boxes    = r0.boxes

            if kps_all is not None and len(kps_all) > 0 and boxes is not None:
                boxes_np = boxes.xyxy.cpu().numpy()          # (N, 4)
                best_idx = _select_subject(boxes_np, prev_cx, prev_cy,
                                           frame_w=w, frame_h=h)

                xy   = kps_all.xy.cpu().numpy()[best_idx]    # (17, 2) pixels
                conf = kps_all.conf.cpu().numpy()[best_idx]  # (17,)

                # Update rolling centre (normalised)
                bx = boxes_np[best_idx]
                prev_cx = float((bx[0] + bx[2]) / 2 / w)
                prev_cy = float((bx[1] + bx[3]) / 2 / h)

                kps = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
                for dst, src in enumerate(_YOLO_BODY_INDICES):
                    kps[dst, 0] = xy[src, 0] / w
                    kps[dst, 1] = xy[src, 1] / h
                    kps[dst, 2] = conf[src]

                keypoint_seq.append(kps)
                derived_seq.append(_compute_derived(kps, handedness))
                n_ok += 1
            else:
                n_fail += 1

        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                logger.debug(f"[YOLO] Frame {frame_idx} skipped: {e}")

        frame_idx += 1

    cap.release()
    logger.info(f"[YOLO] {video_path.name}: {n_ok} frames kept, {n_fail} skipped")
    return keypoint_seq, derived_seq


# ── MediaPipe backend (fallback) ──────────────────────────────────────────────

def _ensure_mp_model() -> Path:
    if not _MP_MODEL_PATH.exists():
        logger.info(f"[MediaPipe] Downloading model → {_MP_MODEL_PATH.name} ...")
        urllib.request.urlretrieve(_MP_MODEL_URL, str(_MP_MODEL_PATH))
        logger.info("[MediaPipe] Model downloaded.")
    return _MP_MODEL_PATH


def _run_mediapipe(
    video_path: Path, handedness: str,
    start_sec: Optional[float] = None, end_sec: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[Dict]]:
    """
    Extract keypoints with MediaPipe PoseLandmarker (CPU).
    Returns (keypoint_sequence, derived_sequence).
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    model_path   = _ensure_mp_model()
    base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap, fps, active_frames, frame_interval, start_frame, end_frame = \
        _open_video(video_path, start_sec, end_sec)
    logger.info(
        f"[MediaPipe] {video_path.name}  "
        f"({active_frames} frames @ {fps:.1f} fps, interval={frame_interval}"
        + (f", trim={start_sec}s–{end_sec}s" if start_sec or end_sec else "") + ")"
    )

    keypoint_seq: List[np.ndarray] = []
    derived_seq:  List[Dict]       = []
    n_ok = n_fail = 0
    frame_idx = start_frame

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_idx >= end_frame:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            try:
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms    = int((frame_idx / fps) * 1000)
                result   = landmarker.detect_for_video(mp_image, ts_ms)

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    kps = np.array(
                        [
                            [lms[i].x, lms[i].y, lms[i].visibility]
                            for i in _MP_INDICES
                        ],
                        dtype=np.float32,
                    )
                    keypoint_seq.append(kps)
                    derived_seq.append(_compute_derived(kps, handedness))
                    n_ok += 1
                else:
                    n_fail += 1

            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    logger.debug(f"[MediaPipe] Frame {frame_idx} skipped: {e}")

            frame_idx += 1

    cap.release()
    logger.info(f"[MediaPipe] {video_path.name}: {n_ok} frames kept, {n_fail} skipped")
    return keypoint_seq, derived_seq


# ─── Step 4: Main video processor ────────────────────────────────────────────

def process_video(
    video_path: Path,
    pitcher_name: str,
    handedness:  str = "R",
    backend:     Optional[str]   = None,
    start_sec:   Optional[float] = None,
    end_sec:     Optional[float] = None,
) -> Optional[Path]:
    """
    Extract pose keypoints from one video, with cache.

    Cache: if keypoints/<pitcher>/<stem>_keypoints.npy already exists, load
    and return it immediately without reprocessing.

    start_sec / end_sec: trim the video to this window before processing.
    backend: 'mmpose' | 'yolo' | 'mediapipe' | None (auto-detect)
    """
    kp_dir   = KEYPOINTS_DIR / _safe_name(pitcher_name)
    npy_path = kp_dir / f"{video_path.stem}_keypoints.npy"

    # ── Cache hit ──────────────────────────────────────────────────────────────
    if npy_path.exists():
        logger.info(f"[Cache] Hit: {npy_path.name}  (skipping reprocessing)")
        return npy_path

    # ── Choose backend ─────────────────────────────────────────────────────────
    if backend is None:
        backend = _detect_backend()

    # ── Extract keypoints ──────────────────────────────────────────────────────
    keypoint_seq: List[np.ndarray] = []
    derived_seq:  List[Dict]       = []

    if backend == "mmpose":
        try:
            keypoint_seq, derived_seq = _run_mmpose(video_path, handedness, start_sec, end_sec)
        except Exception as e:
            logger.warning(f"[MMPose] Failed ({e}) — falling back to YOLO")
            backend = "yolo"

    if backend == "yolo":
        try:
            keypoint_seq, derived_seq = _run_yolo(video_path, handedness, start_sec, end_sec)
        except Exception as e:
            logger.warning(f"[YOLO] Failed ({e}) — falling back to MediaPipe")
            backend = "mediapipe"

    if backend == "mediapipe":
        try:
            keypoint_seq, derived_seq = _run_mediapipe(video_path, handedness, start_sec, end_sec)
        except Exception as e:
            logger.error(f"[MediaPipe] Failed: {e}")
            return None

    if not keypoint_seq:
        logger.warning(f"[Pose] No keypoints from {video_path.name} — skipping")
        return None

    # ── Save ───────────────────────────────────────────────────────────────────
    kp_dir.mkdir(parents=True, exist_ok=True)
    derived_path = kp_dir / f"{video_path.stem}_derived.json"

    kp_array = np.array(keypoint_seq, dtype=np.float32)
    np.save(str(npy_path), kp_array)
    derived_path.write_text(json.dumps(derived_seq, indent=2))

    logger.info(
        f"[Pose] Saved {npy_path.name}  shape={kp_array.shape}  "
        f"backend={backend}"
    )
    return npy_path


def process_all_videos(
    pitcher_name: str,
    handedness: str = "R",
    video_manifest: Optional[List[Dict]] = None,
    backend: Optional[str] = None,
) -> List[Path]:
    """
    Process every video for a pitcher.

    video_manifest: list of dicts from pitchers.yaml `videos:` section.
                    Each dict has at least `file` (relative to raw_video/).
                    If None, scans raw_video/<pitcher>/ recursively for *.mp4.
    """
    if video_manifest:
        video_paths = []
        for entry in video_manifest:
            rel = entry.get("file", "")
            p   = RAW_VIDEO_DIR / rel
            if p.exists():
                video_paths.append((p, entry.get("start_sec"), entry.get("end_sec")))
            else:
                logger.warning(f"[Pose] Video not found: {p}  (check pitchers.yaml)")
    else:
        video_dir = RAW_VIDEO_DIR / _safe_name(pitcher_name)
        if not video_dir.exists():
            logger.warning(f"[Pose] No video dir for {pitcher_name}")
            return []
        video_paths = [(v, None, None) for v in sorted(video_dir.rglob("*.mp4"))]

    if not video_paths:
        logger.warning(f"[Pose] No videos to process for {pitcher_name}")
        return []

    results = []
    for v, start_sec, end_sec in video_paths:
        r = process_video(
            v, pitcher_name,
            handedness=handedness,
            backend=backend,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if r:
            results.append(r)
    return results


# ─── Step 5: Profile aggregator ───────────────────────────────────────────────

def _resample_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample (T, 12, 3) → (target_len, 12, 3)."""
    T = seq.shape[0]
    if T == target_len:
        return seq
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    flat  = seq.reshape(T, -1)
    out   = np.zeros((target_len, flat.shape[1]), dtype=np.float32)
    for j in range(flat.shape[1]):
        out[:, j] = interp1d(x_old, flat[:, j], kind="linear")(x_new)
    return out.reshape(target_len, seq.shape[1], seq.shape[2])


def aggregate_pitcher_profile(
    pitcher_name: str,
    savant_id,
    handedness: str = "R",
) -> Optional[Dict]:
    """
    Aggregate all keypoint sequences for a pitcher:
      - Resample each to the median sequence length
      - Compute mean + std across videos (raw keypoints)
      - Compute biomechanical features per video, resample, mean + std
      - Merge with Savant stats
      - Write profiles/<pitcher>.json
    """
    logger.info(f"[Profile] Aggregating {pitcher_name}")

    # Local import to avoid a circular-ish dependency at module load.
    from features import FEATURE_NAMES, N_FEATURES, compute_features

    kp_dir    = KEYPOINTS_DIR / _safe_name(pitcher_name)
    npy_files = sorted(kp_dir.glob("*_keypoints.npy")) if kp_dir.exists() else []

    sequences = []
    for f in npy_files:
        try:
            seq = np.load(str(f))
            if seq.ndim == 3 and seq.shape[1:] == (N_LANDMARKS, 3):
                sequences.append(seq)
            else:
                logger.warning(f"[Profile] Unexpected shape {seq.shape} in {f.name} — skipping")
        except Exception as e:
            logger.warning(f"[Profile] Could not load {f.name}: {e}")

    kp_profile:  Dict = {}
    feat_profile: Dict = {}

    if sequences:
        lengths    = [s.shape[0] for s in sequences]
        target_len = int(np.median(lengths))

        # ─ Keypoint profile (raw, for visualization + legacy DTW mode) ─
        resampled  = [_resample_sequence(s, target_len) for s in sequences]
        stacked    = np.stack(resampled, axis=0)          # (N, T, 12, 3)

        mean_seq = stacked.mean(axis=0)                   # (T, 12, 3)
        std_seq  = stacked.std(axis=0)                    # (T, 12, 3)

        kp_profile = {
            "num_videos":      len(sequences),
            "sequence_length": target_len,
            "video_lengths":   lengths,
            "landmark_names":  LANDMARK_NAMES,
            "mean_keypoints":  mean_seq.tolist(),
            "std_keypoints":   std_seq.tolist(),
        }

        # ─ Feature profile: compute per video, resample, stack, mean/std ─
        try:
            feat_seqs = [compute_features(s, handedness) for s in sequences]
            # Resample each (T_i, F) → (target_len, F)
            resampled_feats = []
            for fs in feat_seqs:
                if fs.shape[0] == target_len:
                    resampled_feats.append(fs)
                else:
                    x_old = np.linspace(0, 1, fs.shape[0])
                    x_new = np.linspace(0, 1, target_len)
                    out = np.zeros((target_len, fs.shape[1]), dtype=np.float32)
                    for j in range(fs.shape[1]):
                        col = fs[:, j]
                        # Mask NaN before interpolation
                        valid = np.isfinite(col)
                        if valid.sum() < 2:
                            out[:, j] = np.nan
                        else:
                            out[:, j] = np.interp(x_new, x_old[valid], col[valid])
                    resampled_feats.append(out)

            stacked_feats = np.stack(resampled_feats, axis=0)  # (N, T, F)
            mean_feats = np.nanmean(stacked_feats, axis=0)
            std_feats  = np.nanstd(stacked_feats,  axis=0)

            feat_profile = {
                "num_videos":      len(feat_seqs),
                "sequence_length": target_len,
                "feature_names":   FEATURE_NAMES,
                "handedness":      handedness,
                "mean_features":   _nan_safe_list(mean_feats),
                "std_features":    _nan_safe_list(std_feats),
            }
            logger.info(
                f"[Profile] {len(sequences)} sequences → T={target_len}, "
                f"F={N_FEATURES} biomech features"
            )
        except Exception as e:
            logger.warning(f"[Profile] Feature computation failed: {e}")
    else:
        logger.warning(f"[Profile] No keypoint files for {pitcher_name}")

    savant_path  = SAVANT_DIR / f"{_safe_name(pitcher_name)}.json"
    savant_stats = {}
    if savant_path.exists():
        try:
            savant_stats = json.loads(savant_path.read_text())
        except Exception as e:
            logger.warning(f"[Profile] Could not load savant stats: {e}")
    else:
        logger.warning(f"[Profile] No savant stats for {pitcher_name}")

    profile = {
        "pitcher_name":     pitcher_name,
        "savant_id":        savant_id,
        "handedness":       handedness,
        "created_at":       datetime.now().isoformat(),
        "keypoint_profile": kp_profile,
        "feature_profile":  feat_profile,
        "savant_stats":     savant_stats,
    }

    out_path = PROFILES_DIR / f"{_safe_name(pitcher_name)}.json"
    out_path.write_text(json.dumps(profile, indent=2))
    logger.info(f"[Profile] Saved → {out_path.name}")
    return profile


def _nan_safe_list(arr: np.ndarray) -> List:
    """Convert a float array with NaN to nested lists, replacing NaN with None
    so JSON round-trips without `NaN` tokens (which are not valid JSON)."""
    if arr.dtype.kind not in "fc":
        return arr.tolist()
    out = arr.tolist()
    def replace(x):
        if isinstance(x, list):
            return [replace(v) for v in x]
        return None if isinstance(x, float) and (x != x) else x  # NaN != NaN
    return replace(out)
