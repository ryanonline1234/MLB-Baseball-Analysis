#!/usr/bin/env python3
"""
visualize.py — Overlay extracted keypoints onto a pitcher video.

Usage:
    python visualize.py --pitcher "Shohei Ohtani" [--video filename.mp4]
                        [--out output.mp4] [--fps 30]

Reads the .npy keypoint files from keypoints/<pitcher>/ and draws the
skeleton on top of the original video frames, writing a new annotated mp4.
"""

import argparse
import json
import sys
import numpy as np
import cv2
from pathlib import Path

BASE_DIR      = Path(__file__).parent
RAW_VIDEO_DIR = BASE_DIR / "raw_video"
KEYPOINTS_DIR = BASE_DIR / "keypoints"

LANDMARK_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
]

# Skeleton connections — (landmark_a_idx, landmark_b_idx)
SKELETON = [
    (0, 1),   # shoulder–shoulder
    (0, 2),   # L shoulder–L elbow
    (2, 4),   # L elbow–L wrist
    (1, 3),   # R shoulder–R elbow
    (3, 5),   # R elbow–R wrist
    (0, 6),   # L shoulder–L hip
    (1, 7),   # R shoulder–R hip
    (6, 7),   # hip–hip
    (6, 8),   # L hip–L knee
    (8, 10),  # L knee–L ankle
    (7, 9),   # R hip–R knee
    (9, 11),  # R knee–R ankle
]

# Colour palette (BGR)
COL_JOINT    = (0, 255, 128)    # bright green
COL_BONE_L   = (255, 140, 0)    # orange  (left side)
COL_BONE_R   = (0, 140, 255)    # blue    (right side)
COL_BONE_C   = (200, 200, 200)  # grey    (centre)
COL_TEXT     = (255, 255, 255)
COL_TEXT_BG  = (0, 0, 0)

# Which bones belong to which colour group
_LEFT_BONES   = {(0, 2), (2, 4), (0, 6), (6, 8), (8, 10)}
_RIGHT_BONES  = {(1, 3), (3, 5), (1, 7), (7, 9), (9, 11)}


def _safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


def draw_skeleton(
    frame: np.ndarray,
    kps: np.ndarray,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw skeleton on a copy of frame.
    kps: (12, 3) normalised [x, y, confidence]
    """
    out  = frame.copy()
    h, w = frame.shape[:2]

    # Pixel coords for each landmark
    pts = [(int(kps[i, 0] * w), int(kps[i, 1] * h)) for i in range(len(kps))]

    # Bones
    for a, b in SKELETON:
        if kps[a, 2] < conf_threshold or kps[b, 2] < conf_threshold:
            continue
        pair = (min(a, b), max(a, b))
        if pair in _LEFT_BONES:
            colour = COL_BONE_L
        elif pair in _RIGHT_BONES:
            colour = COL_BONE_R
        else:
            colour = COL_BONE_C
        cv2.line(out, pts[a], pts[b], colour, 2, cv2.LINE_AA)

    # Joints
    for i, (x, y) in enumerate(pts):
        if kps[i, 2] < conf_threshold:
            continue
        cv2.circle(out, (x, y), 5, COL_JOINT, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 5, (0, 0, 0), 1,  cv2.LINE_AA)   # outline

    return out


def _find_video(pitcher_name: str, stem_hint: str = "") -> Path:
    """Find the source video for a pitcher."""
    safe = _safe_name(pitcher_name)
    search_root = RAW_VIDEO_DIR / safe

    candidates = sorted(search_root.rglob("*.mp4"))
    if not candidates:
        sys.exit(f"No mp4 found under {search_root}")

    if stem_hint:
        for c in candidates:
            if stem_hint.lower() in c.stem.lower():
                return c
        sys.exit(f"No video matching '{stem_hint}' in {search_root}")

    if len(candidates) == 1:
        return candidates[0]

    print(f"Multiple videos for {pitcher_name}:")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c.name}")
    idx = int(input("Pick index: "))
    return candidates[idx]


def visualize(
    pitcher_name: str,
    video_hint: str = "",
    out_path: str = "",
    target_fps: int = 30,
    conf_threshold: float = 0.3,
):
    safe    = _safe_name(pitcher_name)
    kp_dir  = KEYPOINTS_DIR / safe

    # Match video to keypoint file
    video_path = _find_video(pitcher_name, video_hint)
    npy_path   = kp_dir / f"{video_path.stem}_keypoints.npy"

    if not npy_path.exists():
        sys.exit(
            f"No keypoints for {video_path.name}.\n"
            f"Run: python run.py add-pitcher --name '{pitcher_name}' first."
        )

    kp_seq = np.load(str(npy_path))   # (T, 12, 3)
    print(f"Loaded keypoints: {kp_seq.shape}  from {npy_path.name}")

    # Optionally load derived features for overlay
    derived_path = kp_dir / f"{video_path.stem}_derived.json"
    derived_seq  = []
    if derived_path.exists():
        with open(derived_path) as f:
            derived_seq = json.load(f)

    # Open video
    cap       = cv2.VideoCapture(str(video_path))
    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(1, round(src_fps / 30))   # same as pipeline

    if not out_path:
        out_path = str(BASE_DIR / f"{safe}_{video_path.stem}_skeleton.mp4")

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (src_w, src_h),
    )

    kp_idx    = 0   # index into kp_seq (one entry per sampled frame)
    frame_idx = 0
    written   = 0

    print(f"Rendering {n_frames} source frames → {out_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0 and kp_idx < len(kp_seq):
            kps        = kp_seq[kp_idx]
            annotated  = draw_skeleton(frame, kps, conf_threshold)

            # Overlay derived features (top-left HUD)
            if kp_idx < len(derived_seq):
                d = derived_seq[kp_idx]
                lines = [
                    f"Elbow angle:    {d.get('elbow_angle', 0):.1f}°",
                    f"Hip-shldr sep:  {d.get('hip_shoulder_sep', 0):.1f}°",
                    f"Wrist rel hip:  {d.get('wrist_rel_hip', 0):.3f}",
                ]
                for li, txt in enumerate(lines):
                    y = 28 + li * 22
                    cv2.rectangle(annotated, (8, y - 16), (290, y + 5), COL_TEXT_BG, -1)
                    cv2.putText(annotated, txt, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_TEXT, 1, cv2.LINE_AA)

            # Pitcher name watermark
            cv2.putText(annotated, pitcher_name, (10, src_h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_TEXT, 1, cv2.LINE_AA)

            writer.write(annotated)
            kp_idx += 1
            written += 1
        elif frame_idx % frame_interval != 0:
            pass   # skipped frame — don't advance kp_idx

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done. {written} frames written → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Visualize pitcher keypoints on video")
    parser.add_argument("--pitcher", required=True, help="Pitcher name (must match pitchers.yaml)")
    parser.add_argument("--video",   default="",   help="Partial video filename filter")
    parser.add_argument("--out",     default="",   help="Output mp4 path")
    parser.add_argument("--fps",     default=30, type=int, help="Output fps (default 30)")
    parser.add_argument("--conf",    default=0.3, type=float, help="Min keypoint confidence (default 0.3)")
    args = parser.parse_args()

    visualize(
        pitcher_name   = args.pitcher,
        video_hint     = args.video,
        out_path       = args.out,
        target_fps     = args.fps,
        conf_threshold = args.conf,
    )


if __name__ == "__main__":
    main()
