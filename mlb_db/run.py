#!/usr/bin/env python3
"""
MLB Pitcher Reference Database — CLI

Usage:
    python run.py add-pitcher [--name "Shohei Ohtani"] [--backend mediapipe|mmpose]
    python run.py list
    python run.py status
"""

import argparse
import json
import sys
from pathlib import Path

from pipeline import (
    ensure_dirs,
    load_pitchers_config,
    fetch_savant_stats,
    process_all_videos,
    aggregate_pitcher_profile,
    PROFILES_DIR,
    KEYPOINTS_DIR,
    SAVANT_DIR,
    RAW_VIDEO_DIR,
    _safe_name,
)


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_add_pitcher(args):
    """Run the full pipeline for all pitchers (or one if --name is given)."""
    ensure_dirs()
    pitchers = load_pitchers_config()
    if not pitchers:
        print("ERROR: No pitchers found in pitchers.yaml")
        sys.exit(1)

    name_filter = args.name.lower() if args.name else None
    matched = [
        p for p in pitchers
        if name_filter is None or name_filter in p["name"].lower()
    ]

    if not matched:
        print(f"No pitcher matched '{args.name}'. Check pitchers.yaml.")
        sys.exit(1)

    backend = args.backend if hasattr(args, "backend") else None

    for p in matched:
        name       = p["name"]
        savant_id  = p["savant_id"]
        handedness = p.get("handedness", "R")
        videos     = p.get("videos")   # may be None → fall back to dir scan

        print(f"\n{'='*60}")
        print(f"  {name}  (Savant ID {savant_id}, throws {handedness})")
        print(f"{'='*60}")

        # Step 2 — Savant stats
        fetch_savant_stats(name, savant_id)

        # Step 4 — Keypoint extraction (cache-aware, auto backend)
        process_all_videos(
            name,
            handedness=handedness,
            video_manifest=videos,
            backend=backend,
        )

        # Step 5 — Aggregate profile
        aggregate_pitcher_profile(name, savant_id)

        print(f"\n  [Done] {name}")

    print("\nAll requested pitchers processed.")


def cmd_list(args):
    """Print a summary table of every pitcher profile in /profiles/."""
    ensure_dirs()
    profiles = sorted(PROFILES_DIR.glob("*.json"))

    if not profiles:
        print("No profiles found. Run: python run.py add-pitcher")
        return

    header = f"{'Pitcher':<26} {'Videos':>7} {'Frames':>7} {'Pitch Types':>12}  {'Updated'}"
    print(f"\n{header}")
    print("-" * len(header))

    for path in profiles:
        try:
            data    = json.loads(path.read_text())
            name    = data.get("pitcher_name", path.stem)
            kp      = data.get("keypoint_profile", {})
            sv      = data.get("savant_stats", {})
            videos  = kp.get("num_videos", 0)
            frames  = kp.get("sequence_length", 0)
            ptypes  = len(sv.get("pitch_stats", {}))
            updated = (data.get("created_at") or "")[:10]
            print(f"{name:<26} {videos:>7} {frames:>7} {ptypes:>12}  {updated}")
        except Exception:
            print(f"{path.stem:<26}  (error reading profile)")


def cmd_status(args):
    """Show which data components exist or are missing for each pitcher."""
    ensure_dirs()
    pitchers = load_pitchers_config()

    if not pitchers:
        print("No pitchers in pitchers.yaml")
        return

    header = (
        f"{'Pitcher':<26} {'Savant':^8} {'Videos':^8} "
        f"{'Keypoints':^10} {'Profile':^8}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    all_ok = True
    for p in pitchers:
        name = p["name"]
        safe = _safe_name(name)

        has_savant  = (SAVANT_DIR   / f"{safe}.json").exists()

        # Count videos: from manifest if defined, else scan dir recursively
        video_manifest = p.get("videos")
        if video_manifest:
            num_videos = sum(
                1 for v in video_manifest
                if (RAW_VIDEO_DIR / v.get("file", "")).exists()
            )
        else:
            video_dir  = RAW_VIDEO_DIR / safe
            num_videos = len(list(video_dir.rglob("*.mp4"))) if video_dir.exists() else 0

        kp_dir      = KEYPOINTS_DIR / safe
        num_kp      = len(list(kp_dir.glob("*_keypoints.npy"))) if kp_dir.exists() else 0
        has_profile = (PROFILES_DIR / f"{safe}.json").exists()

        def _fmt(ok, val=None):
            if val is not None:
                return str(val).center(8) if ok else "  0   ".center(8)
            return "  OK  ".center(8) if ok else "MISSING".center(8)

        incomplete = not has_savant or num_videos == 0 or num_kp == 0 or not has_profile
        if incomplete:
            all_ok = False

        flag = "  <-- INCOMPLETE" if incomplete else ""
        print(
            f"{name:<26} {_fmt(has_savant)} {_fmt(num_videos > 0, num_videos)} "
            f"{_fmt(num_kp > 0, num_kp)} {_fmt(has_profile)}{flag}"
        )

    print()
    if all_ok:
        print("All pitchers have complete data.")
    else:
        print("Run 'python run.py add-pitcher' to fill in missing data.")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="MLB Pitcher Reference Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run.py add-pitcher                          # all pitchers\n"
            "  python run.py add-pitcher --name 'Ohtani'         # single pitcher\n"
            "  python run.py add-pitcher --backend mmpose        # force MMPose\n"
            "  python run.py add-pitcher --backend mediapipe     # force MediaPipe\n"
            "  python run.py list\n"
            "  python run.py status\n"
        ),
    )
    sub = parser.add_subparsers(dest="command")

    add_p = sub.add_parser("add-pitcher", help="Run full pipeline")
    add_p.add_argument(
        "--name", type=str, default=None,
        help="Partial pitcher name filter (default: all pitchers)",
    )
    add_p.add_argument(
        "--backend", type=str, choices=["mmpose", "yolo", "mediapipe"], default=None,
        help="Force a specific pose backend (default: auto-detect)",
    )

    sub.add_parser("list",   help="List pitcher profiles")
    sub.add_parser("status", help="Show missing data")

    args = parser.parse_args()

    if args.command == "add-pitcher":
        cmd_add_pitcher(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
