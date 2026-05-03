#!/usr/bin/env python3
"""
MLB Pitcher Reference Database — CLI

Usage:
    python run.py add-pitcher [--name "Shohei Ohtani"] [--backend mediapipe|mmpose]
    python run.py list
    python run.py status
    python run.py compare [--name "Paul Skenes"] [--save]
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

        # Step 5 — Aggregate profile (keypoints + derived biomech features)
        aggregate_pitcher_profile(name, savant_id, handedness=handedness)

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


def cmd_compute_anchors(args):
    """Compute and persist fixed DTW anchor distances to profiles/anchors.json."""
    ensure_dirs()
    from similarity import compute_anchors
    result = compute_anchors()
    print()
    print("=" * 60)
    print("  ANCHOR DISTANCES COMPUTED")
    print("=" * 60)
    print(f"  overall  anchor_100 : {result['anchor_100_dist']:.6f}  ({result['anchor_100_source']})")
    print(f"  overall  anchor_0   : {result['anchor_0_dist']:.6f}  ({result['anchor_0_source']})")
    print(f"           worst pair : {result['anchor_0_pair'][0]} ↔ {result['anchor_0_pair'][1]}")
    print()
    print(f"  {'component':<18}  {'anchor_100':>12}  {'anchor_0':>10}  worst pair")
    print(f"  {'-'*18}  {'-'*12}  {'-'*10}  {'-'*30}")
    for g, cd in result.get("components", {}).items():
        wp = " ↔ ".join(cd["anchor_0_pair"])
        print(
            f"  {g:<18}  {cd['anchor_100_dist']:>12.6f}  "
            f"{cd['anchor_0_dist']:>10.6f}  {wp}"
        )
    print()
    print(f"  computed_at  : {result['computed_at']}")
    from pipeline import PROFILES_DIR
    print(f"  Saved →  {PROFILES_DIR / 'anchors.json'}")
    print()


def cmd_compare(args):
    """All-pairs DTW similarity across every pitcher profile."""
    ensure_dirs()
    from similarity import (
        compute_similarity_matrix,
        print_similarity_report,
        print_side_by_side,
    )

    pitchers = load_pitchers_config()
    if not pitchers:
        print("ERROR: No pitchers found in pitchers.yaml")
        sys.exit(1)

    mode = (args.mode or "features").lower()

    # Resolve --name partial match against config names (before running DTW
    # so the error happens before minutes of compute — not relevant at N=4
    # but cheap safety).
    filter_name = None
    if args.name:
        lower = args.name.lower()
        matches = [p["name"] for p in pitchers if lower in p["name"].lower()]
        if not matches:
            print(f"No pitcher matched '{args.name}'.")
            sys.exit(1)
        filter_name = matches[0]

    if mode == "both":
        feat_result = compute_similarity_matrix(pitchers, mode="features")
        kp_result   = compute_similarity_matrix(pitchers, mode="keypoints")
        print_similarity_report(feat_result, filter_name=filter_name)
        print_similarity_report(kp_result,   filter_name=filter_name)
        if filter_name is None:
            print_side_by_side(feat_result, kp_result)
        if args.save or filter_name is None:
            (PROFILES_DIR / "similarity_matrix_features.json").write_text(
                json.dumps(feat_result, indent=2))
            (PROFILES_DIR / "similarity_matrix_keypoints.json").write_text(
                json.dumps(kp_result, indent=2))
            print(f"Saved → {PROFILES_DIR}/similarity_matrix_{{features,keypoints}}.json")
        return

    result = compute_similarity_matrix(pitchers, mode=mode)
    if not result.get("pitchers"):
        print("No pitcher profiles to compare. Run: python run.py add-pitcher")
        return

    print_similarity_report(result, filter_name=filter_name)

    if args.save or filter_name is None:
        out = PROFILES_DIR / f"similarity_matrix_{mode}.json"
        out.write_text(json.dumps(result, indent=2))
        print(f"Saved → {out}")


def cmd_features(args):
    """Print the mean feature sequence for a pitcher as a phase-binned table."""
    ensure_dirs()
    import numpy as np
    from features import FEATURE_NAMES, N_FEATURES
    from similarity import load_pitcher_features

    pitchers = load_pitchers_config()
    matches = [p for p in pitchers if args.name.lower() in p["name"].lower()]
    if not matches:
        print(f"No pitcher matched '{args.name}'.")
        sys.exit(1)
    p = matches[0]
    name = p["name"]

    feats = load_pitcher_features(name)
    if feats is None:
        print(f"No feature profile for {name}. Run add-pitcher first.")
        sys.exit(1)

    T = feats.shape[0]
    n_phases = 6
    phase_labels = [
        "1. setup",
        "2. leg-lift peak",
        "3. cocking",
        "4. acceleration",
        "5. release",
        "6. follow-through",
    ]
    # Split time axis into 6 equal chunks and take the nan-mean of each.
    edges = np.linspace(0, T, n_phases + 1, dtype=int)
    phase_means = np.full((n_phases, N_FEATURES), np.nan, dtype=np.float32)
    for k in range(n_phases):
        chunk = feats[edges[k]:edges[k+1]]
        if chunk.size:
            phase_means[k] = np.nanmean(chunk, axis=0)

    print(f"\n{name}  ({p.get('handedness', 'R')})  — "
          f"{T} frames, 6-phase mean feature values")
    print()
    col_w = 10
    header = f"{'feature':<26}" + "".join(f"{lab.split('.')[0]:>{col_w}}" for lab in phase_labels)
    print(header)
    print("-" * len(header))
    for fi, fname in enumerate(FEATURE_NAMES):
        row = f"{fname:<26}"
        for pk in range(n_phases):
            v = phase_means[pk, fi]
            row += f"{'   —':>{col_w}}" if not np.isfinite(v) else f"{v:>{col_w}.2f}"
        print(row)
    print()
    print("Phase legend: " + "  ".join(phase_labels))
    print()


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
            "  python run.py compute-anchors                      # fix scoring scale\n"
            "  python run.py compare                              # all-pairs DTW\n"
            "  python run.py compare --name 'Skenes'              # single pitcher's rankings\n"
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
    sub.add_parser(
        "compute-anchors",
        help="Compute fixed DTW anchor distances and save to profiles/anchors.json",
    )

    cmp_p = sub.add_parser("compare", help="DTW similarity across all pitcher profiles")
    cmp_p.add_argument(
        "--name", type=str, default=None,
        help="Partial pitcher name — show only this pitcher's ranked list",
    )
    cmp_p.add_argument(
        "--mode", type=str,
        choices=["features", "keypoints", "both"], default="features",
        help="features: biomech + velocity (default). keypoints: legacy raw DTW. "
             "both: run both and print side-by-side A/B comparison.",
    )
    cmp_p.add_argument(
        "--save", action="store_true",
        help="Write profiles/similarity_matrix_<mode>.json (implicit when --name is omitted)",
    )

    feat_p = sub.add_parser("features", help="Print a pitcher's mean feature table")
    feat_p.add_argument("name", type=str, help="Partial pitcher name (required)")

    args = parser.parse_args()

    if args.command == "add-pitcher":
        cmd_add_pitcher(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "compute-anchors":
        cmd_compute_anchors(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "features":
        cmd_features(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
