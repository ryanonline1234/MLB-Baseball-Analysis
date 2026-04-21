# MLB Baseball Analysis

A computer vision pipeline that builds a biomechanical reference database of MLB pitchers, combining skeletal keypoint sequences extracted from video with pitch metrics from Baseball Savant.

The end goal is a mobile app where a user films their own pitching mechanics, and the system tells them which MLB pitcher they most resemble — then uses Claude to generate a personalized coaching breakdown of what to change to pitch more like their chosen idol.

---

## What it does

For each pitcher in the database the pipeline:

1. **Scrapes Baseball Savant** — pulls pitch-by-pitch Statcast data (velocity, spin rate, extension, release point, movement) and aggregates by pitch type
2. **Extracts skeletal keypoints** from curated side-profile video using YOLO-Pose (local) or MMPose ViTPose-B (GPU/DGX), with MediaPipe as a final fallback
3. **Saves keypoint sequences** as `(T, 12, 3)` NumPy arrays — 12 body landmarks × (x, y, confidence) per frame
4. **Builds a pitcher profile** combining the mean keypoint sequence across all videos with the full Savant stat block

---

## Current database

| Pitcher | Throws | Videos | Frames | Pitch Types |
|---|---|---|---|---|
| Shohei Ohtani | L | 1 | 957 | 7 |
| Jacob deGrom | R | 1 | 711 | 5 |
| Logan Webb | R | 1 | 893 | 5 |
| Paul Skenes | R | 3 | 968 | 7 |

---

## Quick start

```bash
# 1. System dependency
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Ubuntu

# 2. Python environment
cd mlb_db
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run the pipeline
python run.py add-pitcher               # all pitchers in pitchers.yaml
python run.py add-pitcher --name Skenes # single pitcher
python run.py list                      # summary table
python run.py status                    # show missing data
```

GPU (DGX) — see `requirements-gpu.txt` for the MMPose install and `transfer.sh` for syncing videos to the DGX.

---

## CLI reference

```
python run.py add-pitcher [--name NAME] [--backend yolo|mmpose|mediapipe]
python run.py list
python run.py status
```

`--backend` defaults to auto-detect: MMPose if CUDA + mmpose are available, YOLO otherwise, MediaPipe as final fallback.

---

## Adding a pitcher

1. Add an entry to `pitchers.yaml`:

```yaml
- name: "Gerrit Cole"
  savant_id: 543037
  handedness: R
  angle: side
  videos:
    - file: "gerrit_cole/cole-ff-2023-3b.mp4"
      pitch_type: FF
      angle: 3B
      start_sec: 15        # optional trim — cuts real-speed section
      notes: "2023 4-seam, 3B side slow motion"
```

2. Drop the video into `raw_video/gerrit_cole/`
3. `python run.py add-pitcher --name "Gerrit Cole"`

---

## File structure

```
mlb_db/
  pipeline.py          — core pipeline (Savant scraper, pose extraction, aggregator)
  run.py               — CLI
  visualize.py         — renders skeleton overlay video with biomechanics HUD
  transfer.sh          — rsync raw_video/ to DGX, pull keypoints/ back
  pitchers.yaml        — pitcher roster and video manifest
  requirements.txt     — CPU / local dev dependencies
  requirements-gpu.txt — MMPose / DGX install instructions
  savant/              — per-pitcher Savant JSON (committed as seed data)
  profiles/            — final merged profiles (committed as seed data)
  raw_video/           — source mp4s (gitignored, store locally or on S3)
  keypoints/           — .npy keypoint arrays (gitignored, reproducible)
```

---

## Roadmap

- [ ] DTW similarity engine — compare user video against pitcher database
- [ ] REST API — accepts uploaded video, returns similarity ranking
- [ ] Mobile app — side-profile recording, upload, results
- [ ] Claude coaching layer — keypoint delta + Savant diff → personalised drill breakdown
- [ ] Expand database — target 30+ pitchers across velocity/movement archetypes
