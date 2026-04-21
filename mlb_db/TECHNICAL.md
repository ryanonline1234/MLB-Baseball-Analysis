# Technical Documentation — MLB Pitcher Reference Database

---

## Architecture overview

```
pitchers.yaml
     │
     ├─ savant_id ──► Baseball Savant API ──► savant/{pitcher}.json
     │
     └─ videos[] ───► Pose backend ──────────► keypoints/{pitcher}/*.npy
                            │                            │
                            │                            ▼
                            └──────────────► profiles/{pitcher}.json
                                              (keypoints + savant merged)
```

Everything is file-based and idempotent. Re-running the pipeline skips any step whose output file already exists.

---

## Pipeline steps

### Step 1 — Configuration (`pitchers.yaml`)

Each pitcher entry defines:

| Field | Type | Description |
|---|---|---|
| `name` | string | Display name, used as dict key throughout |
| `savant_id` | int | Baseball Savant / MLB player ID |
| `handedness` | `R` or `L` | Throwing hand — controls which arm is analysed |
| `angle` | string | Camera angle convention (informational) |
| `videos[]` | list | Manually curated video manifest |
| `videos[].file` | string | Path relative to `raw_video/` |
| `videos[].pitch_type` | string | Savant pitch code (FF, SI, SL, etc.) |
| `videos[].start_sec` | float | Optional trim start (seconds) |
| `videos[].end_sec` | float | Optional trim end (seconds) |
| `videos[].notes` | string | Free-text context |

If `videos` is omitted or empty, the pipeline scans `raw_video/{safe_name}/` recursively for all `.mp4` files with no trimming applied.

---

### Step 2 — Baseball Savant scraper

**Source:** `pybaseball.statcast_pitcher()` — pulls pitch-by-pitch Statcast data.

**Retry logic:** tries current season, falls back up to 3 prior seasons until non-empty data is found. Handles mid-season gaps.

**Aggregation:** groups by `pitch_type`, computes means for:

| Field | Statcast column | Unit |
|---|---|---|
| `avg_velocity` | `release_speed` | mph |
| `avg_spin_rate` | `release_spin_rate` | rpm |
| `avg_extension` | `release_extension` | ft |
| `release_pos_x` | `release_pos_x` | ft (normalised) |
| `release_pos_z` | `release_pos_z` | ft (normalised) |
| `horizontal_break` | `pfx_x × 12` | inches |
| `vertical_break` | `pfx_z × 12` | inches |

Output: `savant/{safe_name}.json`

---

### Step 3 — Video curation

Videos are **manually sourced** side-profile clips. Convention:
- **RHP:** 3B-side camera (shows full arm action on the right)
- **LHP:** 1B-side camera (shows full arm action on the left)

Side profile is required for the mobile app comparison — user videos will also be filmed from the side.

Slow-motion footage is preferred and extracted via `start_sec` trim to exclude real-speed portions of the same clip, preventing the keypoint sequence from representing the same motion at inconsistent time scales.

---

### Step 4 — Pose extraction

#### Backend priority

```
CUDA + mmpose installed?  ──yes──► MMPose (ViTPose-B)
        │no
        ▼
ultralytics installed?    ──yes──► YOLO-Pose (YOLOv8x-pose)
        │no
        ▼
                                   MediaPipe (PoseLandmarker Full)
```

`--backend yolo|mmpose|mediapipe` overrides auto-detection.

#### Frame sampling

Source videos may be any frame rate. The pipeline samples at `interval = round(fps / 30)` to normalise output to ~30 fps regardless of source rate.

#### Keypoint format

All three backends output the same format:

```
shape : (T, 12, 3)  float32
axis 0: time (sampled frames)
axis 1: landmark index (see table below)
axis 2: [x, y, confidence]
          x, y  — normalised [0, 1] relative to frame dimensions
          conf  — model confidence / visibility score [0, 1]
```

| Index | Landmark | MediaPipe idx | COCO idx |
|---|---|---|---|
| 0 | left_shoulder | 11 | 5 |
| 1 | right_shoulder | 12 | 6 |
| 2 | left_elbow | 13 | 7 |
| 3 | right_elbow | 14 | 8 |
| 4 | left_wrist | 15 | 9 |
| 5 | right_wrist | 16 | 10 |
| 6 | left_hip | 23 | 11 |
| 7 | right_hip | 24 | 12 |
| 8 | left_knee | 25 | 13 |
| 9 | right_knee | 26 | 14 |
| 10 | left_ankle | 27 | 15 |
| 11 | right_ankle | 28 | 16 |

#### Subject selection (multi-person handling)

Side-profile baseball videos often contain background people (coaches, fielders, umpires). The pipeline resolves the correct subject each frame via `_select_subject()`:

1. **Primary:** pick the detection with the **largest bounding box area** — the pitcher is always closest to the camera
2. **Jump guard:** compute the normalised centre of the largest box; if it moved more than **30% of the frame width** from the previous frame's centre, fall back to the detection **closest to the previous centre** instead

This handles follow-through moments where a background person briefly becomes the largest detection as the pitcher rotates away from camera.

#### Derived biomechanics

Computed per frame and saved as `{stem}_derived.json` alongside the `.npy`:

| Feature | Description |
|---|---|
| `elbow_angle` | Interior angle at the elbow (shoulder–elbow–wrist), throwing arm |
| `hip_shoulder_sep` | Angle between hip axis and shoulder axis vectors (xy plane) — measures rotation separation |
| `wrist_rel_hip` | Wrist y-coordinate minus hip midpoint y-coordinate — negative = wrist above hip |

Throwing arm is selected by `handedness`: right arm for RHP, left arm for LHP.

---

### Step 5 — Profile aggregation

1. Load all `*_keypoints.npy` files for the pitcher
2. **Resample** each sequence to the median sequence length using `scipy.interpolate.interp1d` (linear, applied per landmark per axis)
3. **Stack** resampled sequences: `(N_videos, T, 12, 3)`
4. Compute **mean** and **std** across videos: `(T, 12, 3)` each
5. Merge with Savant stats JSON

Output: `profiles/{safe_name}.json`

```json
{
  "pitcher_name": "Paul Skenes",
  "savant_id": 694973,
  "created_at": "2026-04-17T10:25:25",
  "keypoint_profile": {
    "num_videos": 3,
    "sequence_length": 968,
    "video_lengths": [957, 926, 968],
    "landmark_names": ["left_shoulder", ...],
    "mean_keypoints": [...],   // (T, 12, 3) as nested list
    "std_keypoints":  [...]    // (T, 12, 3) as nested list
  },
  "savant_stats": {
    "total_pitches": 3733,
    "pitch_stats": {
      "FF": { "count": 1419, "avg_velocity": 98.06, "avg_spin_rate": 2259.3, ... },
      ...
    }
  }
}
```

The `std_keypoints` array encodes **mechanical consistency** — low std means the pitcher repeats the same position reliably at each phase of the delivery.

---

## Caching

Every expensive step checks for its output before running:

| Step | Cache check |
|---|---|
| Savant scrape | `savant/{name}.json` exists |
| Keypoint extraction | `keypoints/{name}/{stem}_keypoints.npy` exists |
| Profile aggregation | Runs every time (fast, ~100ms) |

Delete the relevant file to force a re-run of that step.

---

## DGX / GPU workflow

```bash
# 1. Push raw videos to DGX
./transfer.sh 192.168.1.100

# 2. On DGX — install GPU deps and run
pip install -r requirements-gpu.txt
# (follow mim install instructions in that file)
python run.py add-pitcher --backend mmpose

# 3. Pull processed keypoints back locally
./transfer.sh 192.168.1.100   # runs both directions
```

`transfer.sh` uses `rsync --checksum` — only transfers files that have actually changed. Resume-safe.

---

## Production architecture (planned)

When the mobile app is built, the local file system is replaced as follows:

| Local | Production |
|---|---|
| `raw_video/` | AWS S3 (user uploads ephemeral — deleted after processing) |
| `keypoints/*.npy` | S3 with pointer stored in PostgreSQL |
| `savant/*.json` | PostgreSQL (JSONB column) |
| `profiles/*.json` | Metadata in PostgreSQL; keypoint arrays in S3 |
| Similarity search | Precomputed embeddings in pgvector or Pinecone |
| Coaching output | Claude API (stateless — no storage, optionally cached in Postgres) |

### Similarity search pipeline (planned)

```
User video upload
      │
      ▼
Pose extraction (same YOLO/MMPose pipeline)
      │
      ▼
DTW comparison against all pitcher profiles
      │
      ▼
Ranked similarity results  (e.g. "you throw like Logan Webb — 91% match")
      │
      ▼
User selects idol pitcher
      │
      ▼
Claude API — keypoint delta + Savant diff → coaching plan
```

DTW (Dynamic Time Warping) is used for sequence comparison because it handles variable-length sequences and different delivery tempos naturally — a slower pitcher and a faster pitcher can still be mechanically similar.

---

## File naming convention

Pitcher names are normalised to `safe_name` for file system use:

```python
"Shohei Ohtani"  →  "shohei_ohtani"
"Jacob deGrom"   →  "jacob_degrom"
```

Rule: lowercase, spaces → underscores, apostrophes and dots removed.
