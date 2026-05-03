# ROADMAP — MLB Pitcher Similarity Engine

Tracked items for future sessions. Each entry is self-contained so a new
session can pick it up without re-loading the full history. For engine
context see `TECHNICAL.md`; for the design decisions already implemented
see the docstrings in `similarity.py` and `features.py`.

Status legend:
- **Blocked** — waiting on another ROADMAP item or external data
- **Ready** — can be picked up whenever
- **Deferred** — intentionally postponed until DB reaches a scale threshold

---

## Engine TODOs (similarity + features)

### ROADMAP-001 — Fix 1: Self-anchored rescaling
Status: **Blocked on ROADMAP-007 (Skenes re-shoot)**

The current pipeline min-max normalizes distances within whatever pitchers
are in the DB, which means absolute scores shift as pitchers are added.
Cole↔deGrom was 93% at N=4 and 82% at N=7 — same underlying similarity,
different scale. Need two permanent anchors:

- **100% anchor** — split one pitcher's videos into halves A and B,
  compute `dtw_distance_masked(features(A), features(B))`. That's the
  "same delivery" ceiling cost for the weighted z-scored feature space.
- **0% anchor** — DTW between a pitching delivery and a non-pitching
  motion (golf swing, jumping jack, walking). Floor cost.

Calibrate once, store anchor distances in `profiles/anchors.json`, then
every future similarity score maps linearly into that fixed range
instead of shifting with DB composition.

**Do this once DB has ~10 clean pitchers.** Don't calibrate against a
corrupted DB — see ROADMAP-007.

Where the change lands: `similarity._minmax_similarity_matrix`, or a new
`_anchored_similarity_matrix` sibling function.

---

### ROADMAP-002 — Phase segmentation
Status: **Ready**

DTW currently aligns full deliveries end-to-end with no awareness of
canonical pitching phases. A bad warp can align leg-lift to arm-cock if
the cost works out — the math has no notion that these are different
biomechanical events.

Fix: segment every delivery into canonical phases using keypoint
heuristics, then run DTW within phases and sum phase costs.

Canonical phases:
```
windup
  → leg lift peak          max stride knee height
stride
  → foot plant             stride ankle velocity → 0
arm cock
  → max shoulder abduction (feature 2 peaks)
release
  → max wrist extension velocity
follow through
```

Benefits:
- Prevents nonsense alignments (no more leg-lift aligned to arm-cock)
- Enables phase-specific coaching feedback downstream
  ("your release phase is ~15° behind your idol's arm slot")
- Natural fit with ROADMAP-011's coaching prompt

Where the change lands: new `features.segment_phases(features) → dict[phase, slice]`
plus a `dtw_distance_phased` in `similarity.py`.

---

### ROADMAP-003 — Velocity weight tuning
Status: **Deferred until N ≥ 10 clean pitchers**

`features.VELOCITY_WEIGHT_FACTOR = 0.5` is a conservative starting
point, not a tuned value. Velocity channels inherit their base feature's
weight scaled by this factor.

Once DB has 10+ clean pitchers, A/B test by comparing full rank orderings
with `factor ∈ {0.0, 0.3, 0.5, 1.0}`. Metric: agreement with eye-test
priors (e.g. Cole↔deGrom ≈ 85%, Sale↔Cole ≤ 40%).

The specific channel to watch is **wrist velocity through release** —
it captures "whip" which separates power pitchers from finesse pitchers
even when their static poses look similar. If that single channel is
doing most of the work, consider breaking it out of the blanket factor.

Where the change lands: `features.VELOCITY_WEIGHT_FACTOR` and optionally
a per-feature velocity weight dict.

---

### ROADMAP-004 — Medoid or DBA instead of mean canonical delivery
Status: **Ready**

`pipeline.aggregate_pitcher_profile()` currently:
1. Resamples each video's `(T_i, 12, 3)` to `(median_T, 12, 3)`
2. Straight `mean(axis=0)` across videos → `mean_keypoints`

Same for `mean_features`. This washes out timing dynamics — two
deliveries with different tempos but the same average pose collapse to
identical canonical sequences.

Options:

- **Medoid** — pick the actual video closest to all others (by total DTW
  distance to the rest). Preserves real mechanics; no synthesis.
- **DBA (DTW Barycenter Averaging)** — `tslearn.barycenters.dtw_barycenter_averaging`
  computes a DTW-optimal average that respects temporal structure.

Implement both behind a `--aggregation {mean, medoid, dba}` flag on
`run.py add-pitcher`. A/B test on current DB once there are pitchers
with ≥3 videos each (currently only Ohtani/deGrom/Webb qualify).

Where the change lands: `pipeline.aggregate_pitcher_profile` + a new
flag on `run.py`.

---

### ROADMAP-005 — Fix 2 + Fix 5: z-score contrast and percentile ranking
Status: **Deferred until N > 20 pitchers**

Both require a large enough DB for their statistics to be meaningful.

- **Fix 2 (z-score contrast)** — "you're 2 SDs closer to Skenes than
  average." Not statistically meaningful at small N; becomes useful at
  scale when the null distribution of pairwise distances is populated.
- **Fix 5 (percentile ranking)** — "you're in the top 5% of similarity
  to Skenes across all pitchers we have." Requires a population big
  enough that top-5% is more than two pitchers.

Where the change lands: `similarity.compute_similarity_matrix` result
dict — add per-pair `z_score` and `percentile` fields.

---

## Data TODOs

### ROADMAP-006 — Glasnow re-clip
Status: **Ready**

Current clip: `raw_video/tyler_glasnow/Tyler-Glasnow-1st.mp4`, 192
frames (~6 sec after frame_interval sampling). Mostly setup / balance-
point; the delivery phase is compressed into a small fraction of the
sequence.

Symptom: Cole↔Glasnow `arm_action` scores **0%** — clearly wrong, these
are both overhead power RHPs with similar mechanics.

Need: 20+ sec clip (comparable to Kershaw's 1769 frames) OR a tight
trim (`start_sec` / `end_sec` in `pitchers.yaml`) around just the
delivery phase.

Expected outcome after fix: Cole↔Glasnow should land 70–85% overall
with strong `arm_action` score. Glasnow's #1 neighbor should be Cole
or deGrom, not Kershaw.

---

### ROADMAP-007 — Skenes side-profile re-shoot
Status: **Ready**

All Skenes pairs currently flagged `*` in `compare --mode features`
because the profile is ~100% NaN (front/back-angle source videos
produce unusable feature sequences — no meaningful 2D separation
between the throwing-side and glove-side landmarks in the camera
plane).

`pitchers.yaml` entry has `videos: []` which triggers directory-scan
fallback; the raw videos under `raw_video/paul_skenes/` are the
wrong-angle ones.

Every Skenes pair hits the NaN fallback branch inside
`dtw_distance_masked` → falls through to plain `dtw_distance` on
nan_to_num'd sequences → distances on a different numeric scale than
clean pairs. `_degraded_mask` excludes them from min-max endpoints
but their displayed % scores (often "100%" or "0%") are unreliable.

Need: at least one clean side-profile clip, ideally from the 3B side
(throwing-arm visible for a RHP).

Unlocks 6 clean pairs and pushes clean-DB count to 8 pitchers, closer
to the ROADMAP-001 threshold.

Expected: Skenes should cluster near Cole / deGrom (overhead power
RHP archetype).

---

### ROADMAP-008 — Add amateur/youth pitcher baseline
Status: **Ready — highest-impact data item**

The DB currently contains only MLB pitchers. Without a non-MLB baseline
there is no way to know where a real user's video will land on the
similarity scale — every pro pitcher is closer to every other pro
pitcher than to a typical amateur, but we have no data point showing
where that gap is.

This is the only way to validate the floor effect before building the
app side (ROADMAP-010). If an amateur video scores 75% similar to Cole,
the engine isn't actually discriminating archetype from proficiency.

Simplest sufficient data: **one side-profile video of anyone throwing**
— even the developer is enough to establish a rough lower bound.

Adds to `pitchers.yaml` as a regular entry (with a note in `notes`
marking it as baseline).

---

### ROADMAP-009 — Add camera-side validation pitcher
Status: **Ready**

An LHP cluster appeared at N=7: Ohtani ↔ Kershaw 78.6%, Sale ↔ Kershaw
77.3%, Ohtani ↔ Sale 60.5%. LHPs pull toward each other more than most
RHPs do.

This might be real handedness signal (surviving the x-flip mirror) or
might be a **camera-angle artifact** — all three LHPs are filmed from
the 1B side while most RHPs are filmed from the 3B side. Features 3, 4,
and 8 are camera-frame-relative (though 3 and 4 are weight-zeroed);
anything that doesn't cancel out between 3B and 1B camera positions
could be driving the cluster.

Falsification test: add **one RHP filmed from the 1B side** (same angle
as the LHPs).

- If that RHP clusters with the LHPs → the cluster is a camera artifact
- If not → it's real handedness signal

Candidate: any RHP with an existing 1B-side clip. Tyler Glasnow's
current clip is 1B-side (see ROADMAP-006 for the separate short-clip
problem) — once that's re-clipped, check whether Glasnow clusters with
the LHPs or with the RHPs.

---

## Product / App TODOs

### ROADMAP-010 — End-to-end user flow (first real milestone)
Status: **Blocked on ROADMAP-008 (baseline) for meaningful validation**

Before building any mobile app, get one complete user flow working on
the command line:

1. Developer films self throwing from side profile (~20 sec, 1080p)
2. Run through existing pipeline: YOLO pose → `compute_features` →
   profile JSON (reuse `process_video` + `aggregate_pitcher_profile`)
3. Run `compare --mode features --name "<dev name>"` → see ranked
   similarities against current DB
4. Generate Claude API coaching breakdown given:
   - user's feature profile
   - idol pitcher's feature profile (top match from step 3)
   - component deltas (arm_action / lower_body / rotation_timing /
     posture scores)
   - Savant stat deltas (velocity, spin, extension, release point,
     break) where the user has priors

This validates the end-to-end product loop before committing to a
mobile build. If the CLI version produces useful coaching output, the
mobile app is a wrapper; if it doesn't, mobile build wouldn't have
helped either.

---

### ROADMAP-011 — Claude API coaching prompt
Status: **Blocked on ROADMAP-010**

Given: user feature profile, idol pitcher feature profile, component
deltas (output of `compute_similarity_matrix`), Savant stat deltas.

Output: structured coaching plan with:
- Top 3 mechanical changes prioritized by impact
- Per change:
  - what to do
  - why it matters
  - a specific drill to work on it
- Realistic assessment of what's achievable vs fixed:
  - arm slot is hard to change
  - stride length is easier
  - hip-shoulder separation is trainable
- Injury risk flags (especially for younger pitchers — shoulder
  abduction extremes, elbow angles at release)

**Key requirement:** pass **pre-computed component deltas explicitly** —
don't dump two raw feature profiles and ask Claude to figure it out.
The specificity of
  "your arm_action score is 45% vs your idol's 92% — here are the exact
   joint angle differences per phase"
produces dramatically better coaching output than vague similarity.
ROADMAP-002 (phase segmentation) feeds directly into this — per-phase
deltas are the ideal input format.

Where the change lands: new `coaching.py` with a `generate_coaching_plan`
function that calls the Anthropic API with a structured prompt.

---

### ROADMAP-012 — Mobile app stack decision
Status: **Blocked on ROADMAP-010, ROADMAP-011**

Recommended stack when ready:

- **Frontend:** React Native (Expo)
  - `expo-camera` for side-profile filming with an angle guide overlay
  - `expo-file-system` for video upload
  - Polling pattern: `POST /analyze → job_id`, `GET /analyze/{job_id}`
    every 2s
- **Backend:** FastAPI on Render free tier (sufficient for dev + small
  alpha)
- **Inference:**
  - Dev/testing: DGX Spark as home inference server via Tailscale
  - Real users: RunPod / Vast.ai at $0.20–0.40/hr on-demand

Do **not** build this until ROADMAP-010 (CLI end-to-end) is working.
Building mobile before validating the loop is a waste of 4+ weeks.

---

### ROADMAP-013 — "Best model to work toward" retrieval
Status: **Deferred — research question first**

Product reframe: instead of
  "who do you already throw like"  (nearest-neighbor)
find
  "which pitcher is the best model for you to improve toward"

Query: close on **fixed** attributes (arm slot, body geometry,
handedness) AND **far** on **improvable** ones (hip-shoulder separation,
stride length, timing, trunk tilt).

Requires a biomechanical theory of what's fixed vs trainable — this is
a research question before it's an engineering one. Before
implementing, the feature list needs an explicit `FIXED_ATTRIBUTES`
vs `TRAINABLE_ATTRIBUTES` split with citations.

Do **not** build until:
- Component scores are validated at N > 15 (eye-test stable across
  rebuilds)
- ROADMAP-010 (CLI end-to-end) is done — can't evaluate the reframe
  without a user loop

Where the change lands: new `retrieval.py` with a `find_best_model`
function that scores pitchers by a weighted combination of closeness
on fixed attributes and distance on trainable ones.
