# Offline analytics (before upstream PR)

Pipeline: **export → analyze → review → later PR to roboflow/sports**

## Step 1 — Export tracking (GPU)

```bash
cd /users/PAS2699/pratham2210/sports_cv

python export_tracking.py \
  --source_video_path roboflow-sports/examples/soccer/data/2e57b9_0.mp4 \
  --output_path outputs/2e57b9_0_tracking.jsonl \
  --device cuda
```

Each line after `meta` is one frame:

- `players[]`: track_id, team_id, pitch_xy_cm
- `ball`: detected, pitch_xy_cm, confidence

## Step 2 — Team shape (CPU, current JSONL)

```bash
python analyze_team_shape.py \
  --jsonl outputs/2e57b9_0_frames.jsonl \
  --out_dir outputs/2e57b9_0_shape
```

Outputs: `line_height_timeline.png`, `width_timeline.png`, `compactness_timeline.png`, `shape_snapshot_*.png`, `summary.json`.

## Step 3 — Heatmap offline replay (CPU)

```bash
python render_heatmap_offline.py \
  --jsonl outputs/2e57b9_0_frames.jsonl \
  --export_final_png outputs/2e57b9_0_occupancy_offline.png
```

Same occupancy map as online `--viz accumulated`, without re-running models.

## Step 4 — Other offline analysis (CPU)

```bash
python analyze_match.py \
  --jsonl outputs/2e57b9_0_tracking.jsonl \
  --out_dir outputs/2e57b9_0_analysis
```

Outputs:

| File | Content |
|------|---------|
| `summary.json` | Possession %, top distances/speeds |
| `possession_timeline.png` | Who had the ball over time |
| `top_speeds.png` | Max speed per track |
| `occupancy_from_export.png` | Final heatmap from export |

## Step 5 — PR to Roboflow (later)

Once numbers look reasonable on 2–3 demo clips:

1. PR1: RADAR + ball + bugfixes (`main.py`)
2. PR2: `export_tracking.py` + `sports/analytics/*`
3. PR3: `notebooks/match_analytics.ipynb` using same JSONL schema

## Optional next modules

- Pass detection (from possession switches)
- Homography smoothing before kinematics
- Pass network graph
