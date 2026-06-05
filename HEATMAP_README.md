# Dynamic heatmap minimap

Two layers on the bottom pitch panel:

1. **Team heatmap** — fixed grid; cells get brighter as players visit (accumulates over the clip).
2. **Footprints** — 3×3 cell patch under each player, moves every frame.

## Run (GPU node)

**Occupancy heatmap only** (time in each cell → color; no dots, no footprints):

```bash
cd /users/PAS2699/pratham2210/sports_cv

python render_heatmap_radar.py \
  --source_video_path roboflow-sports/examples/soccer/data/2e57b9_0.mp4 \
  --target_video_path outputs/2e57b9_0_occupancy.mp4 \
  --h264_output outputs/2e57b9_0_occupancy_h264.mp4 \
  --export_final_png outputs/2e57b9_0_occupancy_final.png \
  --minimap_only \
  --viz accumulated \
  --device cuda
```

The video shows the heat building over time; the PNG is the **final** full-match map (dominant team per cell).

**Heatmap + broadcast overlay** (original composite):

```bash
python render_heatmap_radar.py \
  --source_video_path roboflow-sports/examples/soccer/data/2e57b9_0.mp4 \
  --target_video_path outputs/2e57b9_0_heatmap_radar.mp4 \
  --h264_output outputs/2e57b9_0_heatmap_radar_h264.mp4 \
  --export_jsonl outputs/2e57b9_0_frames.jsonl \
  --device cuda
```

## Tuning

| Flag | Default | Meaning |
|------|---------|---------|
| `--cell_size_cm` | 300 | Grid cell size (~3 m) |
| `--footprint_radius` | 1 | 1 → 3×3 cells under each player |

Requires `pip install -e roboflow-sports` and soccer `data/*.pt` models.

## Offline heatmap (CPU, from JSONL)

After you have `frames.jsonl` from the online run (or `export_tracking.py`):

```bash
python render_heatmap_offline.py \
  --jsonl outputs/2e57b9_0_frames.jsonl \
  --export_final_png outputs/2e57b9_0_occupancy_offline.png \
  --viz accumulated

# Optional: replay heatmap-building video without GPU
python render_heatmap_offline.py \
  --jsonl outputs/2e57b9_0_frames.jsonl \
  --target_video_path outputs/2e57b9_0_occupancy_offline.mp4 \
  --viz accumulated --minimap_upscale 2
```

| Mode | Script | Needs GPU |
|------|--------|-----------|
| **Online** | `render_heatmap_radar.py` | Yes |
| **Offline** | `render_heatmap_offline.py` | No (JSONL only) |
