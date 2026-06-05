# Output artifacts

Generated under `sports_cv/outputs/`. Re-run commands in the [main README](../README.md) to regenerate.

## Naming

`<clip_id>_<suffix>` — e.g. `2e57b9_0` from `2e57b9_0.mp4`.

## Files

### `2e57b9_0_frames.jsonl`

Per-frame tracking log from **online** `render_heatmap_radar.py --export_jsonl`.

- **Use for:** all offline scripts (shape, heatmap replay, future possession).  
- **Players:** `track_id`, `team_id` (0 pink / 1 cyan in visuals), `pitch_xy_cm`.  
- **No ball** in heatmap-only exports — possession scripts need `export_tracking.py`.

### `2e57b9_0_shape/` — team shape (offline)

From `analyze_team_shape.py --jsonl ... --out_dir outputs/2e57b9_0_shape`.

| Artifact | Read this as |
|----------|----------------|
| `line_height_timeline.png` | Y-axis = meters from own goal to deepest outfield player. Up = higher defensive line. |
| `width_timeline.png` | Y-axis = meters between widest and narrowest player in y (touchline axis). |
| `compactness_timeline.png` | Y-axis = m² convex hull; lower = tighter shape. |
| `shape_snapshot_*.png` | Single-frame hull; clipped coords; legend shows which goal each team defends. |
| `summary.json` | Mean/std per team for the clip. |

### Occupancy heatmaps

| File | Source |
|------|--------|
| `*_occupancy_online.png` | GPU run, `--export_final_png` |
| `*_occupancy_offline.png` | `render_heatmap_offline.py` from JSONL |

Should agree if JSONL came from the same video pass. Darker/brighter cells = more time in that zone; color = dominant team in cell.

### Videos (optional, large)

| Pattern | Content |
|---------|---------|
| `*_radar_ball.mp4` | RADAR minimap + ball (`main.py`) |
| `*_heatmap*.mp4` | Heatmap building over time |
| `*_h264.mp4` | Browser-friendly re-encode |

Not committed to git by default (see root `.gitignore`).

## Example numbers (`2e57b9_0_shape/summary.json`)

- **Team 0** (left goal): line ~34 m, width ~42 m  
- **Team 1** (right goal): line ~21 m, width ~44 m  

Team 1 sits deeper on average; both use most of the pitch width (~60% of 70 m).
