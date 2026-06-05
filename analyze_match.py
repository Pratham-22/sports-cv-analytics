#!/usr/bin/env python3
"""
Offline match analytics from export_tracking.py JSONL.

Produces summary JSON + matplotlib charts in an output directory.

Example:
  python analyze_match.py \\
    --jsonl outputs/2e57b9_0_tracking.jsonl \\
    --out_dir outputs/2e57b9_0_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SPORTS_CV_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPORTS_CV_ROOT / "roboflow-sports"))
sys.path.insert(0, str(SPORTS_CV_ROOT))

from analytics.heatmap import DynamicPitchHeatmap, save_accumulated_heatmap_image  # noqa: E402
from analytics.kinematics import compute_kinematics  # noqa: E402
from analytics.load import load_jsonl  # noqa: E402
from analytics.possession import compute_possession  # noqa: E402
from sports.configs.soccer import SoccerPitchConfiguration  # noqa: E402


def _plot_possession_timeline(summary, out_path: Path, fps: float) -> None:
    import matplotlib.pyplot as plt

    times = [e.time_s for e in summary.timeline]
    teams = [
        -1 if e.team_id is None else e.team_id for e in summary.timeline
    ]
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.step(times, teams, where="post", color="#333")
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["loose", "team 0", "team 1"])
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"Possession (heuristic) — team0 {summary.team_0_pct:.1f}% · "
        f"team1 {summary.team_1_pct:.1f}% · loose {summary.loose_pct:.1f}%"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_top_speeds(stats, out_path: Path, top_n: int = 12) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(stats.values(), key=lambda s: s.max_speed_kmh, reverse=True)[:top_n]
    if not rows:
        return
    labels = [f"T{s.track_id} (team {s.team_id})" for s in rows]
    speeds = [s.max_speed_kmh for s in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels[::-1], speeds[::-1], color="#00BFFF")
    ax.set_xlabel("Max speed (km/h)")
    ax.set_title("Top player speeds (smoothed pitch track)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_occupancy_png(export, out_path: Path) -> None:
    heatmap = DynamicPitchHeatmap(SoccerPitchConfiguration(), cell_size_cm=300.0)
    for fr in export.frames:
        if not fr.players:
            continue
        xy = [[p.pitch_xy_cm[0], p.pitch_xy_cm[1]] for p in fr.players]
        teams = [p.team_id for p in fr.players]
        import numpy as np
        heatmap.update(np.array(xy, dtype=np.float32), np.array(teams))
    save_accumulated_heatmap_image(heatmap, str(out_path), upscale=2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export = load_jsonl(args.jsonl)
    possession = compute_possession(export)
    kinematics = compute_kinematics(export)
    ball_frames = sum(1 for fr in export.frames if fr.ball.detected)

    summary = {
        "source_video": export.source_video,
        "fps": export.fps,
        "frames_analyzed": len(export.frames),
        "ball_frames": ball_frames,
        "notes": (
            "Re-run export_tracking.py for ball + possession metrics."
            if ball_frames == 0
            else None
        ),
        "possession_pct": {
            "team_0": round(possession.team_0_pct, 2),
            "team_1": round(possession.team_1_pct, 2),
            "loose": round(possession.loose_pct, 2),
        },
        "top_distance_m": sorted(
            [
                {
                    "track_id": s.track_id,
                    "team_id": s.team_id,
                    "distance_m": round(s.distance_m, 2),
                    "max_speed_kmh": round(s.max_speed_kmh, 2),
                }
                for s in kinematics.values()
            ],
            key=lambda x: x["distance_m"],
            reverse=True,
        )[:10],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _plot_possession_timeline(possession, out_dir / "possession_timeline.png", export.fps)
    _plot_top_speeds(kinematics, out_dir / "top_speeds.png")
    _build_occupancy_png(export, out_dir / "occupancy_from_export.png")

    print(f"Wrote analysis to {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
