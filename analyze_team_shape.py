#!/usr/bin/env python3
"""
Offline team shape analytics from JSONL (no GPU).

Metrics per team per frame (smoothed):
  - Defensive line height (m from own goal)
  - Width of play (m)
  - Compactness (convex hull area, m²)

Example:
  python analyze_team_shape.py \\
    --jsonl outputs/2e57b9_0_frames.jsonl \\
    --out_dir outputs/2e57b9_0_shape
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv

SPORTS_CV_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPORTS_CV_ROOT / "roboflow-sports"))
sys.path.insert(0, str(SPORTS_CV_ROOT))

from analytics.load import load_jsonl  # noqa: E402
from analytics.team_shape import (  # noqa: E402
    clip_pitch_xy,
    compute_team_shape_series,
    infer_defending_sides,
    snapshot_frames,
    summarize_team_shape,
)
from sports.annotators.soccer import draw_pitch  # noqa: E402
from sports.configs.soccer import SoccerPitchConfiguration  # noqa: E402

TEAM_COLORS = ("#FF1493", "#00BFFF")


def _plot_metric_timeline(series, metric: str, ylabel: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    for team_id, color, label in [
        (0, TEAM_COLORS[0], "team 0"),
        (1, TEAM_COLORS[1], "team 1"),
    ]:
        rows = [r for r in series.frames if r.team_id == team_id]
        if not rows:
            continue
        times = [r.time_s for r in rows]
        vals = [getattr(r, metric) for r in rows]
        ax.plot(times, vals, color=color, label=label, linewidth=1.2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _draw_shape_snapshot(
    export,
    frame_id: int,
    sides: dict,
    out_path: Path,
    *,
    exclude_goalkeepers: bool,
) -> None:
    config = SoccerPitchConfiguration()
    fr = next(f for f in export.frames if f.frame == frame_id)
    base = draw_pitch(config=config)
    scale, padding = 0.1, 50

    for team_id, hex_color in enumerate(TEAM_COLORS):
        pts = [
            p.pitch_xy_cm
            for p in fr.players
            if p.team_id == team_id
            and (not exclude_goalkeepers or p.class_id != 1)
        ]
        if len(pts) < 3:
            continue
        clipped = clip_pitch_xy(pts).astype(np.float32)
        hull = cv2.convexHull(clipped)
        hull_px = []
        for x_cm, y_cm in hull.reshape(-1, 2):
            x_px = int(x_cm * scale) + padding
            y_px = int(y_cm * scale) + padding
            hull_px.append([x_px, y_px])
        color_bgr = sv.Color.from_hex(hex_color).as_bgr()
        cv2.polylines(base, [np.array(hull_px)], True, color_bgr, 2)
        for x_cm, y_cm in clipped:
            x_px = int(x_cm * scale) + padding
            y_px = int(y_cm * scale) + padding
            cv2.circle(base, (x_px, y_px), 5, color_bgr, -1)

    side_txt = (
        f"T0 defends {sides[0]} goal | T1 defends {sides[1]} goal"
    )
    cv2.putText(
        base, side_txt, (padding, padding + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA,
    )
    cv2.putText(
        base, f"frame {frame_id}  t={fr.time_s:.2f}s", (padding, padding + 45),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Team shape analytics (offline JSONL)")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--smooth_window", type=int, default=7)
    parser.add_argument("--min_players", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export = load_jsonl(args.jsonl)
    series = compute_team_shape_series(
        export,
        min_players=args.min_players,
        smooth_window=args.smooth_window,
    )
    summary = summarize_team_shape(series)

    payload = {
        "source": export.source_video,
        "fps": export.fps,
        "frames_analyzed": len(export.frames),
        "defending_sides": series.defending_side,
        "summary": [
            {
                "team_id": s.team_id,
                "defends": s.defending_side,
                "mean_line_height_m": round(s.mean_line_height_m, 2),
                "mean_width_m": round(s.mean_width_m, 2),
                "mean_compactness_m2": round(s.mean_compactness_m2, 1),
                "std_line_height_m": round(s.std_line_height_m, 2),
                "std_width_m": round(s.std_width_m, 2),
            }
            for s in summary
        ],
        "notes": (
            "Line height = distance of deepest outfield player from own goal (m). "
            "Lower median-x team defends left goal (x=0)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    _plot_metric_timeline(
        series, "line_height_m", "Defensive line height (m from own goal)",
        out_dir / "line_height_timeline.png",
    )
    _plot_metric_timeline(
        series, "width_m", "Width of play (m)",
        out_dir / "width_timeline.png",
    )
    _plot_metric_timeline(
        series, "compactness_m2", "Compactness (hull area m²)",
        out_dir / "compactness_timeline.png",
    )

    sides = infer_defending_sides(export)
    for i, fid in enumerate(snapshot_frames(export, n_snapshots=3)):
        _draw_shape_snapshot(
            export, fid, sides, out_dir / f"shape_snapshot_{i + 1}_f{fid}.png",
            exclude_goalkeepers=True,
        )

    print(f"Wrote team shape analysis to {out_dir}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
