#!/usr/bin/env python3
"""
Offline heatmap from JSONL (no GPU, no YOLO).

Rebuilds the same occupancy / full minimap heatmap by replaying exported pitch positions.
Use after render_heatmap_radar.py --export_jsonl (online) or export_tracking.py.

Example — final PNG only:
  python render_heatmap_offline.py \\
    --jsonl outputs/2e57b9_0_frames.jsonl \\
    --export_final_png outputs/2e57b9_0_occupancy_offline.png \\
    --viz accumulated

Example — heatmap-building video (CPU):
  python render_heatmap_offline.py \\
    --jsonl outputs/2e57b9_0_frames.jsonl \\
    --target_video_path outputs/2e57b9_0_occupancy_offline.mp4 \\
    --viz accumulated --minimap_upscale 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import supervision as sv

SPORTS_CV_ROOT = Path(__file__).resolve().parent
SPORTS_REPO = SPORTS_CV_ROOT / "roboflow-sports"
sys.path.insert(0, str(SPORTS_REPO))
sys.path.insert(0, str(SPORTS_CV_ROOT))

from analytics.heatmap import (  # noqa: E402
    DynamicPitchHeatmap,
    render_heatmap_radar_panel,
    save_accumulated_heatmap_image,
)
from analytics.load import load_jsonl  # noqa: E402
from analytics.team_shape import clip_pitch_xy  # noqa: E402
from sports.annotators.soccer import draw_pitch  # noqa: E402
from sports.configs.soccer import SoccerPitchConfiguration  # noqa: E402

CONFIG = SoccerPitchConfiguration()
COLORS = ["#FF1493", "#00BFFF"]


def minimap_frame_size(
    scale: float = 0.1,
    padding: int = 50,
    upscale: float = 1.0,
) -> Tuple[int, int]:
    w = int(CONFIG.length * scale) + 2 * padding
    h = int(CONFIG.width * scale) + 2 * padding
    return int(w * upscale), int(h * upscale)


def frames_from_export(export, *, skip_empty: bool = True):
    for fr in export.frames:
        if not fr.players:
            if skip_empty:
                continue
            yield fr, np.zeros((0, 2)), np.array([], dtype=int)
            continue
        xy = clip_pitch_xy([p.pitch_xy_cm for p in fr.players]).astype(np.float32)
        teams = np.array([p.team_id for p in fr.players], dtype=int)
        mask = teams != 3
        yield fr, xy[mask], teams[mask]


def render_offline_heatmap(
    jsonl_path: str | Path,
    *,
    cell_size_cm: float = 300.0,
    footprint_radius: int = 1,
    viz: str = "accumulated",
    minimap_upscale: float = 2.0,
) -> Tuple[DynamicPitchHeatmap, List[np.ndarray]]:
    export = load_jsonl(jsonl_path)
    heatmap = DynamicPitchHeatmap(
        CONFIG,
        cell_size_cm=cell_size_cm,
        footprint_radius_cells=footprint_radius,
        team_colors=COLORS[:2],
    )
    panels: List[np.ndarray] = []
    out_w, out_h = minimap_frame_size(upscale=minimap_upscale)

    for fr, pitch_xy, team_ids in frames_from_export(export):
        if len(pitch_xy) == 0:
            base = draw_pitch(config=CONFIG)
            panel = heatmap.render_accumulated(base) if viz == "accumulated" else base
        else:
            base = draw_pitch(config=CONFIG)
            panel = render_heatmap_radar_panel(
                base,
                pitch_xy,
                team_ids,
                heatmap,
                dot_colors=COLORS,
                viz=viz,
            )
        panels.append(sv.resize_image(panel, (out_w, out_h)))

    return heatmap, panels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay heatmap from JSONL (offline, CPU only)",
    )
    parser.add_argument("--jsonl", required=True)
    parser.add_argument(
        "--target_video_path",
        type=str,
        default=None,
        help="Optional MP4: heatmap grows frame-by-frame",
    )
    parser.add_argument("--export_final_png", type=str, default=None)
    parser.add_argument("--cell_size_cm", type=float, default=300.0)
    parser.add_argument("--footprint_radius", type=int, default=1)
    parser.add_argument(
        "--viz",
        choices=("accumulated", "full", "footprints"),
        default="accumulated",
    )
    parser.add_argument("--minimap_upscale", type=float, default=2.0)
    parser.add_argument(
        "--h264_output",
        type=str,
        default=None,
        help="Re-encode video with libx264 (needs imageio-ffmpeg)",
    )
    args = parser.parse_args()

    heatmap, panels = render_offline_heatmap(
        args.jsonl,
        cell_size_cm=args.cell_size_cm,
        footprint_radius=args.footprint_radius,
        viz=args.viz,
        minimap_upscale=args.minimap_upscale,
    )

    if not panels:
        raise SystemExit("No frames with players in JSONL")

    if args.export_final_png:
        png = Path(args.export_final_png)
        png.parent.mkdir(parents=True, exist_ok=True)
        save_accumulated_heatmap_image(
            heatmap, str(png), upscale=args.minimap_upscale,
        )
        print(f"Wrote {png}")

    if args.target_video_path:
        export = load_jsonl(args.jsonl)
        fps = export.fps or 25.0
        w, h = panels[0].shape[1], panels[0].shape[0]
        video_info = sv.VideoInfo(width=w, height=h, fps=fps, total_frames=len(panels))
        target = Path(args.target_video_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with sv.VideoSink(str(target), video_info) as sink:
            for panel in panels:
                sink.write_frame(panel)
        print(f"Wrote {target}")

        if args.h264_output:
            from render_heatmap_radar import convert_h264
            convert_h264(target, Path(args.h264_output))
            print(f"Wrote {args.h264_output}")


if __name__ == "__main__":
    main()
