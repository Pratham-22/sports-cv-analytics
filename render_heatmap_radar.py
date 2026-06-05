#!/usr/bin/env python3
"""
RADAR-style video with dynamic minimap heatmaps:
  - Team accumulation grid (fills in over time)
  - Per-player moving footprint cells (3x3 patch, updates each frame)

Usage (on GPU node, from sports_cv/):
  python render_heatmap_radar.py \\
    --source_video_path roboflow-sports/examples/soccer/data/2e57b9_0.mp4 \\
    --target_video_path outputs/2e57b9_0_heatmap_radar.mp4 \\
    --device cuda

Heatmap-only video (no broadcast background):
  python render_heatmap_radar.py ... --minimap_only --h264_output outputs/heatmap_only_h264.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# Paths: local analytics + roboflow sports package
SPORTS_CV_ROOT = Path(__file__).resolve().parent
SPORTS_REPO = SPORTS_CV_ROOT / "roboflow-sports"
SOCCER_DIR = SPORTS_REPO / "examples" / "soccer"
sys.path.insert(0, str(SPORTS_REPO))
sys.path.insert(0, str(SPORTS_CV_ROOT))

from analytics.heatmap import (  # noqa: E402
    DynamicPitchHeatmap,
    render_heatmap_radar_panel,
    save_accumulated_heatmap_image,
)
from sports.annotators.soccer import draw_pitch  # noqa: E402
from sports.common.team import TeamClassifier  # noqa: E402
from sports.common.view import ViewTransformer  # noqa: E402
from sports.configs.soccer import SoccerPitchConfiguration  # noqa: E402

# Reuse soccer example constants
PLAYER_DETECTION_MODEL_PATH = str(SOCCER_DIR / "data/football-player-detection.pt")
PITCH_DETECTION_MODEL_PATH = str(SOCCER_DIR / "data/football-pitch-detection.pt")

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
STRIDE = 60
CONFIG = SoccerPitchConfiguration()
COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS), thickness=2
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def _team_centroid(players_xy, players_team_id, team_id: int) -> Optional[np.ndarray]:
    mask = players_team_id == team_id
    if not np.any(mask):
        return None
    return players_xy[mask].mean(axis=0)


def resolve_goalkeepers_team_id(players, players_team_id, goalkeepers) -> np.ndarray:
    if len(goalkeepers) == 0:
        return np.array([], dtype=int)
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    if len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0 = _team_centroid(players_xy, players_team_id, 0)
    team_1 = _team_centroid(players_xy, players_team_id, 1)
    out = []
    for gk_xy in goalkeepers_xy:
        if team_0 is None:
            out.append(1)
        elif team_1 is None:
            out.append(0)
        else:
            d0 = np.linalg.norm(gk_xy - team_0)
            d1 = np.linalg.norm(gk_xy - team_1)
            out.append(0 if d0 < d1 else 1)
    return np.array(out)


def has_track_ids(detections: sv.Detections) -> bool:
    return len(detections) > 0 and detections.tracker_id is not None


def pitch_transform(
    keypoints: sv.KeyPoints,
    pixel_xy: np.ndarray,
) -> Optional[np.ndarray]:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if mask.sum() < 4:
        return None
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
    )
    if len(pixel_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return transformer.transform_points(pixel_xy.astype(np.float32))


def minimap_frame_size(
    config: SoccerPitchConfiguration = CONFIG,
    scale: float = 0.1,
    padding: int = 50,
    upscale: float = 1.0,
) -> tuple[int, int]:
    """(width, height) in pixels for the drawn pitch panel."""
    w = int(config.length * scale) + 2 * padding
    h = int(config.width * scale) + 2 * padding
    return int(w * upscale), int(h * upscale)


def run_heatmap_radar(
    source_video_path: str,
    device: str,
    cell_size_cm: float,
    footprint_radius: int,
    export_jsonl: Optional[Path],
    minimap_only: bool = False,
    minimap_upscale: float = 2.0,
    viz: str = "full",
    heatmap_out: Optional[List] = None,
) -> Iterator[np.ndarray]:
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    frame_gen = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_gen, desc="collecting crops"):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        det = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, det[det.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    heatmap = DynamicPitchHeatmap(
        CONFIG,
        cell_size_cm=cell_size_cm,
        footprint_radius_cells=footprint_radius,
        team_colors=COLORS[:2],
    )
    if heatmap_out is not None:
        heatmap_out.clear()
        heatmap_out.append(heatmap)

    jsonl_file = None
    if export_jsonl is not None:
        export_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(export_jsonl, "w", encoding="utf-8")

    frame_gen = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    frame_idx = 0
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = video_info.fps or 25.0

    for frame in tqdm(frame_gen, desc="heatmap radar"):
        pitch_result = pitch_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
        player_result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        players_team_id = team_classifier.predict(get_crops(frame, players))
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        gk_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist()
            + gk_team_id.tolist()
            + [REFEREE_CLASS_ID] * len(referees)
        )

        pixel_xy = (
            detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            if len(detections) > 0
            else np.zeros((0, 2))
        )
        pitch_xy = pitch_transform(keypoints, pixel_xy) if len(pixel_xy) else None

        minimap = None
        if (
            has_track_ids(detections)
            and pitch_xy is not None
            and len(pitch_xy) == len(color_lookup)
        ):
            hm_mask = color_lookup != REFEREE_CLASS_ID
            base_pitch = draw_pitch(config=CONFIG)
            minimap = render_heatmap_radar_panel(
                base_pitch,
                pitch_xy[hm_mask],
                color_lookup[hm_mask],
                heatmap,
                dot_colors=COLORS[:2],
                viz=viz,
            )

        if minimap_only:
            out_w, out_h = minimap_frame_size(upscale=minimap_upscale)
            if minimap is not None:
                yield sv.resize_image(minimap, (out_w, out_h))
            else:
                yield sv.resize_image(draw_pitch(config=CONFIG), (out_w, out_h))
        else:
            if not has_track_ids(detections):
                yield frame.copy()
                frame_idx += 1
                continue

            labels = [str(t) for t in detections.tracker_id]
            annotated = frame.copy()
            annotated = ELLIPSE_ANNOTATOR.annotate(
                annotated, detections, custom_color_lookup=color_lookup
            )
            annotated = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated, detections, labels, custom_color_lookup=color_lookup
            )
            if minimap is not None:
                h, w, _ = annotated.shape
                minimap_resized = sv.resize_image(minimap, (w // 2, h // 2))
                mh, mw, _ = minimap_resized.shape
                rect = sv.Rect(x=w // 2 - mw // 2, y=h - mh, width=mw, height=mh)
                annotated = sv.draw_image(
                    annotated, minimap_resized, opacity=0.55, rect=rect
                )
            yield annotated

        if jsonl_file is not None and pitch_xy is not None and has_track_ids(detections):
            record = {
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "players": [],
            }
            for i in range(len(detections)):
                if color_lookup[i] == REFEREE_CLASS_ID:
                    continue
                record["players"].append({
                    "track_id": int(detections.tracker_id[i]),
                    "team_id": int(color_lookup[i]),
                    "pixel_xy": pixel_xy[i].tolist(),
                    "pitch_xy_cm": pitch_xy[i].tolist(),
                })
            jsonl_file.write(json.dumps(record) + "\n")

        frame_idx += 1

    if jsonl_file is not None:
        jsonl_file.close()


def convert_h264(src: Path, dst: Path) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise SystemExit("pip install imageio-ffmpeg for --h264_output") from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess

    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(src),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic heatmap + RADAR video")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument("--target_video_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--cell_size_cm",
        type=float,
        default=300.0,
        help="Grid cell size in cm (300 = 3m squares)",
    )
    parser.add_argument(
        "--footprint_radius",
        type=int,
        default=1,
        help="Footprint radius in cells (1 => 3x3 patch)",
    )
    parser.add_argument(
        "--export_jsonl",
        type=str,
        default=None,
        help="Optional path to write per-frame pitch positions",
    )
    parser.add_argument(
        "--h264_output",
        type=str,
        default=None,
        help="Optional second output path (H.264) for IDE/browser playback",
    )
    parser.add_argument(
        "--minimap_only",
        action="store_true",
        help="Output only the pitch heatmap panel (no broadcast video)",
    )
    parser.add_argument(
        "--minimap_upscale",
        type=float,
        default=2.0,
        help="Scale factor for minimap-only output resolution",
    )
    parser.add_argument(
        "--viz",
        type=str,
        choices=("accumulated", "full", "footprints"),
        default="accumulated",
        help="accumulated=time-in-cell heat only; full=all layers; footprints=no history",
    )
    parser.add_argument(
        "--export_final_png",
        type=str,
        default=None,
        help="Save end-of-match occupancy heatmap image (dominant team per cell)",
    )
    args = parser.parse_args()

    target = Path(args.target_video_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    export_path = Path(args.export_jsonl) if args.export_jsonl else None
    heatmap_holder: List = []
    generator = run_heatmap_radar(
        args.source_video_path,
        args.device,
        args.cell_size_cm,
        args.footprint_radius,
        export_path,
        minimap_only=args.minimap_only,
        minimap_upscale=args.minimap_upscale,
        viz=args.viz,
        heatmap_out=heatmap_holder,
    )

    source_info = sv.VideoInfo.from_video_path(args.source_video_path)
    if args.minimap_only:
        mw, mh = minimap_frame_size(upscale=args.minimap_upscale)
        video_info = sv.VideoInfo(
            width=mw,
            height=mh,
            fps=source_info.fps,
            total_frames=source_info.total_frames,
        )
    else:
        video_info = source_info

    with sv.VideoSink(str(target), video_info) as sink:
        for frame in generator:
            sink.write_frame(frame)

    print(f"Wrote {target}")
    if export_path:
        print(f"Wrote {export_path}")

    if args.export_final_png and heatmap_holder:
        png_path = Path(args.export_final_png)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        save_accumulated_heatmap_image(
            heatmap_holder[0],
            str(png_path),
            upscale=args.minimap_upscale,
        )
        print(f"Wrote {png_path}")

    if args.h264_output:
        convert_h264(target, Path(args.h264_output))
        print(f"Wrote {args.h264_output}")


if __name__ == "__main__":
    main()
