#!/usr/bin/env python3
"""
Export per-frame tracking data for offline analytics.

Output: JSONL where each line is one frame with players + ball in pitch coordinates (cm).

Example:
  python export_tracking.py \\
    --source_video_path roboflow-sports/examples/soccer/data/2e57b9_0.mp4 \\
    --output_path outputs/2e57b9_0_tracking.jsonl \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

SPORTS_CV_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SPORTS_CV_ROOT / "roboflow-sports"))
sys.path.insert(0, str(SPORTS_CV_ROOT))

from soccer_infer import (  # noqa: E402
    BALL_MODEL,
    GOALKEEPER_CLASS_ID,
    PITCH_MODEL,
    PLAYER_CLASS_ID,
    PLAYER_MODEL,
    REFEREE_CLASS_ID,
    STRIDE,
    build_transformer,
    create_ball_detector,
    detect_ball,
    get_crops,
    has_track_ids,
    resolve_goalkeepers_team_id,
    transform_xy,
)
from sports.common.team import TeamClassifier  # noqa: E402


def export_video(
    source_video_path: str,
    output_path: Path,
    device: str,
) -> None:
    player_model = YOLO(PLAYER_MODEL).to(device=device)
    pitch_model = YOLO(PITCH_MODEL).to(device=device)
    ball_model = YOLO(BALL_MODEL).to(device=device)
    ball_slicer, ball_tracker = create_ball_detector(ball_model)

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    fps = float(video_info.fps or 25.0)

    frame_gen = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_gen, desc="fit teams"):
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        det = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, det[det.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    with open(output_path, "w", encoding="utf-8") as out:
        meta = {
            "type": "meta",
            "source_video": str(source_video_path),
            "fps": fps,
            "total_frames": video_info.total_frames,
            "pitch_units": "cm",
        }
        out.write(json.dumps(meta) + "\n")

        frame_gen = sv.get_video_frames_generator(source_path=source_video_path)
        for frame_idx, frame in enumerate(tqdm(frame_gen, desc="export")):
            ball_det = detect_ball(frame, ball_slicer, ball_tracker)
            pitch_kp = sv.KeyPoints.from_ultralytics(
                pitch_model(frame, verbose=False)[0]
            )
            transformer = build_transformer(pitch_kp)

            player_det = sv.Detections.from_ultralytics(
                player_model(frame, imgsz=1280, verbose=False)[0]
            )
            player_det = tracker.update_with_detections(player_det)

            players = player_det[player_det.class_id == PLAYER_CLASS_ID]
            p_team = team_classifier.predict(get_crops(frame, players))
            gks = player_det[player_det.class_id == GOALKEEPER_CLASS_ID]
            gk_team = resolve_goalkeepers_team_id(players, p_team, gks)
            refs = player_det[player_det.class_id == REFEREE_CLASS_ID]
            merged = sv.Detections.merge([players, gks, refs])

            record = {
                "frame": frame_idx,
                "time_s": round(frame_idx / fps, 4),
                "players": [],
                "ball": {"detected": False},
            }

            if has_track_ids(merged):
                pixel_xy = merged.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_xy = transform_xy(transformer, pixel_xy)
                class_ids = merged.class_id
                team_ids = np.concatenate(
                    [p_team, gk_team, np.full(len(refs), REFEREE_CLASS_ID)]
                )
                if pitch_xy is not None:
                    for i in range(len(merged)):
                        if int(team_ids[i]) == REFEREE_CLASS_ID:
                            continue
                        record["players"].append({
                            "track_id": int(merged.tracker_id[i]),
                            "team_id": int(team_ids[i]),
                            "class_id": int(class_ids[i]),
                            "pixel_xy": pixel_xy[i].tolist(),
                            "pitch_xy_cm": pitch_xy[i].tolist(),
                        })

            if len(ball_det) > 0:
                ball_px = ball_det.get_anchors_coordinates(sv.Position.CENTER)
                ball_pitch = transform_xy(transformer, ball_px)
                conf = (
                    float(ball_det.confidence[0])
                    if ball_det.confidence is not None
                    else None
                )
                record["ball"] = {
                    "detected": True,
                    "pixel_xy": ball_px[0].tolist(),
                    "pitch_xy_cm": ball_pitch[0].tolist() if ball_pitch is not None else None,
                    "confidence": conf,
                }

            out.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tracking JSONL for offline analytics")
    parser.add_argument("--source_video_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    export_video(args.source_video_path, Path(args.output_path), args.device)
    print(f"Wrote {args.output_path}")


if __name__ == "__main__":
    main()
