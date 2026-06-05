"""Shared soccer inference helpers (Roboflow sports models)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv
from ultralytics import YOLO

from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.ball import BallTracker

SOCCER_DIR = Path(__file__).resolve().parent / "roboflow-sports" / "examples" / "soccer"
PLAYER_MODEL = str(SOCCER_DIR / "data/football-player-detection.pt")
PITCH_MODEL = str(SOCCER_DIR / "data/football-pitch-detection.pt")
BALL_MODEL = str(SOCCER_DIR / "data/football-ball-detection.pt")

PLAYER_CLASS_ID = 2
GOALKEEPER_CLASS_ID = 1
REFEREE_CLASS_ID = 3
STRIDE = 60

CONFIG = SoccerPitchConfiguration()


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
    gk_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    if len(players) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    c0 = _team_centroid(players_xy, players_team_id, 0)
    c1 = _team_centroid(players_xy, players_team_id, 1)
    out = []
    for xy in gk_xy:
        if c0 is None:
            out.append(1)
        elif c1 is None:
            out.append(0)
        else:
            out.append(0 if np.linalg.norm(xy - c0) < np.linalg.norm(xy - c1) else 1)
    return np.array(out)


def has_track_ids(detections: sv.Detections) -> bool:
    return len(detections) > 0 and detections.tracker_id is not None


def build_transformer(keypoints: sv.KeyPoints) -> Optional[ViewTransformer]:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if mask.sum() < 4:
        return None
    return ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32),
    )


def transform_xy(
    transformer: Optional[ViewTransformer],
    pixel_xy: np.ndarray,
) -> Optional[np.ndarray]:
    if transformer is None or len(pixel_xy) == 0:
        return None
    return transformer.transform_points(pixel_xy.astype(np.float32))


def create_ball_detector(model: YOLO) -> Tuple[sv.InferenceSlicer, BallTracker]:
    ball_tracker = BallTracker(buffer_size=20)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )
    return slicer, ball_tracker


def detect_ball(
    frame: np.ndarray,
    slicer: sv.InferenceSlicer,
    ball_tracker: BallTracker,
) -> sv.Detections:
    return ball_tracker.update(slicer(frame).with_nms(threshold=0.1))
