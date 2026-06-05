"""Team shape metrics from pitch-coordinate tracking (offline)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

from analytics.load import TrackingExport

Side = Literal["left", "right"]

PITCH_LENGTH_CM = 12_000.0
PITCH_WIDTH_CM = 7_000.0


def clip_pitch_xy(
    points: List[Tuple[float, float]],
    *,
    length_cm: float = PITCH_LENGTH_CM,
    width_cm: float = PITCH_WIDTH_CM,
) -> np.ndarray:
    """Clamp homography outliers to pitch bounds before hull / line / width."""
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.array(points, dtype=np.float64)
    arr[:, 0] = np.clip(arr[:, 0], 0.0, length_cm)
    arr[:, 1] = np.clip(arr[:, 1], 0.0, width_cm)
    return arr


@dataclass
class TeamShapeFrame:
    frame: int
    time_s: float
    team_id: int
    n_players: int
    line_height_m: float
    width_m: float
    compactness_m2: float
    centroid_x_m: float
    centroid_y_m: float


@dataclass
class TeamShapeSeries:
    defending_side: Dict[int, Side]
    frames: List[TeamShapeFrame] = field(default_factory=list)


@dataclass
class TeamShapeSummary:
    team_id: int
    defending_side: Side
    mean_line_height_m: float
    mean_width_m: float
    mean_compactness_m2: float
    std_line_height_m: float
    std_width_m: float


def _outfield_points(players, *, exclude_goalkeepers: bool) -> List[Tuple[float, float]]:
    pts = []
    for p in players:
        if p.team_id not in (0, 1):
            continue
        if exclude_goalkeepers and p.class_id == 1:
            continue
        pts.append(p.pitch_xy_cm)
    return pts


def infer_defending_sides(export: TrackingExport) -> Dict[int, Side]:
    """Team with lower median x defends the left goal (x=0). Uses clipped x."""
    by_team: Dict[int, List[float]] = {0: [], 1: []}
    for fr in export.frames:
        for p in fr.players:
            if p.team_id in (0, 1):
                x, _ = clip_pitch_xy([p.pitch_xy_cm])[0]
                by_team[p.team_id].append(float(x))
    medians = {t: float(np.median(xs)) if xs else 0.0 for t, xs in by_team.items()}
    if medians[0] <= medians[1]:
        return {0: "left", 1: "right"}
    return {0: "right", 1: "left"}


def _line_height_m(
    xs: np.ndarray,
    side: Side,
    pitch_length_cm: float,
) -> float:
    if len(xs) == 0:
        return float("nan")
    if side == "left":
        return float(np.min(xs)) / 100.0
    return float(pitch_length_cm - np.max(xs)) / 100.0


def _width_m(ys: np.ndarray) -> float:
    if len(ys) < 2:
        return 0.0
    return float(np.max(ys) - np.min(ys)) / 100.0


def _compactness_m2(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    arr = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(arr)
    return float(cv2.contourArea(hull)) / 10_000.0


def compute_team_shape_series(
    export: TrackingExport,
    *,
    pitch_length_cm: float = 12_000.0,
    min_players: int = 6,
    exclude_goalkeepers: bool = True,
    smooth_window: int = 7,
) -> TeamShapeSeries:
    sides = infer_defending_sides(export)
    raw: List[TeamShapeFrame] = []

    for fr in export.frames:
        for team_id in (0, 1):
            team_players = [p for p in fr.players if p.team_id == team_id]
            pts = _outfield_points(team_players, exclude_goalkeepers=exclude_goalkeepers)
            if len(pts) < min_players:
                continue
            clipped = clip_pitch_xy(pts, length_cm=pitch_length_cm, width_cm=PITCH_WIDTH_CM)
            xs = clipped[:, 0]
            ys = clipped[:, 1]
            side = sides[team_id]
            hull_pts = [tuple(row) for row in clipped]
            raw.append(
                TeamShapeFrame(
                    frame=fr.frame,
                    time_s=fr.time_s,
                    team_id=team_id,
                    n_players=len(pts),
                    line_height_m=_line_height_m(xs, side, pitch_length_cm),
                    width_m=_width_m(ys),
                    compactness_m2=_compactness_m2(hull_pts),
                    centroid_x_m=float(xs.mean()) / 100.0,
                    centroid_y_m=float(ys.mean()) / 100.0,
                )
            )

    if smooth_window < 2:
        return TeamShapeSeries(defending_side=sides, frames=raw)

    smoothed: List[TeamShapeFrame] = []
    for team_id in (0, 1):
        team_rows = [r for r in raw if r.team_id == team_id]
        if not team_rows:
            continue
        for i, row in enumerate(team_rows):
            sl = team_rows[max(0, i - smooth_window + 1) : i + 1]
            smoothed.append(
                TeamShapeFrame(
                    frame=row.frame,
                    time_s=row.time_s,
                    team_id=team_id,
                    n_players=row.n_players,
                    line_height_m=float(np.mean([s.line_height_m for s in sl])),
                    width_m=float(np.mean([s.width_m for s in sl])),
                    compactness_m2=float(np.mean([s.compactness_m2 for s in sl])),
                    centroid_x_m=row.centroid_x_m,
                    centroid_y_m=row.centroid_y_m,
                )
            )
    smoothed.sort(key=lambda r: (r.frame, r.team_id))
    return TeamShapeSeries(defending_side=sides, frames=smoothed)


def summarize_team_shape(series: TeamShapeSeries) -> List[TeamShapeSummary]:
    out: List[TeamShapeSummary] = []
    for team_id in (0, 1):
        rows = [r for r in series.frames if r.team_id == team_id]
        if not rows:
            continue
        lh = [r.line_height_m for r in rows]
        w = [r.width_m for r in rows]
        c = [r.compactness_m2 for r in rows]
        out.append(
            TeamShapeSummary(
                team_id=team_id,
                defending_side=series.defending_side[team_id],
                mean_line_height_m=float(np.mean(lh)),
                mean_width_m=float(np.mean(w)),
                mean_compactness_m2=float(np.mean(c)),
                std_line_height_m=float(np.std(lh)),
                std_width_m=float(np.std(w)),
            )
        )
    return out


def snapshot_frames(
    export: TrackingExport,
    n_snapshots: int = 3,
) -> List[int]:
    """Frame indices at ~25%, 50%, 75% of export."""
    if not export.frames:
        return []
    frames = [fr.frame for fr in export.frames]
    idxs = np.linspace(0, len(frames) - 1, n_snapshots, dtype=int)
    return [frames[int(i)] for i in idxs]
