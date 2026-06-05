"""Speed and distance from pitch-coordinate tracks."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from analytics.load import TrackingExport


@dataclass
class PlayerMatchStats:
    track_id: int
    team_id: int
    distance_m: float
    max_speed_kmh: float
    avg_speed_kmh: float


def _smooth_positions(
    positions: List[tuple[int, float, float, float]],
    window: int = 5,
) -> List[tuple[int, float, float, float]]:
    """positions: (frame, time_s, x_cm, y_cm)"""
    if len(positions) < window:
        return positions
    out = []
    for i in range(len(positions)):
        sl = positions[max(0, i - window + 1) : i + 1]
        x = np.mean([p[2] for p in sl])
        y = np.mean([p[3] for p in sl])
        out.append((positions[i][0], positions[i][1], x, y))
    return out


def compute_kinematics(
    export: TrackingExport,
    smooth_window: int = 5,
    max_step_m: float = 0.6,
) -> Dict[int, PlayerMatchStats]:
    by_track: Dict[int, List[tuple[int, float, float, float, int]]] = defaultdict(list)
    for fr in export.frames:
        for p in fr.players:
            by_track[p.track_id].append(
                (fr.frame, fr.time_s, p.pitch_xy_cm[0], p.pitch_xy_cm[1], p.team_id)
            )

    stats: Dict[int, PlayerMatchStats] = {}
    for tid, pts in by_track.items():
        pts = _smooth_positions(
            [(a, b, c, d) for a, b, c, d, _ in pts],
            window=smooth_window,
        )
        team_id = by_track[tid][0][4]
        dist_m = 0.0
        speeds_kmh = []
        for i in range(1, len(pts)):
            dt = pts[i][1] - pts[i - 1][1]
            if dt <= 0:
                continue
            dx = (pts[i][2] - pts[i - 1][2]) / 100.0
            dy = (pts[i][3] - pts[i - 1][3]) / 100.0
            step_m = float(np.hypot(dx, dy))
            if step_m > max_step_m:
                continue
            dist_m += step_m
            speeds_kmh.append((step_m / dt) * 3.6)

        stats[tid] = PlayerMatchStats(
            track_id=tid,
            team_id=team_id,
            distance_m=dist_m,
            max_speed_kmh=max(speeds_kmh) if speeds_kmh else 0.0,
            avg_speed_kmh=float(np.mean(speeds_kmh)) if speeds_kmh else 0.0,
        )
    return stats
