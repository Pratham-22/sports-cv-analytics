"""Heuristic possession from ball + player pitch positions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from analytics.load import TrackingExport


@dataclass
class PossessionEvent:
    frame: int
    time_s: float
    team_id: Optional[int]
    state: str  # confirmed | contested | loose
    holder_track_id: Optional[int] = None


@dataclass
class PossessionSummary:
    team_0_pct: float
    team_1_pct: float
    loose_pct: float
    timeline: List[PossessionEvent] = field(default_factory=list)


def _nearest_player(
    ball_xy: tuple[float, float],
    players,
    max_dist_cm: float,
) -> Optional[tuple[int, int]]:
    bx, by = ball_xy
    best = None
    best_d = max_dist_cm
    for p in players:
        d = np.hypot(p.pitch_xy_cm[0] - bx, p.pitch_xy_cm[1] - by)
        if d < best_d:
            best_d = d
            best = (p.track_id, p.team_id)
    return best


def compute_possession(
    export: TrackingExport,
    *,
    max_dist_cm: float = 200.0,
    confirm_frames: int = 8,
    release_dist_cm: float = 500.0,
) -> PossessionSummary:
    timeline: List[PossessionEvent] = []
    counts = {0: 0, 1: 0, "loose": 0}

    candidate_team: Optional[int] = None
    candidate_holder: Optional[int] = None
    candidate_streak = 0
    confirmed_team: Optional[int] = None
    confirmed_holder: Optional[int] = None

    for fr in export.frames:
        state = "loose"
        team_out: Optional[int] = None
        holder_out: Optional[int] = None

        if fr.ball.detected and fr.ball.pitch_xy_cm is not None:
            near = _nearest_player(fr.ball.pitch_xy_cm, fr.players, max_dist_cm)
            if near is not None:
                holder_out, team_cand = near
                if candidate_team == team_cand and candidate_holder == holder_out:
                    candidate_streak += 1
                else:
                    candidate_team = team_cand
                    candidate_holder = holder_out
                    candidate_streak = 1

                if candidate_streak >= confirm_frames:
                    confirmed_team = candidate_team
                    confirmed_holder = candidate_holder
                    state = "confirmed"
                    team_out = confirmed_team
                else:
                    state = "contested"
                    team_out = team_cand
            else:
                candidate_streak = 0
                if confirmed_team is not None and fr.ball.pitch_xy_cm is not None:
                    if fr.players:
                        holder_xy = None
                        for p in fr.players:
                            if p.track_id == confirmed_holder:
                                holder_xy = p.pitch_xy_cm
                                break
                        if holder_xy is not None:
                            d = np.hypot(
                                fr.ball.pitch_xy_cm[0] - holder_xy[0],
                                fr.ball.pitch_xy_cm[1] - holder_xy[1],
                            )
                            if d > release_dist_cm:
                                confirmed_team = None
                                confirmed_holder = None
                state = "loose" if confirmed_team is None else "confirmed"
                team_out = confirmed_team
        else:
            candidate_streak = 0
            state = "loose"

        if team_out in (0, 1):
            counts[team_out] += 1
        else:
            counts["loose"] += 1

        timeline.append(
            PossessionEvent(
                frame=fr.frame,
                time_s=fr.time_s,
                team_id=team_out,
                state=state,
                holder_track_id=holder_out if state != "loose" else confirmed_holder,
            )
        )

    total = sum(counts.values()) or 1
    return PossessionSummary(
        team_0_pct=100.0 * counts[0] / total,
        team_1_pct=100.0 * counts[1] / total,
        loose_pct=100.0 * counts["loose"] / total,
        timeline=timeline,
    )
