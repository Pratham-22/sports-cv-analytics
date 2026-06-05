"""Load tracking JSONL exports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass
class PlayerFrame:
    track_id: int
    team_id: int
    class_id: int
    pitch_xy_cm: tuple[float, float]


@dataclass
class BallFrame:
    detected: bool
    pitch_xy_cm: Optional[tuple[float, float]] = None
    confidence: Optional[float] = None


@dataclass
class FrameRecord:
    frame: int
    time_s: float
    players: List[PlayerFrame] = field(default_factory=list)
    ball: BallFrame = field(default_factory=lambda: BallFrame(detected=False))


@dataclass
class TrackingExport:
    source_video: str
    fps: float
    total_frames: int
    frames: List[FrameRecord]


def load_jsonl(path: str | Path, *, default_fps: float = 25.0) -> TrackingExport:
    path = Path(path)
    meta = {}
    frames: List[FrameRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("type") == "meta":
                meta = row
                continue
            if "frame" not in row:
                continue
            players = [
                PlayerFrame(
                    track_id=p["track_id"],
                    team_id=p["team_id"],
                    class_id=int(p.get("class_id", 2)),
                    pitch_xy_cm=(p["pitch_xy_cm"][0], p["pitch_xy_cm"][1]),
                )
                for p in row.get("players", [])
            ]
            b = row.get("ball", {})
            ball = BallFrame(
                detected=bool(b.get("detected", False)),
                pitch_xy_cm=(
                    (b["pitch_xy_cm"][0], b["pitch_xy_cm"][1])
                    if b.get("pitch_xy_cm")
                    else None
                ),
                confidence=b.get("confidence"),
            )
            frames.append(
                FrameRecord(
                    frame=row["frame"],
                    time_s=row["time_s"],
                    players=players,
                    ball=ball,
                )
            )
    fps = float(meta.get("fps", default_fps))
    if not meta and frames:
        fps = default_fps
    return TrackingExport(
        source_video=meta.get("source_video", str(path)),
        fps=fps,
        total_frames=int(meta.get("total_frames", len(frames))),
        frames=frames,
    )
