#!/usr/bin/env python3
"""
Single entry point for soccer analytics.

Offline (CPU, no models) — uses existing JSONL or bundled sample:
  python run.py offline
  python run.py offline --jsonl outputs/2e57b9_0_frames.jsonl --out_dir outputs/demo_shape

Full pipeline (GPU) — export JSONL from video then run offline:
  python run.py full --video roboflow-sports/examples/soccer/data/2e57b9_0.mp4 --device cuda
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_JSONL = ROOT / "outputs" / "2e57b9_0_frames.jsonl"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def cmd_offline(jsonl: Path, out_dir: Path, occupancy_png: Path) -> None:
    if not jsonl.is_file():
        raise SystemExit(
            f"Missing {jsonl}. Run full pipeline first or use bundled sample after clone."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    occupancy_png.parent.mkdir(parents=True, exist_ok=True)

    _run([
        sys.executable, "analyze_team_shape.py",
        "--jsonl", str(jsonl),
        "--out_dir", str(out_dir),
    ])
    _run([
        sys.executable, "render_heatmap_offline.py",
        "--jsonl", str(jsonl),
        "--export_final_png", str(occupancy_png),
    ])
    print(f"\nDone.\n  Shape charts: {out_dir}/\n  Occupancy PNG: {occupancy_png}")


def cmd_full(video: Path, device: str, out_dir: Path) -> None:
    if not video.is_file():
        raise SystemExit(f"Video not found: {video}")

    stem = video.stem
    jsonl = out_dir / f"{stem}_frames.jsonl"
    shape_dir = out_dir / f"{stem}_shape"
    occupancy = out_dir / f"{stem}_occupancy_offline.png"
    out_dir.mkdir(parents=True, exist_ok=True)

    _run([
        sys.executable, "render_heatmap_radar.py",
        "--source_video_path", str(video),
        "--target_video_path", str(out_dir / f"{stem}_heatmap.mp4"),
        "--export_jsonl", str(jsonl),
        "--minimap_only",
        "--viz", "accumulated",
        "--device", device,
    ])
    cmd_offline(jsonl, shape_dir, occupancy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sports CV analytics entry point")
    sub = parser.add_subparsers(dest="command", required=True)

    off = sub.add_parser("offline", help="Team shape + heatmap from JSONL (CPU)")
    off.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    off.add_argument("--out_dir", type=Path, default=ROOT / "outputs" / "demo_shape")
    off.add_argument(
        "--occupancy_png",
        type=Path,
        default=ROOT / "outputs" / "demo_occupancy.png",
    )

    full = sub.add_parser("full", help="GPU export JSONL then offline analytics")
    full.add_argument("--video", type=Path, required=True)
    full.add_argument("--device", default="cuda")
    full.add_argument("--out_dir", type=Path, default=ROOT / "outputs")

    args = parser.parse_args()
    if args.command == "offline":
        cmd_offline(args.jsonl, args.out_dir, args.occupancy_png)
    else:
        cmd_full(args.video, args.device, args.out_dir)


if __name__ == "__main__":
    main()
