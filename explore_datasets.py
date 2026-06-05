#!/usr/bin/env python3
"""Download and summarize Roboflow football + basketball sports datasets."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "roboflow-jvuqo"
DATA_DIR = Path(__file__).resolve().parent / "data"

DATASETS = {
    "football_players": {
        "project": "football-players-detection-3zvbc",
        "task": "object detection",
        "sport": "football",
    },
    "football_ball": {
        "project": "football-ball-detection-rejhg",
        "task": "object detection",
        "sport": "football",
    },
    "football_pitch": {
        "project": "football-field-detection-f07vi",
        "task": "keypoint detection",
        "sport": "football",
    },
    "basketball_court": {
        "project": "basketball-court-detection-2",
        "task": "keypoint detection",
        "sport": "basketball",
    },
    "basketball_jersey_ocr": {
        "project": "basketball-jersey-numbers-ocr",
        "task": "OCR / multimodal",
        "sport": "basketball",
    },
}


def count_split(root: Path) -> dict:
    counts = {}
    for split in ("train", "valid", "test"):
        img_dir = root / split / "images"
        if img_dir.is_dir():
            counts[split] = len(list(img_dir.glob("*")))
    return counts


def read_yaml_classes(root: Path) -> list[str]:
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        return []
    names: list[str] = []
    in_names = False
    for line in data_yaml.read_text().splitlines():
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            if line.startswith(" ") or line.startswith("\t"):
                val = line.split(":", 1)[-1].strip().strip("'\"")
                if val:
                    names.append(val)
            else:
                break
    return names


def label_histogram(root: Path, max_files: int = 500) -> Counter:
    hist: Counter = Counter()
    label_dirs = list((root / "train" / "labels").glob("*.txt"))[:max_files]
    for label_file in label_dirs:
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if parts:
                hist[parts[0]] += 1
    return hist


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY to download Universe datasets.")

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(WORKSPACE)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary = []

    for key, meta in DATASETS.items():
        print(f"\n{'=' * 60}\n{key} ({meta['project']})\n{'=' * 60}")
        entry = {**meta, "key": key, "status": "ok"}

        try:
            project = ws.project(meta["project"])
            versions = project.versions()
            version_nums = [v.version for v in versions]
            latest = max(version_nums) if version_nums else None
            entry["versions"] = version_nums
            entry["latest_version"] = latest
            print(f"Available versions: {version_nums} -> using v{latest}")

            out_dir = DATA_DIR / key
            if out_dir.exists():
                print(f"Already downloaded at {out_dir}, skipping download.")
            else:
                dataset = project.version(latest).download("yolov8", location=str(out_dir))
                out_dir = Path(dataset.location)

            entry["path"] = str(out_dir)
            entry["splits"] = count_split(out_dir)
            entry["classes"] = read_yaml_classes(out_dir)

            if (out_dir / "train" / "labels").exists() and meta["task"] == "object detection":
                hist = label_histogram(out_dir)
                if entry["classes"]:
                    entry["label_counts_sample"] = {
                        entry["classes"][int(k)]: v for k, v in sorted(hist.items(), key=lambda x: int(x[0]))
                    }
                else:
                    entry["label_counts_sample"] = dict(hist)

            print(f"Splits: {entry['splits']}")
            print(f"Classes: {entry['classes']}")
            if entry.get("label_counts_sample"):
                print(f"Label counts (train sample): {entry['label_counts_sample']}")

        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"ERROR: {exc}")

        summary.append(entry)

    report_path = DATA_DIR / "dataset_summary.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary -> {report_path}")


if __name__ == "__main__":
    main()
