# Football vs Basketball — Roboflow Sports Data

Side-by-side reference for the official [roboflow/sports](https://github.com/roboflow/sports) Universe datasets.

## Quick comparison

| | Football (soccer) | Basketball |
|---|---|---|
| **Maturity in repo** | Full `examples/soccer/` pipeline + demo videos | Datasets only; no full example yet ([#47](https://github.com/roboflow/sports/issues/47)) |
| **Pretrained weights** | ✅ Google Drive (via `setup.sh`) | ❌ You train or use Universe hosted models |
| **Demo videos** | ✅ 5 Bundesliga clips (~18–21 MB each) | ❌ Bring your own |
| **Hardest task** | Ball tracking (small, fast) | Jersey OCR + player re-ID |
| **Best for PR to Roboflow** | Extend radar / analytics | Build missing `examples/basketball/` |

---

## Football datasets

All hosted under workspace `roboflow-jvuqo`.

### 1. Player detection
- **Universe:** [football-players-detection-3zvbc](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
- **Task:** Object detection
- **Classes:** `ball`, `goalkeeper`, `player`, `referee`
- **Size:** ~372 images (Universe); notebook uses **v10** → 250 train / 43 valid
- **Metrics:** mAP@50 83%, Precision 72%, Recall 88%
- **Pretrained:** `football-player-detection.pt` (137 MB)

### 2. Ball detection
- **Universe:** [football-ball-detection-rejhg](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg)
- **Task:** Object detection (ball-only, easier to train than multi-class)
- **Size:** notebook uses **v2** → ~1,966 train / 121 valid (much larger than player set)
- **Pretrained:** `football-ball-detection.pt` (137 MB)

### 3. Pitch keypoints
- **Universe:** [football-field-detection-f07vi](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi)
- **Task:** Keypoint detection (32 pitch landmarks)
- **Classes:** `pitch`
- **Size:** ~317 images; notebook uses **v12** → 222 train / 30 valid
- **Pretrained:** `football-pitch-detection.pt` (140 MB)
- **Use:** Homography → radar view, speed/distance in meters

### Demo videos (local after `setup.sh`)
| File | Size |
|------|------|
| `0bfacc_0.mp4` | 19.9 MB |
| `2e57b9_0.mp4` | 21.1 MB |
| `08fd33_0.mp4` | 19.9 MB |
| `573e61_0.mp4` | 18.9 MB |
| `121364_0.mp4` | 17.2 MB |

Source: [DFL Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) (broadcast-style, panning camera).

---

## Basketball datasets

### 1. Court keypoint detection
- **Universe:** [basketball-court-detection-2](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2)
- **Task:** Keypoint detection
- **Classes:** `basketball_court` (court line landmarks)
- **Size:** ~850 images (Roboflow compilation); enables top-down shot charts
- **Note:** Basketball court config + `ShotEventTracker` exist in open [PR #37](https://github.com/roboflow/sports/pull/37) but not merged to main yet

### 2. Jersey number OCR
- **Universe:** [basketball-jersey-numbers-ocr](https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr)
- **Task:** Multimodal / OCR (Roboflow's listed hard problem)
- **Size:** ~3,600 images
- **Use:** Persistent player identity when tracking fails

### What's missing for basketball
- No `examples/basketball/` folder (unlike soccer)
- No bundled pretrained `.pt` weights or sample videos
- No end-to-end demo (court → players → jersey → shot chart)

---

## What we have locally

```
sports_cv/
├── roboflow-sports/examples/soccer/data/   # models + 5 videos ✅
├── previews/football/                      # sample inference outputs ✅
├── explore_datasets.py                       # download all 5 Universe datasets
└── DATASETS.md                               # this file
```

**Football preview (frame 120 of `2e57b9_0.mp4`):**
- Players model: 25 detections (players, goalkeepers, referees)
- Ball model: 1 detection
- Pitch model: 32 keypoints

---

## Download labeled datasets (both sports)

Universe zip downloads need a free [Roboflow API key](https://app.roboflow.com/settings/api):

```bash
export ROBOFLOW_API_KEY="your_key"
cd /users/PAS2699/pratham2210/sports_cv
python3 explore_datasets.py
```

This pulls all 5 datasets into `sports_cv/data/` and writes `dataset_summary.json` with split counts and class histograms.

---

## Annotation formats

**Object detection (YOLOv8):**
```
<class_id> <x_center> <y_center> <width> <height>   # normalized 0–1
```

**Keypoint detection (YOLOv8-pose):**
```
<class_id> <x> <y> <w> <h> <kp1_x> <kp1_y> <kp1_v> ...   # v = visibility
```

Football pitch uses **32 keypoints** mapped to real-world cm coordinates via `SoccerPitchConfiguration`.

---

## Recommendation after looking at both

| If you want… | Pick |
|---|---|
| Fastest path to a working demo | **Football** — models + videos ready |
| Highest differentiation / merge potential | **Basketball** — fill the gap Roboflow asked for |
| Compare before deciding | Run `explore_datasets.py` with API key, inspect `dataset_summary.json` |

You can also **combine**: use football for radar/homography skills, then port the pattern to basketball court keypoints.
