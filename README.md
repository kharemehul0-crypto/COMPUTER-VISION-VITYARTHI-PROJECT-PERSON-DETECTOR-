# Pose-Based Violence Detection

A command-line tool that scans a video and flags frames where two people are close together **and** at least one is showing fast, high-amplitude arm motion — a simple, explainable heuristic for detecting possible violent interactions, built on top of a pretrained pose-estimation model (YOLOv8-pose).

> No custom model training or labeled dataset is required.

<p align="center">
  <img src="sample_output/demo1.gif" width="24%" alt="Sample output 1" />
  <img src="sample_output/demo2.gif" width="24%" alt="Sample output 2" />
  <img src="sample_output/demo3.gif" width="24%" alt="Sample output 3" />
  <img src="sample_output/demo4.gif" width="24%" alt="Sample output 4" />
</p>

---

## Author

| | |
|---|---|
| **Name** | Mehul |
| **Registration No.** | 24BAI10631 |
| **Project** | VIT-Yarthi — Computer Vision Project |
| **Course** | Computer Vision |

---

## Table of Contents

1. [How It Works](#1-how-it-works)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Running the Project](#4-running-the-project)
5. [Tuning Tips](#5-tuning-tips)
6. [Notes & Limitations](#6-notes--limitations)

---

## 1. How It Works

The pipeline runs in five stages:

1. **Pose estimation** — Every frame is run through a pretrained YOLOv8-pose model (Ultralytics), which detects each person and returns 17 body keypoints (shoulders, elbows, wrists, hips, etc.).
2. **Tracking** — Ultralytics' built-in tracker assigns a consistent ID to each person across frames, so the same person's pose can be compared from one frame to the next.
3. **Motion score** — For each tracked person, the system measures how far their wrists/elbows moved since the previous frame, normalized by torso length (so it works regardless of distance from the camera).
4. **Proximity score** — For every pair of people in the frame, the system measures how close together they are, also normalized by body size.
5. **Flagging** — If two people are close together *and* one of them has a high motion score, for several consecutive frames in a row, that segment is logged as a possible violent event.

**Output:**
- An annotated video (skeletons overlaid + a red "ALERT" border on flagged frames)
- A JSON log of event timestamps

> Full design rationale, assumptions, and limitations are documented in [`REPORT.md`](REPORT.md).

---

## 2. Project Structure

```
violence_detection/
├── detect_violence.py   # main CLI script
├── utils.py             # keypoint math: motion score, proximity score
├── requirements.txt     # Python dependencies
├── README.md
├── REPORT.md
└── sample_output/       # example output goes here when you run it
```

---

## 3. Environment Setup

### Prerequisites

- Python 3.9 or later
- pip
- (Recommended) a virtual environment

### Step-by-step setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The first time you run the script, `ultralytics` will automatically download the pretrained `yolov8n-pose.pt` weights file (~7 MB) — no manual download needed, but you do need an internet connection for that first run.

---

## 4. Running the Project

### Basic usage

```bash
python detect_violence.py --source path/to/input_video.mp4
```

This produces:

- `output/annotated_output.mp4` — the input video with skeletons drawn and a red border/label on frames flagged as possible violent activity
- `output/events.json` — a JSON log of every flagged event's start/end timestamp (in seconds)

### Using a webcam instead of a file

```bash
python detect_violence.py --source 0
```

### Full options

```bash
python detect_violence.py \
    --source input.mp4 \
    --output output/annotated_output.mp4 \
    --log output/events.json \
    --motion-threshold 1.4 \
    --proximity-threshold 0.35 \
    --consecutive-frames 5 \
    --conf 0.4
```

| Argument | Default | Meaning |
|---|---|---|
| `--source` | *(required)* | Input video path, or `0` for webcam |
| `--output` | `output/annotated_output.mp4` | Where to save the annotated video |
| `--log` | `output/events.json` | Where to save the JSON event log |
| `--model` | `yolov8n-pose.pt` | Pretrained pose model to use |
| `--motion-threshold` | `1.4` | Higher = requires faster arm movement to flag (fewer false alarms) |
| `--proximity-threshold` | `0.35` | Higher = people must be closer together to count as "interacting" |
| `--consecutive-frames` | `5` | How many flagged frames in a row before it's logged as a real event |
| `--conf` | `0.4` | YOLO person-detection confidence threshold |

### Example output log (`events.json`)

```json
{
  "source": "input.mp4",
  "total_frames": 450,
  "fps": 30.0,
  "num_events": 2,
  "events": [
    { "start_time_sec": 4.2, "end_time_sec": 6.1 },
    { "start_time_sec": 11.8, "end_time_sec": 13.0 }
  ]
}
```

---

## 5. Tuning Tips

- **Too many false alarms?** Increase `--motion-threshold` and/or `--proximity-threshold`.
- **Missing real events?** Decrease those thresholds, or lower `--consecutive-frames` (at the cost of more noise).
- Thresholds were chosen empirically and are meant to be adjusted per camera angle/footage — see [`REPORT.md`](REPORT.md) for discussion.

---

## 6. Notes & Limitations

- This is a **rule-based heuristic**, not a trained violence classifier. It is intentionally simple and explainable rather than state-of-the-art accurate.
- It can be triggered by non-violent fast motion such as hugging, dancing, or sports — see [`REPORT.md`](REPORT.md) for a full discussion of limitations.
- Tested on Python 3.10, macOS / Linux / Windows.

---

<p align="center">
  Made by <b>Mehul</b> (24BAI10631) · VIT-Yarthi Computer Vision Project
</p>
