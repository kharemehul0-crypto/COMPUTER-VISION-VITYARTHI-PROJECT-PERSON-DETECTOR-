# Pose-Based Violence Detection

So this is a small tool I put together that looks at a video and tries to figure out when something violent might be happening in it — basically, whenever two people are standing close together and one of them starts throwing their arms around fast. That's it, that's the whole idea. It's not some deep learning model that's been trained to "recognize violence" — honestly I didn't have the dataset or the time for that. Instead I'm using a pretrained pose model (YOLOv8-pose) to get the skeleton of every person in the frame, and then just doing some math on top of that.

I like this approach because it's dumb in a good way — you can actually look at why it flagged something instead of just trusting a black box.

> You don't need to train anything or download any dataset. It just works using a model that's already trained.

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

Okay so here's roughly how I thought about building this, step by step.

First thing that happens is every frame goes through YOLOv8-pose. This finds each person in the frame and gives back 17 points on their body — shoulders, elbows, wrists, hips, that sort of thing. Nothing fancy here, this part is just using someone else's pretrained model as-is.

The tricky bit is that pose estimation on its own has no memory — it doesn't know that the person in frame 10 is the same person in frame 11, it just sees a bunch of points each time. So I used Ultralytics' tracker to give each person an ID that sticks with them across frames. Without this, none of the motion stuff below would even be possible.

Once I can actually follow a person over time, I calculate a motion score — how far their wrists and elbows moved since the last frame. I had to scale this by torso length though, because otherwise someone standing right next to the camera always looks like they're moving way faster than someone further back, which obviously isn't a fair comparison.

Same deal for proximity — I check how close two people are to each other, again scaled by body size so the camera distance doesn't mess things up.

Then the actual flagging logic is honestly pretty simple: if two people are close AND one of them has high motion, and this keeps happening for a few frames in a row (not just a single weird spike), that gets logged as a possible event. The "few frames in a row" part matters a lot — without it, the thing flags basically everything.

**What you get at the end:**
- A video with the skeletons drawn on and a red border showing up on flagged frames
- A JSON file with the timestamps of everything it caught

If you want the longer version of why I made these particular choices (and where they fall apart), that's all in [`REPORT.md`](REPORT.md).

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

### Before you start, you'll need

- Python 3.9+
- pip
- A virtual environment (not mandatory but seriously just use one)

### Setup

```bash
# grab the code
git clone https://github.com/kharemehul0-crypto/COMPUTER-VISION_VITYARTHI_PROJECT_VIOLENCE_DETECTION
cd COMPUTER-VISION_VITYARTHI_PROJECT_VIOLENCE_DETECTION

# make a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# install everything
pip install -r requirements.txt
```

> Heads up — the very first time you run the script, it's going to silently download the yolov8n-pose weights (~7MB, so not a big deal) because it needs them and doesn't have them yet. Just means you need internet for that first run. After that it's saved locally and you're fine offline.

---

## 4. Running the Project

### Simplest way to run it

```bash
python detect_violence.py --source path/to/input_video.mp4
```

That gives you back:

- `output/annotated_output.mp4` — your video with skeletons on it and red borders wherever it flagged something
- `output/events.json` — a log of when each flagged event started and ended

### If you'd rather use your webcam

```bash
python detect_violence.py --source 0
```

### All the knobs you can turn

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

| Argument | Default | What it does |
|---|---|---|
| `--source` | *(required)* | your video, or `0` for webcam |
| `--output` | `output/annotated_output.mp4` | where the output video goes |
| `--log` | `output/events.json` | where the event log goes |
| `--model` | `yolov8n-pose.pt` | which pose model to load |
| `--motion-threshold` | `1.4` | raise this if arms need to move faster to count |
| `--proximity-threshold` | `0.35` | raise this if people need to be closer to count |
| `--consecutive-frames` | `5` | how many frames in a row before it's a "real" event, not just noise |
| `--conf` | `0.4` | how confident YOLO needs to be that it's even seeing a person |

### What the log file looks like

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

This is probably where you'll spend most of your time if you're testing on your own clips, ngl.

- **Getting flagged on stuff that's clearly not violent?** Bump up the motion and proximity thresholds.
- **It's missing stuff it should be catching?** Do the opposite — lower those, or drop consecutive-frames down. Just know you'll get more false positives as a trade-off, there's no free lunch here.
- I picked the defaults just by messing around with a few test clips, not through any real scientific process, so don't assume they're perfect for your footage. Different angles/lighting will probably need different numbers. More on this in [`REPORT.md`](REPORT.md) if you want the details.

---

## 6. Notes & Limitations

Gonna be straight about what this actually is, because it'd be easy to make it sound more impressive than it is. This is NOT a trained violence classifier. Nothing here has ever seen a labeled example of "this is violence" vs "this isn't." It's just a rule based on motion + proximity, and I did that on purpose — I'd rather have something simple that I fully understand than something slightly more accurate that I can't explain.

The obvious problem with this is that it can't tell the difference between a fight and, like, two people hugging enthusiastically, dancing, or playing a sport — all of those involve fast arm movement and closeness too. So yeah, false positives happen. I go into this more in [`REPORT.md`](REPORT.md).

Tested this on Python 3.10, works fine on macOS, Linux, and Windows.

---

<p align="center">
  Made by <b>Mehul</b> (24BAI10631) · VIT-Yarthi Computer Vision Project
</p>