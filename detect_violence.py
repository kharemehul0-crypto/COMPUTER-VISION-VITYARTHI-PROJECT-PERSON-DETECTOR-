"""
detect_violence.py
-------------------
A command-line tool that scans a video for possible violent activity
using pose estimation, without needing a custom-trained violence
classifier or any training dataset.

HOW IT WORKS (high level):
1. Every frame is passed through a pretrained YOLOv8-pose model, which
   detects each person in the frame and returns 17 body keypoints per
   person (nose, shoulders, elbows, wrists, hips, knees, ankles).
2. We track each person across frames using Ultralytics' built-in
   tracker (model.track), so we can compare the SAME person's pose
   from one frame to the next.
3. For every tracked person, we compute a "motion score": how much
   their arms moved since the last frame, normalized by their body
   size (see utils.motion_score).
4. For every pair of people, we compute a "proximity score": how close
   they are to each other, normalized by body size (see
   utils.proximity_score).
5. If two people are close together AND at least one of them has a
   high motion score for several consecutive frames, the frame is
   flagged as a possible violent event.
6. Flagged frames are annotated with a red "ALERT" box + skeleton, and
   the video is saved with these annotations. A JSON log of every
   flagged timestamp is also written.

This is a heuristic (rule-based) detector, not a trained classifier -
it is meant as a lightweight, explainable baseline, and its
assumptions/limitations are discussed in REPORT.md.

USAGE:
    python detect_violence.py --source input.mp4 --output annotated.mp4 --log events.json

    python detect_violence.py --source 0   # use webcam
"""

import argparse
import json
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

from utils import motion_score, proximity_score, keypoints_to_array


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pose-based violence / distress detector (YOLOv8-pose)."
    )
    parser.add_argument(
        "--source", required=True,
        help="Path to input video file, or '0' for webcam.",
    )
    parser.add_argument(
        "--output", default="output/annotated_output.mp4",
        help="Path to save the annotated output video.",
    )
    parser.add_argument(
        "--log", default="output/events.json",
        help="Path to save the JSON log of flagged events.",
    )
    parser.add_argument(
        "--model", default="yolov8n-pose.pt",
        help="YOLOv8-pose weights to use (auto-downloaded if not present).",
    )
    parser.add_argument(
        "--motion-threshold", type=float, default=1.4,
        help="Normalized arm-motion score above which a person is 'agitated'. "
             "Lower = more sensitive (more false alarms), higher = less sensitive.",
    )
    parser.add_argument(
        "--proximity-threshold", type=float, default=0.35,
        help="Closeness score (0-1) above which two people are considered "
             "'interacting'. Higher = they must be closer together.",
    )
    parser.add_argument(
        "--consecutive-frames", type=int, default=5,
        help="Number of consecutive flagged frames required before logging "
             "an official 'event' (reduces one-frame false alarms).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.4,
        help="YOLO detection confidence threshold.",
    )
    return parser.parse_args()


def draw_skeleton(frame, keypoints, color=(0, 255, 0)):
    """Draw keypoints and simple limb connections for one person."""
    skeleton = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # arms + shoulders
        (5, 11), (6, 12), (11, 12),                 # torso
        (11, 13), (13, 15), (12, 14), (14, 16),     # legs
    ]
    for x, y in keypoints:
        if x == 0 and y == 0:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    for a, b in skeleton:
        pa, pb = keypoints[a], keypoints[b]
        if np.all(pa == 0) or np.all(pb == 0):
            continue
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)


def main():
    args = parse_args()

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    source = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Keep the previous frame's keypoints for each tracked person ID
    prev_keypoints_by_id = {}
    # Rolling history of "is this frame flagged?" so we can require
    # several consecutive flagged frames before logging a real event
    flag_history = deque(maxlen=args.consecutive_frames)

    events = []          # final list of logged events (start/end timestamps)
    current_event = None # event currently being tracked across frames
    frame_idx = 0

    print("[INFO] Processing video... press 'q' in the preview window to stop early.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_idx / fps

        # model.track keeps consistent IDs for the same person across frames
        results = model.track(
            frame, persist=True, conf=args.conf, verbose=False
        )[0]

        frame_flagged = False
        keypoints_by_id = {}

        if results.keypoints is not None and results.boxes is not None and results.boxes.id is not None:
            ids = results.boxes.id.cpu().numpy().astype(int)
            all_kpts = results.keypoints.xy.cpu().numpy()  # shape: (num_people, 17, 2)

            # Compute motion score per person
            motion_by_id = {}
            for pid, kpts in zip(ids, all_kpts):
                kpts = keypoints_to_array(kpts)
                keypoints_by_id[pid] = kpts
                prev_kpts = prev_keypoints_by_id.get(pid)
                motion_by_id[pid] = motion_score(prev_kpts, kpts)
                draw_skeleton(frame, kpts)

            # Check every pair of people for "close + agitated" interaction
            person_ids = list(keypoints_by_id.keys())
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    id_a, id_b = person_ids[i], person_ids[j]
                    closeness = proximity_score(
                        keypoints_by_id[id_a], keypoints_by_id[id_b]
                    )
                    high_motion = (
                        motion_by_id[id_a] > args.motion_threshold
                        or motion_by_id[id_b] > args.motion_threshold
                    )
                    if closeness > args.proximity_threshold and high_motion:
                        frame_flagged = True

            prev_keypoints_by_id = keypoints_by_id

        flag_history.append(frame_flagged)

        # Only treat it as a real event once we see enough consecutive
        # flagged frames in a row (reduces noisy single-frame false alarms)
        is_sustained_event = (
            len(flag_history) == flag_history.maxlen and all(flag_history)
        )

        if is_sustained_event:
            if current_event is None:
                current_event = {"start_time_sec": round(timestamp_sec, 2)}
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 8)
            cv2.putText(
                frame, "ALERT: Possible violent activity", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
            )
        else:
            if current_event is not None:
                current_event["end_time_sec"] = round(timestamp_sec, 2)
                events.append(current_event)
                current_event = None

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 30 == 0:
            print(f"[INFO] Processed {frame_idx} frames...")

    # Close out any event still open when the video ends
    if current_event is not None:
        current_event["end_time_sec"] = round(frame_idx / fps, 2)
        events.append(current_event)

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    print(f"[INFO] Done. Processed {frame_idx} frames in {elapsed:.1f}s.")
    print(f"[INFO] Annotated video saved to: {args.output}")

    log_data = {
        "source": args.source,
        "total_frames": frame_idx,
        "fps": fps,
        "settings": {
            "motion_threshold": args.motion_threshold,
            "proximity_threshold": args.proximity_threshold,
            "consecutive_frames": args.consecutive_frames,
        },
        "num_events": len(events),
        "events": events,
    }
    with open(args.log, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"[INFO] Event log saved to: {args.log}")
    print(f"[INFO] Total flagged events: {len(events)}")


if __name__ == "__main__":
    main()
