# Project Report: Pose-Based Violence Detection

## 1. Problem Statement

Automatically detecting violent or aggressive physical interactions in
video footage (e.g., CCTV, public safety cameras) is a well-studied
Computer Vision problem with applications in public safety monitoring
and automated surveillance alerting. This project builds a lightweight,
explainable system that flags segments of video where a possible violent
interaction is occurring, using human pose estimation rather than a
trained "violence classifier."

## 2. Motivation for the Approach Chosen

Two broad approaches exist for this problem:

1. **End-to-end learned classifiers** (e.g., 3D-CNNs, video transformers)
   trained on labeled violent/non-violent video datasets (such as the
   Hockey Fight or RWF-2000 datasets).
2. **Pose/motion-based heuristics**, which use an off-the-shelf pose
   estimator as a feature extractor and apply interpretable rules on top
   of the extracted keypoints.

Given the time and data constraints of this project (no access to a
large labeled violence dataset, no GPU training pipeline), approach (2)
was chosen. It has three advantages for this context:

- **No training data or training time required** — the pose model
  (YOLOv8-pose) is pretrained on COCO and used purely as a feature
  extractor.
- **Fully explainable** — every flagged event can be traced back to
  exactly which two people were close together and how fast their arms
  were moving, unlike a black-box neural classifier.
- **Runs on CPU** — no GPU dependency, making it easy to set up and run
  in any environment, including the evaluation environment.

The trade-off, discussed in Section 5, is lower accuracy and higher
false-positive rate compared to a properly trained end-to-end model.

## 3. System Design

### 3.1 Pipeline Overview

```
Video frames
   │
   ▼
YOLOv8-pose (pretrained) + built-in tracker
   │  → per-person: 17 keypoints, persistent ID
   ▼
Per-person motion score  (utils.motion_score)
   │  → normalized arm displacement vs. previous frame
   ▼
Per-pair proximity score (utils.proximity_score)
   │  → normalized closeness between two people
   ▼
Rule: closeness > threshold AND motion > threshold
   │  → sustained over N consecutive frames
   ▼
Flagged event → annotated video + JSON log
```

### 3.2 Key Design Decisions

- **Why normalize by torso length?** A person standing close to the
  camera naturally has larger pixel movements than someone far away for
  the same physical action. Dividing displacement by torso length (in
  pixels) makes the motion score roughly scale-invariant.
- **Why focus on wrists/elbows only?** Arm movement (punches, pushes,
  swings) is the most visually distinctive signal of physical
  aggression, compared to leg or torso movement, which is noisier and
  more common in ordinary walking.
- **Why require proximity AND motion together?** High motion alone is
  triggered by many harmless activities (waving, exercising, dancing
  alone). Requiring two people to also be close together substantially
  reduces false positives from a single agitated person.
- **Why require N consecutive frames?** Pose estimation is noisy
  frame-to-frame (a misdetected keypoint can cause a large spurious
  "jump"). Requiring a sustained pattern across several frames filters
  out this single-frame noise.

## 4. Implementation

- **Language:** Python 3
- **Libraries:** `ultralytics` (YOLOv8-pose + tracking), `opencv-python`
  (video I/O, drawing), `numpy` (vector math)
- **Files:**
  - `utils.py` — pure functions for keypoint math (motion score,
    proximity score, body-scale normalization)
  - `detect_violence.py` — CLI entry point: reads video, runs pose
    tracking frame-by-frame, applies the flagging rule, writes annotated
    video + JSON log

The tool is fully command-line driven (`python detect_violence.py
--source input.mp4`), requiring no GUI, per the project's executability
requirement.

## 5. Evaluation & Limitations

### What works well
- Reliably detects two people in close, high-motion contact (e.g.,
  pushing, grappling, rapid arm swinging toward another person).
- Produces clear, timestamped, human-readable output for review.
- Fully deterministic and tunable via CLI thresholds — no retraining
  needed to adjust sensitivity.

### Known limitations
- **False positives:** Non-violent activities that involve two people
  close together with fast arm motion — e.g., hugging enthusiastically,
  playful roughhousing, some dance forms, or sports — can trigger a
  false alert. The system detects "high energy, close-contact motion,"
  which correlates with but is not identical to violence.
- **False negatives:** Slow, low-motion aggression (e.g., a static
  choke-hold, a threat with a weapon but little arm movement) will not
  be flagged, since the heuristic depends specifically on fast arm
  displacement.
- **Occlusion sensitivity:** If people overlap heavily or keypoints are
  occluded, YOLOv8-pose keypoint confidence drops, which can cause
  missed detections or noisy motion scores.
- **No semantic understanding:** Unlike a trained classifier, this
  system has no concept of "violence" beyond motion + proximity — it
  cannot distinguish intent.

### Possible future improvements
- Replace the hand-crafted rule with a small trained classifier (e.g., a
  lightweight MLP or LSTM) that takes the same keypoint-derived features
  as input, trained on a small labeled clip dataset, combining the
  interpretability of pose features with learned decision boundaries.
- Add facial-expression or audio-based signals (e.g., raised voices,
  screaming) as additional evidence, fused with the motion signal.
- Use a proper multi-object tracker (e.g., ByteTrack/DeepSORT) for more
  robust ID persistence in crowded scenes.

## 6. Conclusion

This project demonstrates that a pretrained pose-estimation model can be
repurposed, without any additional training, into a lightweight and
fully explainable violent-activity flagging system, by combining simple,
well-justified motion and proximity heuristics. While not as accurate as
a purpose-trained end-to-end classifier, it is transparent, fast to set
up, requires no dataset, and runs entirely on CPU — making it a
practical baseline for further development.
