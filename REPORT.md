# Project Report: Pose-Based Violence Detection

**Author:** Mehul (24BAI10631) · **Course:** Computer Vision · **Project:** VIT-Yarthi Computer Vision Project

---

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Motivation for the Approach Chosen](#2-motivation-for-the-approach-chosen)
- [3. Related Work](#3-related-work)
- [4. System Design](#4-system-design)
- [5. Implementation](#5-implementation)
- [6. Testing Methodology](#6-testing-methodology)
- [7. Evaluation & Limitations](#7-evaluation--limitations)
- [8. Ethical Considerations & Responsible Use](#8-ethical-considerations--responsible-use)
- [9. Conclusion](#9-conclusion)
- [10. References](#10-references)

---

## 1. Problem Statement

Automatically detecting violent or aggressive physical interactions in video footage — CCTV feeds, public safety cameras, crowd-monitoring systems — is a well-studied computer vision problem with direct applications in public safety monitoring, automated surveillance alerting, workplace incident detection, and post-event forensic review. Manual monitoring of large volumes of live or archived footage does not scale: a human operator watching dozens of camera feeds simultaneously is prone to fatigue and delayed reaction, and reviewing archived footage after an incident is time-consuming.

This creates demand for automated systems that can either (a) flag live footage for human review in near real time, or (b) pre-filter archived footage down to a short list of candidate timestamps worth reviewing, rather than requiring a human to scrub through hours of video. Both use cases share the same underlying requirement: a system that can look at a stream of frames and output a small number of timestamped "this looks suspicious" events, ideally with enough transparency that a human reviewer can understand *why* a given segment was flagged.

This project builds such a system. Rather than training a dedicated "violence classifier" — a model that ingests raw video and outputs a violence/non-violence label — it builds a lightweight, explainable pipeline that flags segments of video where a possible violent interaction is occurring, using human pose estimation as an intermediate representation rather than working on raw pixels.

## 2. Motivation for the Approach Chosen

Two broad families of approaches exist for this problem:

1. **End-to-end learned classifiers** — 3D convolutional networks, two-stream CNNs, or video transformers trained directly on labeled violent/non-violent video clips, using datasets such as Hockey Fight, RWF-2000, or Real Life Violence Situations. These models learn to extract whatever spatiotemporal features best separate the two classes, without any hand-engineered notion of what "violence" looks like.
2. **Pose/motion-based heuristics** — approaches that first run an off-the-shelf pose estimator over the video to obtain a compact, structured representation of each person (a skeleton of keypoints), and then apply interpretable rules or lightweight models on top of that structured representation rather than on raw pixels.

Given the time and resource constraints of this project — no access to a large labeled violence dataset, and no GPU training pipeline available in the evaluation environment — approach (2) was selected. This choice carries three concrete advantages in this context:

| Advantage | Explanation |
|---|---|
| **No training data required** | The pose model (YOLOv8-pose) is pretrained on the COCO keypoints dataset and used purely as a fixed feature extractor; no violence-specific labels or clips need to be collected or annotated. |
| **Fully explainable** | Every flagged event can be traced back to exactly which two tracked person-IDs were involved, how close they were, and how fast their arms were moving at that timestamp — unlike a black-box neural classifier, whose decision boundary is not human-interpretable without additional tooling (e.g., saliency maps). |
| **Runs on CPU** | No GPU dependency at inference time, which keeps setup effort low and makes the system runnable in constrained or shared evaluation environments where GPU access cannot be guaranteed. |

The trade-off, discussed in detail in [Section 7](#7-evaluation--limitations), is that a hand-crafted heuristic will generally have lower accuracy and a higher false-positive rate than a properly trained end-to-end model with access to a large, diverse labeled dataset. This project treats that trade-off as acceptable given the stated goal: a transparent, dependency-light baseline rather than a production-grade detector.

## 3. Related Work

The broader "violence detection in video" literature spans several distinct lines of work, which helps situate the approach taken here:

- **Handcrafted motion features.** Early work in this space (predating deep learning's dominance) used features such as optical flow histograms, motion blobs, and acceleration patterns to characterize aggressive motion, feeding these into classical classifiers such as SVMs. The pose-based motion score used in this project is a conceptual descendant of this line of work, but uses structured keypoint displacement rather than dense optical flow.
- **End-to-end deep video classifiers.** More recent approaches use 3D-CNNs (e.g., C3D, I3D) or two-stream architectures (separate spatial and temporal streams, later fused) trained on labeled violence datasets. These generally outperform handcrafted-feature approaches given enough labeled data, at the cost of requiring that data and offering less interpretability.
- **Skeleton/pose-based action recognition.** A growing body of work performs action recognition — including aggression or fight detection — on top of extracted human skeletons rather than raw pixels, often using graph convolutional networks (ST-GCN and variants) over the keypoint sequence. This project's approach sits at the simple end of this spectrum: rather than training a graph network over pose sequences, it applies a small set of interpretable, hand-derived rules directly to the keypoint trajectories.
- **Anomaly detection framing.** Some surveillance systems frame the problem as unsupervised anomaly detection — learning what "normal" activity looks like for a given scene and flagging deviations — rather than supervised violence/non-violence classification. This project does not take this framing; it is a supervised-by-design-rules heuristic rather than a learned normalcy model.

Positioning this project within that landscape: it is closest in spirit to the skeleton-based line of work, but deliberately avoids any additional model training, instead relying entirely on interpretable geometric rules applied to a pretrained pose estimator's output.

## 4. System Design

### 4.1 Pipeline Overview

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

### 4.2 Pose Estimation and Tracking

Each input frame is passed through a pretrained YOLOv8-pose model (the `yolov8n-pose.pt` nano variant by default, chosen for CPU-friendly inference speed). For every person detected above the confidence threshold (`--conf`, default `0.4`), the model returns:

- A bounding box
- 17 keypoints in the COCO keypoint format (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), each with an `(x, y)` pixel coordinate and a per-keypoint confidence score

Ultralytics' built-in multi-object tracker is layered on top of the raw per-frame detections to assign a persistent integer ID to each person across frames. This identity persistence is what makes it possible to compute *frame-to-frame* motion for "the same person," rather than comparing unrelated detections between frames.

### 4.3 Motion Score

For a tracked person `p`, the motion score at frame `t` is computed conceptually as:

```
motion_score(p, t) = mean(displacement of left/right wrist and elbow keypoints, t-1 → t)
                      -------------------------------------------------------------------
                                        torso_length(p, t)
```

where `torso_length` is derived from the distance between the shoulder and hip keypoints. Dividing by torso length serves as a per-person, per-frame normalization constant: a person occupying more pixels (because they are closer to the camera, or the camera has a tighter zoom) will naturally show larger raw pixel displacements for the same physical action, and this normalization keeps the score roughly comparable across different distances-from-camera and across different body sizes.

Only wrist and elbow keypoints are used in this calculation, rather than all 17 keypoints, for the reason discussed in [Section 4.5](#45-key-design-decisions).

### 4.4 Proximity Score

For every pair of currently-tracked people `(p, q)` in a frame, a proximity score is computed from the distance between their body centers (approximated from hip/shoulder midpoints), again normalized by an average of their torso lengths so that "close" is a relative, scale-invariant notion rather than an absolute pixel distance.

### 4.5 Key Design Decisions

- **Why normalize by torso length?** A person standing close to the camera naturally exhibits larger pixel movements than someone farther away performing the identical physical action. Dividing displacement by torso length (in pixels) makes the motion score approximately scale-invariant, so a single `--motion-threshold` value behaves consistently across different camera distances and framings, rather than needing to be re-tuned per shot.
- **Why focus on wrists and elbows only?** Arm movement — punches, pushes, swings — is the most visually distinctive and semantically relevant signal of physical aggression between two people, compared to leg or torso movement, which is noisier, more frequent during ordinary walking or shifting weight, and less specific to an aggressive interaction. Restricting the motion signal to the arms also reduces sensitivity to camera shake or minor full-body sway that would otherwise inflate the score for stationary people.
- **Why require proximity *and* motion together, rather than either alone?** High motion alone is triggered by many harmless activities — waving, stretching, exercising, dancing alone, or an agitated person pacing. Proximity alone is triggered by any two people standing near each other, which happens constantly in queues, conversations, or crowded spaces. Requiring *both* conditions simultaneously narrows the flagged set specifically to close-contact, high-energy interactions, which substantially reduces false positives compared to either signal in isolation.
- **Why require N consecutive frames (`--consecutive-frames`, default `5`)?** Pose estimation is noisy frame-to-frame: a single misdetected or jittery keypoint can cause a large spurious one-frame "jump" in the computed motion score, even when the person is not actually moving quickly. Requiring the rule to hold across a sustained run of consecutive frames filters out this kind of single-frame noise, at the cost of a small detection latency (roughly `N / fps` seconds before an ongoing event is first logged).

## 5. Implementation

**Language:** Python 3 (tested on Python 3.10; compatible with 3.9+)

**Libraries:**

| Library | Role |
|---|---|
| `ultralytics` | Provides the pretrained YOLOv8-pose model and the built-in multi-object tracker used for pose estimation and identity persistence |
| `opencv-python` | Handles video I/O (reading frames, writing the annotated output video) and drawing overlays — skeletons, bounding boxes, the alert border |
| `numpy` | Vector and array math underlying the keypoint-distance and normalization calculations |

**Files:**

| File | Purpose |
|---|---|
| `utils.py` | Pure, stateless functions for keypoint math: motion score, proximity score, and torso-length-based normalization. Kept separate from the CLI logic so the scoring functions can be unit-tested or reused independently. |
| `detect_violence.py` | The CLI entry point. Orchestrates the full pipeline: opens the video source, runs pose tracking frame-by-frame, maintains per-person keypoint history, applies the proximity-and-motion flagging rule with the consecutive-frame requirement, draws annotations, and writes both the output video and the JSON event log. |

The tool is fully command-line driven (`python detect_violence.py --source input.mp4`), requiring no GUI, per the project's executability requirement. All tunable behavior — thresholds, output paths, model choice — is exposed as CLI flags rather than hardcoded, so the system can be re-tuned for different footage without editing source code.

## 6. Testing Methodology

Because no labeled violence dataset was used, evaluation in this project was necessarily qualitative rather than based on standard classification metrics (precision/recall against ground-truth labels). Testing consisted of running the pipeline against a small set of self-recorded and publicly available sample clips covering three categories:

1. **Clear positive cases** — staged close-contact, high-arm-motion interactions (e.g., mock pushing, mock grappling) intended to resemble the physical signature of an altercation.
2. **Clear negative cases** — ordinary activity such as walking, standing, and conversing at normal distance, used to check that the system does not flag routine footage.
3. **Adversarial negative cases** — activities specifically chosen because they share surface-level characteristics with violence (close proximity *and* fast arm motion) without being violent, such as hugging, high-fives, and casual dancing. These cases were used to probe and characterize the false-positive behavior discussed in [Section 7](#7-evaluation--limitations).

For each test clip, the annotated output video and `events.json` log were manually reviewed to check whether flagged segments aligned with the intended ground truth, and thresholds (`--motion-threshold`, `--proximity-threshold`, `--consecutive-frames`) were adjusted iteratively based on this review. This iterative, small-sample tuning process is the basis for the default threshold values shipped with the tool; they should be treated as reasonable starting points rather than values validated against a large, statistically representative benchmark.

## 7. Evaluation & Limitations

### 7.1 What Works Well

- Reliably detects two people in close, high-motion contact — e.g., pushing, grappling, or rapid arm swinging directed toward another person.
- Produces clear, timestamped, human-readable output suitable for a human reviewer to quickly jump to candidate segments, rather than requiring them to review an entire video.
- Fully deterministic and tunable via CLI thresholds — sensitivity can be adjusted for a new camera angle or setting without any retraining.
- Runs entirely on CPU at a modest frame rate, making it deployable in resource-constrained environments where GPU-based end-to-end classifiers would not be practical.

### 7.2 Known Limitations

| Limitation | Description |
|---|---|
| **False positives** | Non-violent activities involving two people close together with fast arm motion — e.g., enthusiastic hugging, playful roughhousing, some dance forms, or sports — can trigger a false alert. The system detects "high-energy, close-contact motion," which correlates with but is not identical to violence. |
| **False negatives** | Slow, low-motion aggression (e.g., a static choke-hold, a threat made with a weapon but little arm movement, or intimidation without physical contact) will not be flagged, since the heuristic depends specifically on fast arm displacement. |
| **Occlusion sensitivity** | If people overlap heavily or keypoints are occluded — common in exactly the close-contact scenarios this tool targets — YOLOv8-pose keypoint confidence drops, which can cause missed detections, ID switches in the tracker, or noisy motion scores. |
| **No semantic understanding** | Unlike a trained classifier exposed to labeled examples, this system has no learned concept of "violence" beyond motion and proximity — it cannot distinguish intent, and cannot generalize to aggressive interactions that do not fit the specific motion signature it looks for (e.g., verbal aggression, weapon threats). |
| **Single-camera, 2D limitation** | All measurements are computed from 2D pixel coordinates in a single camera view; there is no depth information, so proximity is a projection-based approximation and can be affected by camera angle (e.g., two people who appear close in a wide-angle shot but are actually several meters apart). |

### 7.3 Possible Future Improvements

- Replace the hand-crafted rule with a small trained classifier (e.g., a lightweight MLP, LSTM, or small graph convolutional network) that takes the same keypoint-derived features as input, trained on a small labeled clip dataset — combining the interpretability of pose features with a learned, rather than manually specified, decision boundary.
- Incorporate facial-expression or audio-based signals (e.g., raised voices, screaming, sudden loud sounds) as additional evidence, fused with the motion signal to reduce reliance on arm motion alone.
- Replace the built-in tracker with a dedicated multi-object tracker such as ByteTrack or DeepSORT for more robust identity persistence in crowded or high-occlusion scenes.
- Add temporal smoothing or a Kalman filter over the raw keypoint trajectories to reduce the frame-to-frame jitter that currently motivates the consecutive-frame requirement, potentially allowing faster (lower-latency) event detection.
- Collect a small labeled evaluation set specific to the target deployment scenario and report quantitative precision/recall figures, rather than relying solely on qualitative review as in this project.

## 8. Ethical Considerations & Responsible Use

A system designed to flag "possible violent interactions" from video carries real-world implications beyond its technical accuracy, and these are worth stating explicitly:

- **Human review, not automated action.** This tool is designed to *flag candidate segments for human review*, not to make autonomous decisions (e.g., automatically alerting law enforcement or locking down a facility). Given the known false-positive rate discussed in Section 7, any deployment should keep a human in the loop before any consequential action is taken.
- **Privacy and consent.** Running pose estimation and person-tracking over video footage of real individuals raises privacy considerations, particularly if deployed in spaces where people have a reasonable expectation of privacy, or without appropriate notice/signage in monitored areas. Deployment should comply with applicable local laws and organizational policies on video surveillance and biometric-adjacent data (pose keypoints are not identity-biometric, but tracking IDs persist across a session).
- **Bias and generalization.** The underlying pose model was trained on the COCO dataset, and its detection reliability can vary with clothing, lighting, camera angle, and body pose diversity represented in that training data. No claims are made in this project about equitable performance across demographic groups, camera conditions, or cultures, and this should be evaluated before any real-world deployment.
- **Scope of applicability.** This tool is a course project and academic prototype, not a validated safety system. It has not been evaluated against a standardized benchmark, audited for fairness, or stress-tested at production scale, and should not be relied upon as a sole safety mechanism.

## 9. Conclusion

This project demonstrates that a pretrained pose-estimation model can be repurposed, without any additional training, into a lightweight and fully explainable violent-activity flagging system, by combining simple, well-justified motion and proximity heuristics on top of tracked human keypoints. The resulting pipeline is transparent — every flagged event can be traced to a specific pair of people, a specific proximity value, and a specific motion value — fast to set up, requires no labeled dataset, and runs entirely on CPU.

While it is not as accurate as a purpose-trained end-to-end classifier, and carries known false-positive and false-negative failure modes documented in Section 7, it serves as a practical, inspectable baseline: a starting point that could be extended with learned components (Section 7.3) or deployed as-is in settings where explainability and ease of deployment matter more than maximal detection accuracy, always with human review in the loop.

## 10. References

- Ultralytics YOLOv8 documentation — pose estimation task overview and pretrained model details.
- COCO (Common Objects in Context) keypoints dataset — the 17-keypoint human pose annotation format used by the pretrained model.
- Hockey Fight and RWF-2000 — commonly cited benchmark datasets for learned video violence classification, referenced here for context rather than used directly in this project.
- ST-GCN and related skeleton-based action recognition literature — background for the "pose/motion-based heuristics" family of approaches discussed in Section 3.

---

<p align="center">
  Developed by <b>Mehul</b> (24BAI10631) · VIT-Yarthi Computer Vision Project
</p>
