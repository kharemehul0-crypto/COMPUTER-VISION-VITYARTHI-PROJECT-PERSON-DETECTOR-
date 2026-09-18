# Project Report: Pose-Based Violence Detection

**Author:** Mehul (24BAI10631) · **Course:** Computer Vision · **Project:** VIT-Yarthi Computer Vision Project

---

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Why I Went With This Approach](#2-why-i-went-with-this-approach)
- [3. Related Work](#3-related-work)
- [4. System Design](#4-system-design)
- [5. Implementation](#5-implementation)
- [6. How I Tested This](#6-how-i-tested-this)
- [7. What Works, What Doesn't](#7-what-works-what-doesnt)
- [8. Ethical Considerations](#8-ethical-considerations)
- [9. Conclusion](#9-conclusion)
- [10. References](#10-references)

---

## 1. Problem Statement

Detecting violent or aggressive physical interactions in video automatically — CCTV feeds, public safety cameras, crowd monitoring — is a problem that's been studied a lot, and for good reason: it has real applications in public safety, automated surveillance alerting, workplace incident detection, and reviewing footage after something has already happened. The core issue is that manual monitoring just doesn't scale. A person watching a dozen camera feeds at once gets tired and slow to react, and scrubbing through hours of archived footage after an incident is a huge time sink.

So there's a real need for systems that can either flag live footage for someone to review in close to real time, or narrow down archived footage to a handful of timestamps actually worth looking at, instead of making a human watch everything. Both of these come down to the same core requirement: something that looks at a stream of frames and spits out a short list of "hey, this looks suspicious, check timestamp X" events — ideally in a way where a human can actually understand *why* it got flagged, not just trust a black box.

That's what I set out to build here. Instead of training a dedicated violence classifier — something that takes in raw video and outputs "violent" or "not violent" — I went with a lighter, more explainable pipeline that flags segments where a possible violent interaction might be happening, using human pose estimation as a middle step rather than working directly on pixels.

## 2. Why I Went With This Approach

There are basically two families of approaches to this problem:

1. **End-to-end learned classifiers** — 3D CNNs, two-stream networks, video transformers, trained directly on labeled violent/non-violent clips from datasets like Hockey Fight, RWF-2000, or Real Life Violence Situations. These models figure out on their own whatever spatiotemporal features best separate the two classes, without anyone telling them what "violence" is supposed to look like.
2. **Pose or motion-based heuristics** — run an off-the-shelf pose estimator over the video first to get a compact skeleton representation of each person, then apply interpretable rules (or a lightweight model) on top of that structured data instead of on raw pixels.

Given what I actually had to work with — no access to a big labeled violence dataset, and no GPU training setup in the environment this gets evaluated in — I went with option 2. And honestly, beyond just being the practical choice, it had some real advantages for this specific situation:

| Advantage | Why it matters here |
|---|---|
| **No training data needed** | YOLOv8-pose is already pretrained on COCO keypoints, and I'm just using it as a fixed feature extractor. I didn't need to go collect or label any violence-specific footage. |
| **Fully explainable** | Every flagged event traces back to exactly which two tracked people were involved, how close they were, and how fast their arms were moving at that moment. Compare that to a black-box neural net, where you'd need extra tooling like saliency maps just to guess at why it decided what it decided. |
| **Runs on CPU** | No GPU needed at inference time, which kept setup simple and means this actually runs wherever it needs to get evaluated, without worrying about GPU availability. |

The trade-off — and I want to be upfront about this rather than bury it — is that a hand-crafted heuristic like this is generally going to be less accurate and throw more false positives than a properly trained end-to-end model that's seen a large, diverse labeled dataset. I go into this more in Section 7. Given the goal here was a transparent, easy-to-run baseline rather than a production-grade detector, I think that trade-off is a reasonable one to make.

## 3. Related Work

It helps to place this project within the broader "violence detection in video" space, which splits into a few distinct lines of work:

- **Handcrafted motion features.** Before deep learning took over this space, people used things like optical flow histograms, motion blobs, and acceleration patterns to characterize aggressive motion, feeding those into classical classifiers like SVMs. The motion score I use here is basically a conceptual descendant of that line of thinking, just using structured keypoint displacement instead of dense optical flow.
- **End-to-end deep video classifiers.** More recent work uses 3D-CNNs (C3D, I3D) or two-stream architectures trained on labeled violence datasets. These generally beat handcrafted-feature approaches once you have enough labeled data — the cost is you need that data, and you lose a lot of interpretability.
- **Skeleton/pose-based action recognition.** There's a decent body of work doing action recognition — including fight/aggression detection — on extracted skeletons rather than raw pixels, often using graph convolutional networks (ST-GCN and variants) over the keypoint sequences. My project sits at the simplest end of this spectrum: instead of training a graph network over pose sequences, I just apply a small set of interpretable, hand-derived rules directly to the keypoint trajectories.
- **Anomaly detection framing.** Some surveillance systems frame this as unsupervised anomaly detection instead — learn what "normal" looks like for a given scene and flag deviations from that, rather than doing supervised violence/non-violence classification. I didn't take this route; what I built is closer to a supervised-by-hand-written-rules heuristic than a learned normalcy model.

So where does this project fit? It's closest in spirit to the skeleton-based line of work, but I deliberately skipped any additional model training and just relied on interpretable geometric rules applied to a pretrained pose estimator's output.

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

Every input frame goes through a pretrained YOLOv8-pose model — I used the `yolov8n-pose.pt` nano variant by default since it's fast enough to run on CPU. For every person the model detects above the confidence threshold (`--conf`, default `0.4`), it gives back:

- A bounding box
- 17 keypoints in the COCO format (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), each with its own `(x, y)` coordinate and confidence score

On top of the raw per-frame detections, I use Ultralytics' built-in multi-object tracker to give each person a persistent ID across frames. This is what actually makes it possible to compute frame-to-frame motion for "the same person" — without it, I'd just be comparing unrelated detections from one frame to the next, which wouldn't mean anything.

### 4.3 Motion Score

For a tracked person `p`, I compute the motion score at frame `t` roughly like this:

```
motion_score(p, t) = mean(displacement of left/right wrist and elbow keypoints, t-1 → t)
                      -------------------------------------------------------------------
                                        torso_length(p, t)
```

where `torso_length` comes from the distance between the shoulder and hip keypoints. Dividing by torso length is a per-person, per-frame normalization: someone who takes up more pixels in the frame (because they're closer to the camera, or the shot is more zoomed in) will naturally show bigger raw pixel movements for the exact same physical action. Normalizing this way keeps the score roughly comparable regardless of how far someone is from the camera or how big they are.

I only use wrist and elbow keypoints for this, not all 17 — the reasoning for that is in 4.5 below.

### 4.4 Proximity Score

For every pair of people currently being tracked in a frame, I compute a proximity score from the distance between their body centers (approximated from the midpoint of hips and shoulders), again normalized by the average of their torso lengths, so "close" is a relative, scale-invariant idea rather than a fixed pixel distance.

### 4.5 A Few Design Choices Worth Explaining

- **Why normalize by torso length at all?** Someone standing closer to the camera will always show bigger pixel movements doing the exact same physical action as someone farther away. Dividing by torso length in pixels makes the motion score roughly scale-invariant, so one `--motion-threshold` value behaves consistently no matter how far away or zoomed-in the camera is, instead of needing retuning for every new shot.
- **Why only wrists and elbows, not the whole body?** Arm movement — punches, pushes, swings — is the most visually distinctive signal of physical aggression between two people, compared to leg or torso movement, which happens constantly just from walking or shifting weight and isn't specific to an aggressive interaction. Sticking to arms also cuts down on noise from camera shake or someone just swaying in place.
- **Why require proximity AND motion together, not either alone?** High motion by itself gets triggered by tons of harmless stuff — waving, stretching, exercising, dancing solo, someone pacing while agitated. Proximity by itself gets triggered constantly too — any two people standing near each other, which happens in queues, conversations, crowded rooms, whatever. Requiring both at once narrows things down specifically to close-contact, high-energy interactions, which cuts false positives a lot compared to using either signal on its own.
- **Why require N consecutive frames (`--consecutive-frames`, default 5)?** Pose estimation is noisy from frame to frame — a single misdetected or jittery keypoint can cause a big spurious one-frame "jump" in the motion score even when the person barely moved. Requiring the rule to hold across several frames in a row filters out that kind of single-frame noise. The trade-off is a small delay — roughly `N / fps` seconds — before an ongoing event actually gets logged.

## 5. Implementation

**Language:** Python 3 (I tested on 3.10, should work fine on 3.9+)

**Libraries I used:**

| Library | What it's doing |
|---|---|
| `ultralytics` | Gives me the pretrained YOLOv8-pose model and the built-in tracker for pose estimation and keeping identities consistent |
| `opencv-python` | Handles reading/writing video and drawing all the overlays — skeletons, boxes, the alert border |
| `numpy` | The array math underneath the keypoint distance and normalization calculations |

**How the code is organized:**

| File | What's in it |
|---|---|
| `utils.py` | Pure functions for the keypoint math — motion score, proximity score, torso-length normalization. I kept this separate from the CLI logic so these functions could be tested or reused on their own if needed. |
| `detect_violence.py` | The actual CLI entry point. Opens the video, runs pose tracking frame by frame, keeps a history of keypoints per person, applies the proximity-and-motion rule with the consecutive-frame check, draws everything, and writes out the video and JSON log. |

It's entirely command-line driven (`python detect_violence.py --source input.mp4`) — no GUI, since that wasn't a requirement I needed to meet. Every tunable setting — thresholds, output paths, which model to use — is exposed as a CLI flag instead of being hardcoded, so it can be retuned for different footage without touching the source code.

## 6. How I Tested This

Since I didn't have a labeled violence dataset to work with, my evaluation here was more qualitative than a formal precision/recall-against-ground-truth setup. What I actually did was run the pipeline against a small set of clips — some I recorded myself, some publicly available — split into three categories:

1. **Obvious positive cases** — staged close-contact, high-arm-motion interactions (mock pushing, mock grappling) meant to resemble what an actual altercation looks like physically.
2. **Obvious negative cases** — normal stuff like walking, standing around, talking at a normal distance, just to check the system doesn't flag everyday footage.
3. **Adversarial negative cases** — activities I picked specifically because they share surface-level traits with violence (close proximity + fast arm motion) without actually being violent — hugging, high-fives, casual dancing. These were meant to poke at and understand the false-positive behavior I talk about in Section 7.

For each clip, I manually went through the annotated output video and the `events.json` log to check whether the flagged segments actually lined up with what I intended, and adjusted the thresholds (`--motion-threshold`, `--proximity-threshold`, `--consecutive-frames`) based on what I saw. The default values shipped with the tool came out of this back-and-forth tuning process on a small sample — they're reasonable starting points, not numbers validated against some large, representative benchmark. Worth being honest about that.

## 7. What Works, What Doesn't

### 7.1 What Works Well

- Reliably picks up two people in close, high-motion contact — pushing, grappling, fast arm swinging directed at another person.
- Gives clear, timestamped output that's actually readable, so a human reviewer can jump straight to the relevant few seconds instead of watching the whole video.
- Fully deterministic and tunable through CLI flags — you can adjust sensitivity for a new camera angle or setting without retraining anything.
- Runs entirely on CPU at a decent frame rate, so it's usable in places where a GPU-based end-to-end classifier just wouldn't be practical.

### 7.2 Where It Falls Short

| Limitation | What's going on |
|---|---|
| **False positives** | Non-violent stuff involving two people close together with fast arm motion — enthusiastic hugging, playful roughhousing, some kinds of dancing, sports — can set off a false alert. The system is really detecting "high-energy, close-contact motion," which overlaps with violence but isn't the same thing. |
| **False negatives** | Slow, low-motion aggression — a static choke-hold, a threat with a weapon but not much arm movement, intimidation without physical contact — won't get flagged, since the whole thing hinges on fast arm displacement. |
| **Occlusion sensitivity** | If people overlap a lot or their keypoints get blocked from view — which happens exactly in the close-contact scenarios I'm trying to catch — YOLOv8-pose's confidence drops, which can mean missed detections, the tracker swapping IDs, or noisy motion scores. |
| **No real understanding of "violence"** | Unlike a model trained on labeled examples, this thing has no learned concept of violence beyond motion and proximity. It can't judge intent, and it won't generalize to aggressive situations that don't match this specific motion signature — verbal aggression, someone threatening with a weapon but standing still, etc. |
| **Single camera, 2D only** | Everything is measured from 2D pixel coordinates in one camera view — there's no depth information, so "proximity" is really just a projection-based approximation. Camera angle can throw this off (two people can look close together in a wide shot while actually being several meters apart). |

### 7.3 What I'd Do Differently / Next

- Swap the hand-crafted rule for a small trained classifier (a lightweight MLP, LSTM, or small graph convolutional network) that takes the same keypoint-derived features as input but learns the decision boundary from a small labeled clip dataset, instead of me guessing at thresholds — combining the interpretability of pose features with something actually learned.
- Bring in facial expression or audio signals (raised voices, screaming, sudden loud sounds) as extra evidence alongside motion, so it's not relying on arm motion alone.
- Replace the built-in tracker with something more robust like ByteTrack or DeepSORT for crowded or high-occlusion scenes.
- Add temporal smoothing or a Kalman filter over the raw keypoint trajectories to cut down the frame-to-frame jitter that's the whole reason I need the consecutive-frame requirement in the first place — this could let the system react faster.
- Actually collect a small labeled evaluation set for whatever the target scenario is, and report real precision/recall numbers instead of relying on the qualitative review I did here.

## 8. Ethical Considerations

Building something that flags "possible violent interactions" from video comes with real-world weight beyond just how technically accurate it is, and I think it's worth spelling this out rather than glossing over it:

- **This should flag for human review, not act on its own.** This tool is meant to flag candidate segments for a person to look at — not to make automated decisions like alerting law enforcement or locking down a building by itself. Given the false-positive rate I've already talked about, there should always be a human in the loop before anything consequential happens based on this.
- **Privacy and consent matter here.** Running pose estimation and tracking over footage of real people raises privacy questions, especially in spaces where people reasonably expect privacy, or where there's no clear notice that they're being monitored. Any real deployment needs to follow local laws and organizational policy around video surveillance — pose keypoints aren't identity-biometric data exactly, but tracking IDs do persist across a session, which is still worth being careful about.
- **Bias and how well this generalizes.** The pose model underneath all this was trained on COCO, and how reliably it detects people can shift depending on clothing, lighting, camera angle, and how diverse the training data actually was in terms of body types and poses. I'm not making any claims here about equal performance across different demographic groups, camera setups, or contexts — that would need real evaluation before this went anywhere near an actual deployment.
- **This is a course project, not a safety system.** I want to be clear that this is an academic prototype, not something validated against a standard benchmark, audited for fairness, or stress-tested at any real scale. It shouldn't be treated as a sole safety mechanism for anything.

## 9. Conclusion

What this project shows is that you can take a pretrained pose-estimation model and, without any additional training, turn it into a lightweight, fully explainable system for flagging potentially violent activity — just by combining a couple of simple, well-reasoned motion and proximity rules on top of tracked keypoints. The end result is transparent (every flagged event traces back to a specific pair of people and specific proximity/motion numbers), quick to set up, needs no labeled dataset, and runs entirely on CPU.

It's not going to be as accurate as a properly trained end-to-end classifier, and it has real false-positive and false-negative issues that I've tried to be upfront about in Section 7. But I think it works well as a practical, inspectable baseline — something that could either be extended with learned components down the line, or used as-is in situations where being able to explain what the system is doing and getting it running quickly matter more than squeezing out maximum accuracy, always with a person reviewing the output.

## 10. References

- Ultralytics YOLOv8 documentation — pose estimation task overview and pretrained model details.
- COCO (Common Objects in Context) keypoints dataset — the 17-keypoint human pose format the pretrained model uses.
- Hockey Fight and RWF-2000 — commonly cited benchmark datasets for learned video violence classification, mentioned here for context rather than something I actually used.
- ST-GCN and related skeleton-based action recognition literature — background for the pose/motion-based heuristics approach discussed in Section 3.

---

<p align="center">
  Developed by <b>Mehul</b> (24BAI10631) · VIT-Yarthi Computer Vision Project
</p>