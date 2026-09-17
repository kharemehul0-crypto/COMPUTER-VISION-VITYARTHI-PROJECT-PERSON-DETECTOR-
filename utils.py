"""
utils.py
--------
Helper functions used by detect_violence.py.

These functions turn raw YOLOv8-pose keypoints into two simple numeric
signals that we use to decide whether "violent" motion is happening:

1. Motion score   -> how much a single person's body moved between two
                     consecutive frames (normalized so it doesn't depend
                     on how far the person is from the camera).
2. Proximity score -> how close two people are to each other, because
                     violent interactions almost always involve two
                     people being close together.

Combining "high motion" + "high proximity" is the core heuristic used
in detect_violence.py to flag a frame as a possible violent event.
"""

import numpy as np

# YOLOv8-pose returns 17 COCO keypoints per person, in this order:
# 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows,
# 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles
# We only need shoulders/hips to estimate a rough body size (for
# normalization) and wrists/elbows for detecting fast arm movement.

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_ELBOW, RIGHT_ELBOW = 7, 8


def keypoints_to_array(keypoints_xy):
    """
    Convert a single person's keypoints (list/array of [x, y] pairs)
    into a clean numpy array. Missing/low-confidence points from
    Ultralytics are returned as [0, 0]; we leave them as-is and rely on
    body_scale() + nan-safe averaging elsewhere to reduce their impact.
    """
    return np.array(keypoints_xy, dtype=np.float32)


def body_scale(keypoints):
    """
    Estimate a rough 'size' of the person in pixels, using the distance
    between the shoulder midpoint and the hip midpoint (the torso
    length). This lets us normalize motion so a person close to the
    camera and a person far away are compared fairly.

    Returns a small positive number even if keypoints are missing, to
    avoid division by zero.
    """
    ls, rs = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    lh, rh = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]

    shoulder_mid = (ls + rs) / 2.0
    hip_mid = (lh + rh) / 2.0

    torso_len = np.linalg.norm(shoulder_mid - hip_mid)
    if torso_len < 1e-3:
        torso_len = 50.0  # fallback default so downstream math is stable

    return torso_len


def motion_score(prev_keypoints, curr_keypoints):
    """
    Compare a person's keypoints between the previous frame and the
    current frame. Returns a single normalized number representing how
    much that person's body moved.

    We focus on the arms (wrists + elbows) because fast arm movement
    (swinging, punching, pushing) is the most common visual signature
    of violent action, more so than legs or torso.
    """
    if prev_keypoints is None or curr_keypoints is None:
        return 0.0

    joints_of_interest = [LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW]

    displacements = []
    for j in joints_of_interest:
        prev_pt = prev_keypoints[j]
        curr_pt = curr_keypoints[j]
        # Skip joints Ultralytics couldn't detect in either frame (reported as 0,0)
        if np.all(prev_pt == 0) or np.all(curr_pt == 0):
            continue
        displacements.append(np.linalg.norm(curr_pt - prev_pt))

    if not displacements:
        return 0.0

    scale = body_scale(curr_keypoints)
    avg_displacement = float(np.mean(displacements))

    # Normalize: "how many torso-lengths did the arms move this frame"
    return avg_displacement / scale


def person_center(keypoints):
    """
    Approximate a person's on-screen position using the midpoint of
    their shoulders and hips (more stable than using the full bounding
    box center when limbs are flailing around).
    """
    ls, rs = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
    lh, rh = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
    pts = np.array([ls, rs, lh, rh])
    valid = pts[np.any(pts != 0, axis=1)]
    if len(valid) == 0:
        return None
    return valid.mean(axis=0)


def proximity_score(keypoints_a, keypoints_b):
    """
    Returns a normalized closeness score between two people: higher
    means closer together (relative to their own body size). Used to
    require that flagged 'violent motion' is happening between two
    people who are actually near each other, not two unrelated people
    on opposite sides of the frame.
    """
    center_a = person_center(keypoints_a)
    center_b = person_center(keypoints_b)
    if center_a is None or center_b is None:
        return 0.0

    dist = np.linalg.norm(center_a - center_b)
    avg_scale = (body_scale(keypoints_a) + body_scale(keypoints_b)) / 2.0

    # Convert distance into a 0-1 "closeness" score: 0 = far apart,
    # 1 = essentially touching. Capped so far-apart people always score 0.
    normalized_dist = dist / (avg_scale * 3.0)  # 3 torso-lengths = "far"
    closeness = max(0.0, 1.0 - normalized_dist)
    return closeness
