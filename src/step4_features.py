"""
=== Step 4: Feature Extraction ===

Why do we need this?
  Raw (x, y, t) tells the model WHERE the pen is.
  But the model also needs to know:
    - Which DIRECTION is the pen moving? (angle)
    - How FAST is it moving? (speed)
    - Is the pen TURNING? (curvature)
    - Is the pen UP or DOWN? (pen_state)

  These "derived features" make patterns much easier to learn.

Input:  list of strokes, each stroke = [(x, y, t), ...]
Output: single flat sequence of [x, y, dx, dy, speed, angle, curvature, pen_state]
        where all strokes are concatenated, with pen_state=0 between strokes
"""

import math
import numpy as np
from tutorial import parse_stroke_file  # reuse our parser!


def extract_features(strokes):
    """
    Convert raw strokes into a feature sequence.

    For each point we compute 8 features:
      0. x          - normalized x coordinate
      1. y          - normalized y coordinate
      2. dx         - change in x from previous point
      3. dy         - change in y from previous point
      4. speed      - how fast the pen moved
      5. angle      - direction of movement (radians)
      6. curvature  - how much the direction changed
      7. pen_state  - 1.0 = pen down (drawing), 0.0 = pen up (between strokes)

    Returns: numpy array of shape (total_points, 8)
    """

    all_features = []

    for stroke_idx, stroke in enumerate(strokes):

        for i, (x, y, t) in enumerate(stroke):

            if i == 0:
                # first point in stroke: no previous point to compare
                dx = 0.0
                dy = 0.0
                speed = 0.0
                angle = 0.0
                curvature = 0.0
            else:
                prev_x, prev_y, prev_t = stroke[i - 1]
                dx = x - prev_x
                dy = y - prev_y
                dt = t - prev_t

                # speed = distance / time
                distance = math.sqrt(dx**2 + dy**2)
                speed = distance / dt if dt > 0 else 0.0

                # angle = direction of movement
                angle = math.atan2(dy, dx)

                # curvature = change in angle from previous step
                if i >= 2:
                    prev2_x, prev2_y, _ = stroke[i - 2]
                    prev_dx = prev_x - prev2_x
                    prev_dy = prev_y - prev2_y
                    prev_angle = math.atan2(prev_dy, prev_dx)
                    curvature = angle - prev_angle

                    # normalize to [-pi, pi]
                    while curvature > math.pi:
                        curvature -= 2 * math.pi
                    while curvature < -math.pi:
                        curvature += 2 * math.pi
                else:
                    curvature = 0.0

            pen_state = 1.0  # pen is down (drawing)

            all_features.append([x, y, dx, dy, speed, angle, curvature, pen_state])

        # === Add a "pen up" point between strokes ===
        # This tells the model: "the pen lifted here"
        if stroke_idx < len(strokes) - 1:
            next_stroke = strokes[stroke_idx + 1]
            # pen-up point at the start of the next stroke
            nx, ny, _ = next_stroke[0]
            all_features.append([nx, ny, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # pen_state = 0

    return np.array(all_features, dtype=np.float32)


def normalize_features(features):
    """
    Normalize x, y to [0, 1] range and clip extreme speed/curvature values.
    This helps the model train faster and more stably.
    """
    features = features.copy()

    # normalize x, y to [0, 1] based on min/max
    for col in [0, 1]:  # x and y columns
        min_val = features[:, col].min()
        max_val = features[:, col].max()
        rang = max_val - min_val
        if rang > 0:
            features[:, col] = (features[:, col] - min_val) / rang
        else:
            features[:, col] = 0.5

    # normalize dx, dy: divide by max absolute value so they're in [-1, 1]
    for col in [2, 3]:
        max_abs = np.abs(features[:, col]).max()
        if max_abs > 0:
            features[:, col] = features[:, col] / max_abs

    # clip speed to avoid extreme outliers
    speed_col = features[:, 4]
    p95 = np.percentile(speed_col[speed_col > 0], 95) if np.any(speed_col > 0) else 1.0
    features[:, 4] = np.clip(speed_col, 0, p95) / (p95 + 1e-6)

    # angle is already in [-pi, pi], normalize to [-1, 1]
    features[:, 5] = features[:, 5] / math.pi

    # curvature: clip and normalize
    features[:, 6] = np.clip(features[:, 6], -math.pi, math.pi) / math.pi

    # pen_state stays as 0 or 1
    return features


# ============================================================
# TEST IT
# ============================================================
if __name__ == "__main__":

    # Parse the stroke file
    strokes = parse_stroke_file("../data/901-1.txt")

    # Extract features
    raw_features = extract_features(strokes)
    print(f"Raw feature shape: {raw_features.shape}")
    print(f"  = {raw_features.shape[0]} time steps x {raw_features.shape[1]} features")
    print()

    # Show feature names
    feature_names = ["x", "y", "dx", "dy", "speed", "angle", "curvature", "pen_state"]
    print("First 5 points of Stroke 1:")
    print(f"  {'':>4}  " + "  ".join(f"{name:>10}" for name in feature_names))
    for i in range(5):
        vals = "  ".join(f"{raw_features[i, j]:>10.3f}" for j in range(8))
        print(f"  [{i}]  {vals}")
    print()

    # Normalize
    norm_features = normalize_features(raw_features)
    print(f"Normalized feature shape: {norm_features.shape}")
    print("First 5 points (normalized):")
    print(f"  {'':>4}  " + "  ".join(f"{name:>10}" for name in feature_names))
    for i in range(5):
        vals = "  ".join(f"{norm_features[i, j]:>10.3f}" for j in range(8))
        print(f"  [{i}]  {vals}")

    # Show pen-up points
    print(f"\nPen-up points (between strokes):")
    for i in range(len(norm_features)):
        if norm_features[i, 7] == 0.0:
            print(f"  Index {i}: pen lifted")
