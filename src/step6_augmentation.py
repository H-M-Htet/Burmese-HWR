"""
=== Step 6: Data Augmentation ===

Why is this critical?
  You have ~3.9 samples per class. That's not enough for a model to learn.
  Augmentation creates "new" samples by slightly changing existing ones.
  
  Think of it like this: your handwriting varies slightly every time.
  Augmentation simulates that natural variation.

These augmentations work on RAW strokes (before feature extraction),
because transforming x,y coordinates is more natural than transforming
derived features like speed and angle.

Usage:
  augmented_strokes = augment_strokes(original_strokes)
  features = extract_features(augmented_strokes)
"""

import math
import random
import numpy as np


def augment_strokes(strokes, intensity=1.0):
    """
    Apply a random combination of augmentations to strokes.
    
    Args:
        strokes: list of strokes, each = [(x, y, t), ...]
        intensity: 0.0 = no augmentation, 1.0 = normal, 2.0 = aggressive
    
    Returns:
        new list of augmented strokes (original is not modified)
    """
    # Deep copy so we don't modify the original
    strokes = [[(x, y, t) for x, y, t in stroke] for stroke in strokes]

    # Randomly apply each augmentation
    # Each one has a probability of being applied

    if random.random() < 0.8:
        strokes = jitter(strokes, sigma=2.0 * intensity)

    if random.random() < 0.5:
        strokes = scale(strokes, sigma=0.15 * intensity)

    if random.random() < 0.5:
        strokes = rotate(strokes, max_angle=15 * intensity)

    if random.random() < 0.4:
        strokes = translate(strokes, sigma=10 * intensity)

    if random.random() < 0.3:
        strokes = speed_warp(strokes, sigma=0.3 * intensity)

    if random.random() < 0.2:
        strokes = point_dropout(strokes, drop_rate=0.1 * intensity)

    return strokes


# ============================================================
# INDIVIDUAL AUGMENTATIONS
# ============================================================

def jitter(strokes, sigma=2.0):
    """
    Add small random noise to x,y coordinates.
    Simulates natural hand tremor / imprecision.
    
    sigma: standard deviation of noise in pixels
    """
    result = []
    for stroke in strokes:
        new_stroke = []
        for x, y, t in stroke:
            x += random.gauss(0, sigma)
            y += random.gauss(0, sigma)
            new_stroke.append((x, y, t))
        result.append(new_stroke)
    return result


def scale(strokes, sigma=0.15):
    """
    Randomly resize the writing (bigger or smaller).
    Simulates writing at different sizes.
    
    sigma: controls how much the scale can vary
           e.g., 0.15 means 85%-115% of original size
    """
    # Find center of all strokes
    all_x = [x for stroke in strokes for x, y, t in stroke]
    all_y = [y for stroke in strokes for x, y, t in stroke]
    cx = sum(all_x) / len(all_x)
    cy = sum(all_y) / len(all_y)

    # Random scale factor
    scale_x = 1.0 + random.gauss(0, sigma)
    scale_y = 1.0 + random.gauss(0, sigma)

    # Scale around center
    result = []
    for stroke in strokes:
        new_stroke = []
        for x, y, t in stroke:
            x = cx + (x - cx) * scale_x
            y = cy + (y - cy) * scale_y
            new_stroke.append((x, y, t))
        result.append(new_stroke)
    return result


def rotate(strokes, max_angle=15):
    """
    Randomly rotate the writing slightly.
    Simulates tilted writing.
    
    max_angle: maximum rotation in degrees
    """
    # Find center
    all_x = [x for stroke in strokes for x, y, t in stroke]
    all_y = [y for stroke in strokes for x, y, t in stroke]
    cx = sum(all_x) / len(all_x)
    cy = sum(all_y) / len(all_y)

    # Random angle
    angle = random.uniform(-max_angle, max_angle)
    angle_rad = math.radians(angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # Rotate around center
    result = []
    for stroke in strokes:
        new_stroke = []
        for x, y, t in stroke:
            dx = x - cx
            dy = y - cy
            new_x = cx + dx * cos_a - dy * sin_a
            new_y = cy + dx * sin_a + dy * cos_a
            new_stroke.append((new_x, new_y, t))
        result.append(new_stroke)
    return result


def translate(strokes, sigma=10):
    """
    Randomly shift the entire writing left/right/up/down.
    Simulates writing in a slightly different position.
    
    sigma: standard deviation of shift in pixels
    """
    shift_x = random.gauss(0, sigma)
    shift_y = random.gauss(0, sigma)

    result = []
    for stroke in strokes:
        new_stroke = [(x + shift_x, y + shift_y, t) for x, y, t in stroke]
        result.append(new_stroke)
    return result


def speed_warp(strokes, sigma=0.3):
    """
    Randomly stretch/compress timestamps.
    Simulates writing faster or slower.
    
    This affects the speed and timing features
    without changing the shape.
    """
    warp_factor = 1.0 + random.gauss(0, sigma)
    warp_factor = max(0.5, min(2.0, warp_factor))  # clamp to [0.5, 2.0]

    result = []
    for stroke in strokes:
        if len(stroke) == 0:
            result.append(stroke)
            continue
        t0 = stroke[0][2]  # first timestamp
        new_stroke = []
        for x, y, t in stroke:
            new_t = t0 + (t - t0) * warp_factor
            new_stroke.append((x, y, new_t))
        result.append(new_stroke)
    return result


def point_dropout(strokes, drop_rate=0.1):
    """
    Randomly remove some points from strokes.
    Simulates lower sampling rate or skipped points.
    
    drop_rate: fraction of points to remove (0.1 = 10%)
    Never removes first or last point of a stroke.
    """
    result = []
    for stroke in strokes:
        if len(stroke) <= 3:
            # too short to drop anything
            result.append(stroke)
            continue
        new_stroke = [stroke[0]]  # always keep first
        for point in stroke[1:-1]:
            if random.random() > drop_rate:
                new_stroke.append(point)
        new_stroke.append(stroke[-1])  # always keep last
        result.append(new_stroke)
    return result


# ============================================================
# VISUALIZE AUGMENTATION (to see what it does)
# ============================================================

def demo_augmentation():
    """Show original vs augmented side by side"""
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, '.')

    from tutorial import parse_stroke_file

    filepath = "data/strokes/901-1.txt"  # change to your path
    original = parse_stroke_file(filepath)

    colors = ['#FF0000', '#0066FF', '#00AA00', '#FF8800', '#9900CC']

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Original vs 7 Augmented Versions", fontsize=14)

    # Plot original
    ax = axes[0][0]
    for i, stroke in enumerate(original):
        xs = [p[0] for p in stroke]
        ys = [p[1] for p in stroke]
        ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=2)
    ax.invert_yaxis()
    ax.set_title("ORIGINAL", fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    # Plot 7 augmented versions
    for idx in range(1, 8):
        row = idx // 4
        col = idx % 4
        ax = axes[row][col]

        augmented = augment_strokes(original)
        for i, stroke in enumerate(augmented):
            xs = [p[0] for p in stroke]
            ys = [p[1] for p in stroke]
            ax.plot(xs, ys, color=colors[i % len(colors)], linewidth=2)
        ax.invert_yaxis()
        ax.set_title(f"Augmented #{idx}")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("augmentation_demo.png", dpi=150, bbox_inches='tight')
    print("Saved augmentation_demo.png")
    plt.close()


if __name__ == "__main__":
    demo_augmentation()