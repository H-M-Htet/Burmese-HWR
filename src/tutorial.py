
"""
=== Burmese Handwriting Recognition - Step by Step Tutorial ===

This file teaches you the first building block:
  How to READ and VISUALIZE your stroke data.

Your data format (each .txt file):
  STROKE 1
  x y timestamp
  x y timestamp
  ...
  STROKE 2
  x y timestamp
  ...

Run this file: python tutorial.py
"""

import matplotlib.pyplot as plt


# ============================================================
# STEP 1: Parse a stroke file
# ============================================================
# This function reads a .txt file and returns a list of strokes.
# Each stroke is a list of (x, y, timestamp) tuples.
#
# Example output:
#   [
#     [(161, 175, 1781765469.49), (157, 178, 1781765469.50), ...],   # stroke 1
#     [(235, 195, 1781765471.84), (235, 195, 1781765471.84), ...],   # stroke 2
#     ...
#   ]

def parse_stroke_file(filepath):
    strokes = []
    current_stroke = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # skip empty lines
            if line == "":
                continue

            # new stroke header → save previous stroke, start fresh
            elif line.startswith("STROKE"):
                if len(current_stroke) > 0:
                    strokes.append(current_stroke)
                current_stroke = []

            # data line → parse x, y, t
            else:
                parts = line.split()
                x = float(parts[0])
                y = float(parts[1])
                t = float(parts[2])
                current_stroke.append((x, y, t))

    # save the last stroke
    if len(current_stroke) > 0:
        strokes.append(current_stroke)

    return strokes


# ============================================================
# STEP 2: Print stroke info
# ============================================================

def print_stroke_info(strokes, label=""):
    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"  Total strokes: {len(strokes)}")
    print(f"  Total points:  {sum(len(s) for s in strokes)}")
    print(f"{'='*40}")
    for i, stroke in enumerate(strokes):
        print(f"  Stroke {i+1}: {len(stroke)} points")
        print(f"    Start: ({stroke[0][0]:.0f}, {stroke[0][1]:.0f})")
        print(f"    End:   ({stroke[-1][0]:.0f}, {stroke[-1][1]:.0f})")
        duration = stroke[-1][2] - stroke[0][2]
        print(f"    Duration: {duration:.3f} seconds")


# ============================================================
# STEP 3: Plot strokes
# ============================================================

def plot_strokes(strokes, title="", save_path=None):
    colors = ['#FF0000', '#0066FF', '#00AA00', '#FF8800', '#9900CC', '#00CCCC']

    plt.figure(figsize=(6, 6))
    for i, stroke in enumerate(strokes):
        xs = [point[0] for point in stroke]
        ys = [point[1] for point in stroke]
        color = colors[i % len(colors)]

        # draw stroke line
        plt.plot(xs, ys, color=color, linewidth=2.5, label=f"Stroke {i+1}")
        # mark start point
        plt.plot(xs[0], ys[0], 'o', color=color, markersize=8)

    plt.gca().invert_yaxis()  # screen coords: y goes down
    plt.title(title, fontsize=14)
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to: {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================
# RUN IT
# ============================================================
if __name__ == "__main__":

    # === Change this path to your stroke file ===
    filepath = "../data/901-1.txt"

    # Parse
    strokes = parse_stroke_file(filepath)

    # Print info
    print_stroke_info(strokes, label="Sample 901-1")

    # Plot
    plot_strokes(strokes, title="Sample 901-1", save_path="output.png")

    print("\nDone! Check output.png")
    print("\n--- YOUR NEXT CHALLENGE ---")
    print("1. Try loading multiple files (901-1.txt, 901-2.txt, etc.)")
    print("2. Plot them side by side to see how your writing varies")
    print("3. Try plotting a different class and compare the shapes")

