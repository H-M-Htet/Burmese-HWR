"""
=== Step 5: PyTorch Dataset ===

This is the bridge between your raw data and the model.
It does 3 things:
  1. Scans your data folder and finds all {class_id}-{sample_id}.txt files
  2. Extracts features from each file
  3. Pads/truncates sequences to the same length (required for batching)

Usage:
  dataset = StrokeDataset(data_dir="data/strokes", labels_file="data/labels.txt")
  features, label = dataset[0]   # returns one sample
  dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import os
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PARSING (from tutorial.py)
# ============================================================

def parse_stroke_file(filepath):
    """Read a stroke .txt file → list of strokes, each stroke = [(x,y,t), ...]"""
    strokes = []
    current_stroke = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            elif line.startswith("STROKE"):
                if len(current_stroke) > 0:
                    strokes.append(current_stroke)
                current_stroke = []
            else:
                parts = line.split()
                x = float(parts[0])
                y = float(parts[1])
                t = float(parts[2])
                current_stroke.append((x, y, t))
    if len(current_stroke) > 0:
        strokes.append(current_stroke)
    return strokes


# ============================================================
# FEATURE EXTRACTION (from step4_features.py)
# ============================================================

def extract_features(strokes):
    """Convert raw strokes → feature array of shape (total_points, 8)"""
    all_features = []

    for stroke_idx, stroke in enumerate(strokes):
        for i, (x, y, t) in enumerate(stroke):
            if i == 0:
                dx, dy, speed, angle, curvature = 0.0, 0.0, 0.0, 0.0, 0.0
            else:
                prev_x, prev_y, prev_t = stroke[i - 1]
                dx = x - prev_x
                dy = y - prev_y
                dt = t - prev_t
                distance = math.sqrt(dx**2 + dy**2)
                speed = distance / dt if dt > 0 else 0.0
                angle = math.atan2(dy, dx)
                if i >= 2:
                    prev2_x, prev2_y, _ = stroke[i - 2]
                    prev_angle = math.atan2(prev_y - prev2_y, prev_x - prev2_x)
                    curvature = angle - prev_angle
                    while curvature > math.pi:
                        curvature -= 2 * math.pi
                    while curvature < -math.pi:
                        curvature += 2 * math.pi
                else:
                    curvature = 0.0

            all_features.append([x, y, dx, dy, speed, angle, curvature, 1.0])

        # pen-up point between strokes
        if stroke_idx < len(strokes) - 1:
            nx, ny, _ = strokes[stroke_idx + 1][0]
            all_features.append([nx, ny, 0, 0, 0, 0, 0, 0.0])

    return np.array(all_features, dtype=np.float32)


def normalize_features(features):
    """Normalize all features to reasonable ranges"""
    features = features.copy()

    # x, y → [0, 1]
    for col in [0, 1]:
        min_val = features[:, col].min()
        max_val = features[:, col].max()
        rang = max_val - min_val
        if rang > 0:
            features[:, col] = (features[:, col] - min_val) / rang
        else:
            features[:, col] = 0.5

    # dx, dy → [-1, 1]
    for col in [2, 3]:
        max_abs = np.abs(features[:, col]).max()
        if max_abs > 0:
            features[:, col] = features[:, col] / max_abs

    # speed → [0, 1]
    speed_col = features[:, 4]
    p95 = np.percentile(speed_col[speed_col > 0], 95) if np.any(speed_col > 0) else 1.0
    features[:, 4] = np.clip(speed_col, 0, p95) / (p95 + 1e-6)

    # angle → [-1, 1]
    features[:, 5] = features[:, 5] / math.pi

    # curvature → [-1, 1]
    features[:, 6] = np.clip(features[:, 6], -math.pi, math.pi) / math.pi

    return features


# ============================================================
# PAD OR TRUNCATE
# ============================================================

def pad_or_truncate(features, max_len):
    """
    Make all sequences the same length.

    Why? PyTorch needs all tensors in a batch to have the same shape.
    - If sequence is shorter than max_len → pad with zeros at the end
    - If sequence is longer than max_len → truncate (cut off the end)

    Also returns the actual length (before padding) so the model
    can ignore the padded zeros.
    """
    actual_len = len(features)
    num_features = features.shape[1]  # 8

    if actual_len >= max_len:
        # truncate
        return features[:max_len], max_len
    else:
        # pad with zeros
        padded = np.zeros((max_len, num_features), dtype=np.float32)
        padded[:actual_len] = features
        return padded, actual_len


# ============================================================
# DATASET CLASS
# ============================================================

class StrokeDataset(Dataset):
    """
    PyTorch Dataset for Burmese handwriting stroke data.

    Folder structure expected:
      data_dir/
        1-1.txt
        1-2.txt
        1-3.txt
        2-1.txt
        ...
        1000-4.txt

    labels_file: text file with one syllable per line (line 1 = class 0)
    """

    def __init__(self, data_dir, labels_file=None, max_len=300, max_classes=None):
        """
        Args:
            data_dir:    folder containing all {class_id}-{sample_id}.txt files
            labels_file: path to syllable list (optional, for display)
            max_len:     pad/truncate all sequences to this length
            max_classes: limit number of classes (None = use all found)
        """
        self.data_dir = data_dir
        self.max_len = max_len
        self.num_features = 8

        # Load syllable labels (optional, just for display)
        self.labels = []
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]

        # === Scan the folder for all .txt files ===
        self.samples = []  # list of (filepath, class_index)
        self._scan_files(max_classes)

        # === Pre-load and process all data ===
        # (With ~4000 files, this fits easily in memory)
        self.features_list = []
        self.lengths_list = []
        self.labels_list = []
        self._load_all()

        print(f"Dataset loaded:")
        print(f"  Directory:    {data_dir}")
        print(f"  Total samples: {len(self.features_list)}")
        print(f"  Num classes:   {self.num_classes}")
        print(f"  Max seq len:   {max_len}")
        print(f"  Feature dim:   {self.num_features}")
        if self.raw_lengths:
            print(f"  Seq lengths:   min={min(self.raw_lengths)}, "
                  f"max={max(self.raw_lengths)}, "
                  f"avg={sum(self.raw_lengths)/len(self.raw_lengths):.0f}")

    def _scan_files(self, max_classes):
        """Find all {class_id}-{sample_id}.txt files and map class IDs"""

        # Find all txt files matching the pattern
        all_files = glob.glob(os.path.join(self.data_dir, "*.txt"))

        # Parse filenames to get (class_id, sample_id, filepath)
        file_info = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            # skip non-data files (like labels.txt or syl.txt)
            if "-" not in fname:
                continue
            try:
                name_part = fname.replace(".txt", "")
                class_id_str, sample_id_str = name_part.split("-")
                class_id = int(class_id_str)
                file_info.append((class_id, fpath))
            except (ValueError, IndexError):
                continue  # skip files that don't match the pattern

        if not file_info:
            print(f"WARNING: No stroke files found in {self.data_dir}")
            self.num_classes = 0
            return

        # Sort by class_id
        file_info.sort(key=lambda x: x[0])

        # Get unique class IDs and create mapping
        unique_classes = sorted(set(ci for ci, _ in file_info))
        if max_classes:
            unique_classes = unique_classes[:max_classes]

        # Map original class_id (1-based) → index (0-based)
        # class_id 1 → index 0, class_id 2 → index 1, etc.
        self.class_id_to_index = {cid: idx for idx, cid in enumerate(unique_classes)}
        self.index_to_class_id = {idx: cid for cid, idx in self.class_id_to_index.items()}
        self.num_classes = len(unique_classes)

        # Build sample list
        for class_id, fpath in file_info:
            if class_id in self.class_id_to_index:
                self.samples.append((fpath, self.class_id_to_index[class_id]))

    def _load_all(self):
        """Pre-load all files, extract features, pad sequences"""
        self.raw_lengths = []

        for filepath, class_index in self.samples:
            try:
                strokes = parse_stroke_file(filepath)
                if not strokes:
                    continue

                features = extract_features(strokes)
                features = normalize_features(features)
                self.raw_lengths.append(len(features))

                features_padded, actual_len = pad_or_truncate(features, self.max_len)

                self.features_list.append(features_padded)
                self.lengths_list.append(actual_len)
                self.labels_list.append(class_index)

            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
                continue

    def __len__(self):
        return len(self.features_list)

    def __getitem__(self, idx):
        """
        Returns:
            features: tensor of shape (max_len, 8)
            label:    integer class index
            length:   actual sequence length (before padding)
        """
        features = torch.tensor(self.features_list[idx], dtype=torch.float32)
        label = torch.tensor(self.labels_list[idx], dtype=torch.long)
        length = self.lengths_list[idx]
        return features, label, length

    def get_syllable(self, class_index):
        """Get the syllable text for a class index (for display)"""
        class_id = self.index_to_class_id.get(class_index, -1)
        if 0 < class_id <= len(self.labels):
            return self.labels[class_id - 1]  # labels are 0-indexed, class_ids are 1-indexed
        return f"class_{class_index}"


# ============================================================
# TEST IT
# ============================================================
if __name__ == "__main__":

    # === Test with whatever files we have ===
    # Change this to your actual data folder
    DATA_DIR = "data/strokes"  # <-- UPDATE THIS PATH
    LABELS_FILE = "data/label/syl.txt"

    dataset = StrokeDataset(
        data_dir=DATA_DIR,
        labels_file=LABELS_FILE,
        max_len=300,
    )

    if len(dataset) > 0:
        # Get one sample
        features, label, length = dataset[0]
        print(f"\nSample 0:")
        print(f"  Features shape: {features.shape}")
        print(f"  Label:          {label.item()}")
        print(f"  Actual length:  {length}")
        print(f"  Syllable:       {dataset.get_syllable(label.item())}")

        # Test DataLoader (this is what the training loop uses)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        for batch_features, batch_labels, batch_lengths in dataloader:
            print(f"\nBatch test:")
            print(f"  Features batch shape: {batch_features.shape}")
            print(f"  Labels batch shape:   {batch_labels.shape}")
            print(f"  Lengths:              {batch_lengths.tolist()}")
            break

        print("\n✓ Dataset is working! Ready for model training.")

    else:
        print("\nNo samples loaded. Make sure your stroke files")
        print("are in the data directory with format: {class_id}-{sample_id}.txt")
        print(f"\nExample: put files like 1-1.txt, 1-2.txt, ... in {DATA_DIR}")
