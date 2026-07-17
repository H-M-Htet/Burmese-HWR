"""
=== Step 5b: Dataset with Augmentation ===

Updated StrokeDataset that applies augmentation on-the-fly during training.
Each time you access a sample, it gets a DIFFERENT random augmentation.
This means every epoch the model sees different variations.

With ~4 samples per class and ~10x augmentation effect,
the model effectively sees ~40 variations per class per epoch.
"""

import os
import glob
import math
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader

# Import augmentation
from step6_augmentation import augment_strokes


# ============================================================
# PARSING + FEATURES (same as before)
# ============================================================

def parse_stroke_file(filepath):
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


def extract_features(strokes):
    all_features = []
    for stroke_idx, stroke in enumerate(strokes):
        for i, (x, y, t) in enumerate(stroke):
            if i == 0:
                dx, dy, speed, angle, curvature = 0, 0, 0, 0, 0
            else:
                prev_x, prev_y, prev_t = stroke[i - 1]
                dx = x - prev_x
                dy = y - prev_y
                dt = t - prev_t
                distance = math.sqrt(dx**2 + dy**2)
                speed = distance / dt if dt > 0 else 0
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
                    curvature = 0
            all_features.append([x, y, dx, dy, speed, angle, curvature, 1.0])
        if stroke_idx < len(strokes) - 1:
            nx, ny, _ = strokes[stroke_idx + 1][0]
            all_features.append([nx, ny, 0, 0, 0, 0, 0, 0.0])
    return np.array(all_features, dtype=np.float32)


def normalize_features(features):
    features = features.copy()
    for col in [0, 1]:
        min_val = features[:, col].min()
        max_val = features[:, col].max()
        rang = max_val - min_val
        if rang > 0:
            features[:, col] = (features[:, col] - min_val) / rang
        else:
            features[:, col] = 0.5
    for col in [2, 3]:
        max_abs = np.abs(features[:, col]).max()
        if max_abs > 0:
            features[:, col] = features[:, col] / max_abs
    speed_col = features[:, 4]
    p95 = np.percentile(speed_col[speed_col > 0], 95) if np.any(speed_col > 0) else 1.0
    features[:, 4] = np.clip(speed_col, 0, p95) / (p95 + 1e-6)
    features[:, 5] = features[:, 5] / math.pi
    features[:, 6] = np.clip(features[:, 6], -math.pi, math.pi) / math.pi
    return features


def pad_or_truncate(features, max_len):
    actual_len = len(features)
    num_features = features.shape[1]
    if actual_len >= max_len:
        return features[:max_len], max_len
    else:
        padded = np.zeros((max_len, num_features), dtype=np.float32)
        padded[:actual_len] = features
        return padded, actual_len


# ============================================================
# DATASET WITH AUGMENTATION
# ============================================================

class StrokeDataset(Dataset):
    def __init__(self, data_dir, labels_file=None, max_len=300,
                 max_classes=None, augment=False, aug_intensity=1.0):
        """
        Args:
            data_dir:      folder with {class_id}-{sample_id}.txt files
            labels_file:   syllable list file
            max_len:       pad/truncate sequences to this length
            max_classes:   limit number of classes
            augment:       True = apply random augmentation (for training)
                          False = no augmentation (for validation/testing)
            aug_intensity: how strong the augmentation is (0.0 to 2.0)
        """
        self.data_dir = data_dir
        self.max_len = max_len
        self.num_features = 8
        self.augment = augment
        self.aug_intensity = aug_intensity

        # Load labels
        self.labels = []
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]

        # Scan files
        self.samples = []
        self._scan_files(max_classes)

        # Store RAW strokes (not features) so we can augment on-the-fly
        self.strokes_list = []
        self.labels_list = []
        self._load_all()

        print(f"Dataset loaded:")
        print(f"  Directory:     {data_dir}")
        print(f"  Total samples: {len(self.strokes_list)}")
        print(f"  Num classes:   {self.num_classes}")
        print(f"  Max seq len:   {max_len}")
        print(f"  Augmentation:  {'ON' if augment else 'OFF'}")

    def _scan_files(self, max_classes):
        all_files = glob.glob(os.path.join(self.data_dir, "*.txt"))
        file_info = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            if "-" not in fname:
                continue
            try:
                name_part = fname.replace(".txt", "")
                class_id_str, sample_id_str = name_part.split("-")
                class_id = int(class_id_str)
                file_info.append((class_id, fpath))
            except (ValueError, IndexError):
                continue

        if not file_info:
            self.num_classes = 0
            return

        file_info.sort(key=lambda x: x[0])
        unique_classes = sorted(set(ci for ci, _ in file_info))
        if max_classes:
            unique_classes = unique_classes[:max_classes]

        self.class_id_to_index = {cid: idx for idx, cid in enumerate(unique_classes)}
        self.index_to_class_id = {idx: cid for cid, idx in self.class_id_to_index.items()}
        self.num_classes = len(unique_classes)

        for class_id, fpath in file_info:
            if class_id in self.class_id_to_index:
                self.samples.append((fpath, self.class_id_to_index[class_id]))

    def _load_all(self):
        """Load raw strokes into memory (NOT features yet)"""
        for filepath, class_index in self.samples:
            try:
                strokes = parse_stroke_file(filepath)
                if not strokes:
                    continue
                self.strokes_list.append(strokes)
                self.labels_list.append(class_index)
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")

    def __len__(self):
        return len(self.strokes_list)

    def __getitem__(self, idx):
        """
        Each time this is called, if augment=True, a DIFFERENT
        random augmentation is applied. So every epoch is different!
        """
        strokes = self.strokes_list[idx]

        # === Apply augmentation (only during training) ===
        if self.augment:
            strokes = augment_strokes(strokes, intensity=self.aug_intensity)

        # === Extract features from (possibly augmented) strokes ===
        features = extract_features(strokes)
        features = normalize_features(features)

        # === Pad or truncate ===
        features, actual_len = pad_or_truncate(features, self.max_len)

        # === Convert to tensors ===
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.labels_list[idx], dtype=torch.long)

        return features, label, actual_len

    def get_syllable(self, class_index):
        class_id = self.index_to_class_id.get(class_index, -1)
        if 0 < class_id <= len(self.labels):
            return self.labels[class_id - 1]
        return f"class_{class_index}"


# ============================================================
# HELPER: Create train/val split
# ============================================================

def create_train_val_split(dataset_dir, labels_file, max_len=300,
                           val_ratio=0.25, max_classes=None, aug_intensity=1.0):
    """
    Split data into training (with augmentation) and validation (without).
    
    With ~4 samples per class:
      val_ratio=0.25 → 1 sample for validation, 2-3 for training
    
    Returns: train_dataset, val_dataset
    """
    # First, load all file paths and group by class
    all_files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    class_files = {}

    for fpath in all_files:
        fname = os.path.basename(fpath)
        if "-" not in fname:
            continue
        try:
            name_part = fname.replace(".txt", "")
            class_id_str, _ = name_part.split("-")
            class_id = int(class_id_str)
            if class_id not in class_files:
                class_files[class_id] = []
            class_files[class_id].append(fpath)
        except (ValueError, IndexError):
            continue

    # Sort classes
    sorted_classes = sorted(class_files.keys())
    if max_classes:
        sorted_classes = sorted_classes[:max_classes]

    # Split each class: last sample → validation, rest → training
    train_dir_tmp = os.path.join(dataset_dir, "..", "train_split")
    val_dir_tmp = os.path.join(dataset_dir, "..", "val_split")
    os.makedirs(train_dir_tmp, exist_ok=True)
    os.makedirs(val_dir_tmp, exist_ok=True)

    import shutil

    for class_id in sorted_classes:
        files = sorted(class_files[class_id])
        random.shuffle(files)

        # Put 1 sample in validation, rest in training
        n_val = max(1, int(len(files) * val_ratio))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for f in train_files:
            shutil.copy2(f, os.path.join(train_dir_tmp, os.path.basename(f)))
        for f in val_files:
            shutil.copy2(f, os.path.join(val_dir_tmp, os.path.basename(f)))

    # Create datasets
    train_dataset = StrokeDataset(
        data_dir=train_dir_tmp,
        labels_file=labels_file,
        max_len=max_len,
        augment=True,               # augmentation ON for training
        aug_intensity=aug_intensity,
    )

    val_dataset = StrokeDataset(
        data_dir=val_dir_tmp,
        labels_file=labels_file,
        max_len=max_len,
        augment=False,              # augmentation OFF for validation
    )

    return train_dataset, val_dataset


# ============================================================
# TEST IT
# ============================================================
if __name__ == "__main__":

    DATA_DIR = "data/strokes"
    LABELS_FILE = "data/labels.txt"

    # Test basic dataset with augmentation
    print("=== Testing augmented dataset ===")
    dataset = StrokeDataset(
        data_dir=DATA_DIR,
        labels_file=LABELS_FILE,
        max_len=300,
        augment=True,
    )

    if len(dataset) > 0:
        # Get same sample twice — should be different due to augmentation!
        f1, l1, len1 = dataset[0]
        f2, l2, len2 = dataset[0]
        diff = (f1 - f2).abs().sum().item()
        print(f"\nSame sample, accessed twice:")
        print(f"  Difference: {diff:.2f} (should be > 0 if augmentation works)")
        print(f"  Label: {l1.item()} = {dataset.get_syllable(l1.item())}")

        # Test DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for features, labels, lengths in loader:
            print(f"\nBatch: features={features.shape}, labels={labels.shape}")
            break

        print("\n✓ Augmented dataset working!")