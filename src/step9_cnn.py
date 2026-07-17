"""
=== CNN Pipeline: Image-based Handwriting Recognition ===

Uses your .png images (128x128 grayscale) instead of stroke sequences.
Includes:
  - ImageStrokeDataset: loads and augments images
  - SimpleCNN: lightweight custom CNN
  - ResNetClassifier: pretrained ResNet18 (transfer learning)

Same naming convention as strokes: {class_id}-{sample_id}.png
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.models as models


# ============================================================
# IMAGE DATASET
# ============================================================

class ImageDataset(Dataset):
    """
    Loads .png handwriting images for CNN training.

    Each image:
      - 128x128 grayscale (your raw format)
      - Resized to img_size x img_size
      - Augmented during training (rotation, shift, noise, etc.)
      - Output tensor shape: (1, img_size, img_size)
    """

    def __init__(self, data_dir, labels_file=None, img_size=64,
                 max_classes=None, augment=False):
        """
        Args:
            data_dir:    folder with {class_id}-{sample_id}.png files
            labels_file: syllable list file
            img_size:    resize images to this size (64 or 128)
            max_classes: limit number of classes
            augment:     True = apply augmentation (training)
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.augment = augment

        # Load labels
        self.labels = []
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f if line.strip()]

        # Build transforms
        self.train_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomAffine(
                degrees=15,              # random rotation ±15°
                translate=(0.1, 0.1),    # random shift ±10%
                scale=(0.85, 1.15),      # random zoom 85%-115%
                fill=255,                # fill with white (background)
            ),
            T.RandomPerspective(
                distortion_scale=0.2,    # slight perspective warp
                p=0.3,
                fill=255,
            ),
            T.ToTensor(),                # converts to [0, 1] range
            T.RandomErasing(             # randomly erase small patches
                p=0.1,
                scale=(0.02, 0.08),
                value=1.0,               # erase with white
            ),
            # Invert: make strokes=1 (white) on background=0 (black)
            # This is standard for handwriting recognition
            T.Lambda(lambda x: 1.0 - x),
            T.Normalize(mean=[0.5], std=[0.5]),  # normalize to [-1, 1]
        ])

        self.val_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Lambda(lambda x: 1.0 - x),  # invert
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        # Scan files
        self.samples = []  # list of (filepath, class_index)
        self._scan_files(max_classes)

        print(f"ImageDataset loaded:")
        print(f"  Directory:     {data_dir}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Num classes:   {self.num_classes}")
        print(f"  Image size:    {img_size}x{img_size}")
        print(f"  Augmentation:  {'ON' if augment else 'OFF'}")

    def _scan_files(self, max_classes):
        all_files = glob.glob(os.path.join(self.data_dir, "*.png"))
        file_info = []

        for fpath in all_files:
            fname = os.path.basename(fpath)
            if "-" not in fname:
                continue
            try:
                name_part = fname.replace(".png", "")
                class_id_str, sample_id_str = name_part.split("-")
                class_id = int(class_id_str)
                file_info.append((class_id, fpath))
            except (ValueError, IndexError):
                continue

        if not file_info:
            print(f"  WARNING: No .png files found in {self.data_dir}")
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, class_index = self.samples[idx]

        # Load image as grayscale
        image = Image.open(filepath).convert('L')

        # Apply transforms
        if self.augment:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)

        label = torch.tensor(class_index, dtype=torch.long)
        return image, label

    def get_syllable(self, class_index):
        class_id = self.index_to_class_id.get(class_index, -1)
        if 0 < class_id <= len(self.labels):
            return self.labels[class_id - 1]
        return f"class_{class_index}"


# ============================================================
# MODEL 1: Simple CNN (lightweight, custom)
# ============================================================

class SimpleCNN(nn.Module):
    """
    Custom CNN designed for small grayscale handwriting images.

    Architecture:
      Input (1, 64, 64)
        → Conv block 1: 1→32 channels, pool → (32, 32, 32)
        → Conv block 2: 32→64 channels, pool → (64, 16, 16)
        → Conv block 3: 64→128 channels, pool → (128, 8, 8)
        → Conv block 4: 128→256 channels, pool → (256, 4, 4)
        → Global Average Pooling → (256,)
        → FC → 1000 classes

    Parameters: ~500K (similar to LSTM)
    """

    def __init__(self, num_classes=1000, dropout=0.3):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout * 0.5),
            )

        self.features = nn.Sequential(
            conv_block(1, 32),     # (1,64,64) → (32,32,32)
            conv_block(32, 64),    # → (64,16,16)
            conv_block(64, 128),   # → (128,8,8)
            conv_block(128, 256),  # → (256,4,4)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # → (256,1,1)
            nn.Flatten(),              # → (256,)
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, lengths=None):
        # lengths is ignored (just for compatibility with training loop)
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# MODEL 2: ResNet18 Transfer Learning
# ============================================================

class ResNetClassifier(nn.Module):
    """
    Pretrained ResNet18 adapted for grayscale handwriting.

    Changes from standard ResNet18:
      1. First conv: 3 channels → 1 channel (grayscale)
      2. Last FC: 1000 ImageNet classes → your 1000 syllables

    The pretrained weights give us edge/curve/shape detectors for free.
    We fine-tune the whole network on your data.

    Parameters: ~11M (much larger, but pretrained knowledge helps)
    """

    def __init__(self, num_classes=1000, dropout=0.3, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        # Modify first conv layer: 3 channels → 1 channel
        # Keep the pretrained weights by averaging across the 3 channels
        original_conv = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Average the 3-channel weights into 1 channel
            with torch.no_grad():
                self.resnet.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Replace final FC layer
        in_features = self.resnet.fc.in_features  # 512
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x, lengths=None):
        return self.resnet(x)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    NUM_CLASSES = 1000
    IMG_SIZE = 64

    # Dummy input: batch of 4 grayscale images
    dummy = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)

    print("=" * 60)
    print("  CNN MODEL COMPARISON")
    print("=" * 60)

    # Simple CNN
    print("\n--- SimpleCNN ---")
    cnn = SimpleCNN(num_classes=NUM_CLASSES, dropout=0.3)
    out = cnn(dummy)
    total, trainable = count_parameters(cnn)
    print(f"  Output shape:     {out.shape}")
    print(f"  Parameters:       {trainable:,}")

    # ResNet18
    print("\n--- ResNet18 (transfer learning) ---")
    resnet = ResNetClassifier(num_classes=NUM_CLASSES, dropout=0.3, pretrained=True)
    out = resnet(dummy)
    total, trainable = count_parameters(resnet)
    print(f"  Output shape:     {out.shape}")
    print(f"  Parameters:       {trainable:,}")

    print(f"\n✓ Both CNN models working!")