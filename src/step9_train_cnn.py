"""
=== CNN Training Script ===

Trains SimpleCNN and/or ResNet18 on your .png handwriting images.
Uses the same training infrastructure as step8_train.py.

Usage:
  python step9_train_cnn.py

Config is at the bottom — adjust paths and hyperparameters there.
"""

import os
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from step9_cnn import ImageDataset, SimpleCNN, ResNetClassifier


# ============================================================
# TRAINER (same as step8 but adapted for images)
# ============================================================

class CNNTrainer:
    def __init__(self, model, model_name, device,
                 lr=0.001, label_smoothing=0.1, save_dir="checkpoints"):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_top5': [], 'val_top5': [],
            'lr': [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.epochs_no_improve = 0

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            k = min(5, outputs.size(1))
            _, top5_pred = outputs.topk(k, dim=1)
            correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()

        return total_loss / total, correct / total, correct_top5 / total

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            k = min(5, outputs.size(1))
            _, top5_pred = outputs.topk(k, dim=1)
            correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()

        return total_loss / total, correct / total, correct_top5 / total

    def train(self, train_loader, val_loader, num_epochs=100, patience=25):
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n{'='*60}")
        print(f"  Training: {self.model_name.upper()}")
        print(f"  Parameters: {total_params:,}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc, train_top5 = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_top5 = self.validate(val_loader)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_top5'].append(train_top5)
            self.history['val_top5'].append(val_top5)
            self.history['lr'].append(current_lr)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                save_path = os.path.join(self.save_dir, f"{self.model_name}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_top5': val_top5,
                }, save_path)
            else:
                self.epochs_no_improve += 1

            epoch_time = time.time() - epoch_start
            print(f"  Epoch {epoch:3d}/{num_epochs} ({epoch_time:.1f}s) | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.3f} top5={train_top5:.3f} | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.3f} top5={val_top5:.3f} | "
                  f"LR={current_lr:.6f}"
                  f"{' ★' if self.epochs_no_improve == 0 else ''}")

            if self.epochs_no_improve >= patience:
                print(f"\n  Early stopping! No improvement for {patience} epochs.")
                break

        elapsed = time.time() - start_time
        print(f"\n  Training complete in {elapsed/60:.1f} minutes")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f} (epoch {self.best_epoch})")

        history_path = os.path.join(self.save_dir, f"{self.model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

        return self.history


# ============================================================
# PLOT ALL RESULTS (stroke models + CNN models)
# ============================================================

def plot_all_results(save_dir="checkpoints"):
    """Load all history files and plot comparison"""
    model_names = ['lstm', 'liquid', 'simple_cnn', 'resnet18']
    colors = {'lstm': '#FF4444', 'liquid': '#4444FF',
              'simple_cnn': '#44AA44', 'resnet18': '#FF8800'}
    histories = {}

    for name in model_names:
        path = os.path.join(save_dir, f"{name}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[name] = json.load(f)

    if not histories:
        print("No history files found!")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('All Models Comparison', fontsize=14, fontweight='bold')

    for name, history in histories.items():
        c = colors.get(name, '#888888')
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0].plot(epochs, history['val_loss'], '-', color=c, label=name, linewidth=2)
        axes[1].plot(epochs, history['val_acc'], '-', color=c, label=name, linewidth=2)
        axes[2].plot(epochs, history['val_top5'], '-', color=c, label=name, linewidth=2)

    axes[0].set_title('Validation Loss'); axes[0].set_xlabel('Epoch')
    axes[1].set_title('Validation Accuracy'); axes[1].set_xlabel('Epoch')
    axes[2].set_title('Validation Top-5'); axes[2].set_xlabel('Epoch')

    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "all_models_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison plot: {save_path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    # ==========================
    # CONFIG
    # ==========================
    IMAGE_DIR     = "../data/images"     # your .png files
    LABELS_FILE   = "../data/syl.txt"
    SAVE_DIR      = "../checkpoints"
    IMG_SIZE      = 64              # resize images to 64x64
    BATCH_SIZE    = 64              # images are lighter than sequences
    NUM_EPOCHS    = 100
    LEARNING_RATE = 0.001
    PATIENCE      = 25
    OVERSAMPLE    = 5               # repeat training data 5x
    VAL_RATIO     = 0.25
    SEED          = 42

    # Which CNN models to train
    TRAIN_SIMPLE_CNN = True
    TRAIN_RESNET18   = True

    # ==========================
    # SETUP
    # ==========================
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================
    # LOAD DATA
    # ==========================
    print("\nLoading image dataset...")

    # Load without augmentation for splitting
    full_dataset = ImageDataset(
        data_dir=IMAGE_DIR,
        labels_file=LABELS_FILE,
        img_size=IMG_SIZE,
        augment=False,
    )

    num_classes = full_dataset.num_classes
    total_samples = len(full_dataset)

    if total_samples == 0:
        print("ERROR: No images found!")
        return

    # Split
    val_size = max(1, int(total_samples * VAL_RATIO))
    train_size = total_samples - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Create augmented training set
    train_aug = AugmentedImageSubset(full_dataset, train_subset.indices, IMG_SIZE)

    if OVERSAMPLE > 1:
        train_aug = ConcatDataset([train_aug] * OVERSAMPLE)

    train_loader = DataLoader(train_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Training samples: {train_size} (x{OVERSAMPLE} = {len(train_aug)})")
    print(f"  Validation samples: {val_size}")
    print(f"  Classes: {num_classes}")

    # ==========================
    # TRAIN
    # ==========================
    histories = {}

    if TRAIN_SIMPLE_CNN:
        model = SimpleCNN(num_classes=num_classes, dropout=0.3)
        trainer = CNNTrainer(model, "simple_cnn", device, lr=LEARNING_RATE, save_dir=SAVE_DIR)
        histories['simple_cnn'] = trainer.train(train_loader, val_loader, NUM_EPOCHS, PATIENCE)

    if TRAIN_RESNET18:
        model = ResNetClassifier(num_classes=num_classes, dropout=0.3, pretrained=True)
        trainer = CNNTrainer(model, "resnet18", device, lr=LEARNING_RATE * 0.1, save_dir=SAVE_DIR)
        # Lower LR for pretrained model (fine-tuning)
        histories['resnet18'] = trainer.train(train_loader, val_loader, NUM_EPOCHS, PATIENCE)

    # ==========================
    # PLOT ALL RESULTS
    # ==========================
    plot_all_results(SAVE_DIR)

    # Print final comparison
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for name in ['lstm', 'liquid', 'simple_cnn', 'resnet18']:
        cp_path = os.path.join(SAVE_DIR, f"{name}_best.pt")
        if os.path.exists(cp_path):
            cp = torch.load(cp_path, map_location='cpu', weights_only=False)
            print(f"  {name:12s}: acc={cp['val_acc']:.4f}  top5={cp['val_top5']:.4f}  epoch={cp['epoch']}")

    print(f"\nDone!")


# ============================================================
# HELPER: Augmented image subset
# ============================================================

class AugmentedImageSubset(torch.utils.data.Dataset):
    """Wraps a subset with augmentation enabled"""

    def __init__(self, full_dataset, indices, img_size=64):
        self.full_dataset = full_dataset
        self.indices = indices
        self.img_size = img_size

        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.15), fill=255),
            T.RandomPerspective(distortion_scale=0.2, p=0.3, fill=255),
            T.ToTensor(),
            T.RandomErasing(p=0.1, scale=(0.02, 0.08), value=1.0),
            T.Lambda(lambda x: 1.0 - x),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        filepath, class_index = self.full_dataset.samples[self.indices[idx]]
        from PIL import Image
        image = Image.open(filepath).convert('L')
        image = self.transform(image)
        label = torch.tensor(class_index, dtype=torch.long)
        return image, label


# ============================================================
if __name__ == "__main__":
    main()