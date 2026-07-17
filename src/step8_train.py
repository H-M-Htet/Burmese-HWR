"""
=== Step 8: Training Loop ===

Trains LSTM and Liquid models, tracks metrics, saves best checkpoints.
Handles the unique challenges of small datasets (augmentation, oversampling).

Usage:
  python step8_train.py

Config is at the bottom of the file — adjust paths and hyperparameters there.
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
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

from step7_models import LSTMClassifier, LiquidClassifier, count_parameters
from step5b_dataset_augmented import StrokeDataset


# ============================================================
# TRAINING ENGINE
# ============================================================

class Trainer:
    def __init__(self, model, model_name, num_classes, device,
                 lr=0.001, label_smoothing=0.1, save_dir="checkpoints"):
        """
        Args:
            model:           the PyTorch model (LSTM or Liquid)
            model_name:      "lstm" or "liquid" (for saving)
            num_classes:     number of output classes
            device:          "cuda" or "cpu"
            lr:              learning rate
            label_smoothing: softens targets (helps with small datasets)
            save_dir:        where to save model checkpoints
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Loss function with label smoothing
        # Instead of hard targets [0, 0, 1, 0, ...],
        # it uses soft targets [0.0001, 0.0001, 0.9, 0.0001, ...]
        # This prevents overconfident predictions
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-4,  # L2 regularization
        )

        # Learning rate scheduler: reduce LR when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )

        # Track history
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
        """Train for one epoch, return average loss and accuracy"""
        self.model.train()
        total_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0

        for features, labels, lengths in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            lengths = torch.tensor(lengths) if not isinstance(lengths, torch.Tensor) else lengths

            # Forward pass
            outputs = self.model(features, lengths)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * features.size(0)
            total += features.size(0)

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            k = min(5, outputs.size(1))
            _, top5_pred = outputs.topk(k, dim=1)
            correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        top5_acc = correct_top5 / total
        return avg_loss, accuracy, top5_acc

    @torch.no_grad()
    def validate(self, dataloader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0

        for features, labels, lengths in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            lengths = torch.tensor(lengths) if not isinstance(lengths, torch.Tensor) else lengths

            outputs = self.model(features, lengths)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            total += features.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()

            k = min(5, outputs.size(1))
            _, top5_pred = outputs.topk(k, dim=1)
            correct_top5 += top5_pred.eq(labels.unsqueeze(1)).any(1).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        top5_acc = correct_top5 / total
        return avg_loss, accuracy, top5_acc

    def train(self, train_loader, val_loader, num_epochs=100, patience=25):
        """
        Full training loop with early stopping.

        Args:
            train_loader: DataLoader for training data
            val_loader:   DataLoader for validation data
            num_epochs:   maximum epochs to train
            patience:     stop if no improvement for this many epochs
        """
        print(f"\n{'='*60}")
        print(f"  Training: {self.model_name.upper()}")
        total, trainable = count_parameters(self.model)
        print(f"  Parameters: {trainable:,}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc, train_top5 = self.train_one_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_top5 = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_top5'].append(train_top5)
            self.history['val_top5'].append(val_top5)
            self.history['lr'].append(current_lr)

            # Check for improvement
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                # Save best model
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

            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"  Epoch {epoch:3d}/{num_epochs} ({epoch_time:.1f}s) | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.3f} top5={train_top5:.3f} | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.3f} top5={val_top5:.3f} | "
                  f"LR={current_lr:.6f}"
                  f"{' ★' if self.epochs_no_improve == 0 else ''}")

            # Early stopping
            if self.epochs_no_improve >= patience:
                print(f"\n  Early stopping! No improvement for {patience} epochs.")
                break

        elapsed = time.time() - start_time
        print(f"\n  Training complete in {elapsed/60:.1f} minutes")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f} (epoch {self.best_epoch})")

        # Save history
        history_path = os.path.join(self.save_dir, f"{self.model_name}_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

        return self.history


# ============================================================
# PLOT RESULTS
# ============================================================

def plot_training_history(histories, save_path="training_results.png"):
    """
    Plot training curves for all models side by side.

    Args:
        histories: dict of {model_name: history_dict}
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'lstm': '#FF4444', 'liquid': '#4444FF'}

    for name, history in histories.items():
        c = colors.get(name, '#44AA44')
        epochs = range(1, len(history['train_loss']) + 1)

        # Loss
        axes[0].plot(epochs, history['train_loss'], '--', color=c, alpha=0.5, label=f'{name} train')
        axes[0].plot(epochs, history['val_loss'], '-', color=c, label=f'{name} val')

        # Accuracy
        axes[1].plot(epochs, history['train_acc'], '--', color=c, alpha=0.5, label=f'{name} train')
        axes[1].plot(epochs, history['val_acc'], '-', color=c, label=f'{name} val')

        # Top-5 Accuracy
        axes[2].plot(epochs, history['train_top5'], '--', color=c, alpha=0.5, label=f'{name} train')
        axes[2].plot(epochs, history['val_top5'], '-', color=c, label=f'{name} val')

    axes[0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Top-1 Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title('Top-5 Accuracy', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved training plot to: {save_path}")
    plt.close()


# ============================================================
# MAIN: TRAIN BOTH MODELS
# ============================================================

def main():
    # ==========================
    # CONFIG — change these!
    # ==========================
    DATA_DIR      = "data/strokes"
    LABELS_FILE   = "data/labels.txt"
    SAVE_DIR      = "checkpoints"
    MAX_LEN       = 300          # pad/truncate length (adjust based on your data)
    BATCH_SIZE    = 32
    NUM_EPOCHS    = 100
    LEARNING_RATE = 0.001
    PATIENCE      = 25           # early stopping patience
    OVERSAMPLE    = 5            # repeat training data 5x per epoch
    VAL_RATIO     = 0.25         # 25% of data for validation
    SEED          = 42

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
    print("\nLoading dataset...")

    # Load full dataset (no augmentation) to split
    full_dataset = StrokeDataset(
        data_dir=DATA_DIR,
        labels_file=LABELS_FILE,
        max_len=MAX_LEN,
        augment=False,
    )

    num_classes = full_dataset.num_classes
    total_samples = len(full_dataset)

    if total_samples == 0:
        print("ERROR: No samples found!")
        return

    # Split into train/val
    val_size = max(1, int(total_samples * VAL_RATIO))
    train_size = total_samples - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    # Create augmented training dataset
    # We wrap the train indices to use augmentation
    train_aug_dataset = AugmentedSubset(full_dataset, train_dataset.indices)

    # Oversample: repeat training data for more variety per epoch
    if OVERSAMPLE > 1:
        train_aug_dataset = ConcatDataset([train_aug_dataset] * OVERSAMPLE)
        print(f"  Oversampled {OVERSAMPLE}x: {len(train_aug_dataset)} training samples per epoch")

    train_loader = DataLoader(
        train_aug_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0,
    )

    print(f"  Training samples: {train_size} (x{OVERSAMPLE} = {len(train_aug_dataset)})")
    print(f"  Validation samples: {val_size}")
    print(f"  Number of classes: {num_classes}")

    # ==========================
    # TRAIN MODELS
    # ==========================
    histories = {}

    # --- Train LSTM ---
    lstm_model = LSTMClassifier(
        input_size=8,
        hidden_size=128,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3,
    )
    lstm_trainer = Trainer(
        model=lstm_model,
        model_name="lstm",
        num_classes=num_classes,
        device=device,
        lr=LEARNING_RATE,
        save_dir=SAVE_DIR,
    )
    histories['lstm'] = lstm_trainer.train(
        train_loader, val_loader,
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
    )

    # --- Train Liquid ---
    liquid_model = LiquidClassifier(
        input_size=8,
        ltc_units=128,
        num_classes=num_classes,
        dropout=0.3,
    )
    liquid_trainer = Trainer(
        model=liquid_model,
        model_name="liquid",
        num_classes=num_classes,
        device=device,
        lr=LEARNING_RATE,
        save_dir=SAVE_DIR,
    )
    histories['liquid'] = liquid_trainer.train(
        train_loader, val_loader,
        num_epochs=NUM_EPOCHS, patience=PATIENCE,
    )

    # ==========================
    # COMPARE RESULTS
    # ==========================
    print(f"\n{'='*60}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*60}")
    for name, trainer_obj in [('lstm', lstm_trainer), ('liquid', liquid_trainer)]:
        total, trainable = count_parameters(trainer_obj.model)
        print(f"\n  {name.upper()}:")
        print(f"    Parameters:     {trainable:,}")
        print(f"    Best val acc:   {trainer_obj.best_val_acc:.4f}")
        print(f"    Best epoch:     {trainer_obj.best_epoch}")

    # Plot comparison
    plot_training_history(histories, save_path=os.path.join(SAVE_DIR, "training_results.png"))

    print(f"\nDone! Check '{SAVE_DIR}/' for model checkpoints and plots.")


# ============================================================
# HELPER: Augmented subset wrapper
# ============================================================

class AugmentedSubset(torch.utils.data.Dataset):
    """
    Wraps a subset of StrokeDataset with augmentation enabled.
    This lets us use augmentation on training data only.
    """
    def __init__(self, full_dataset, indices):
        self.full_dataset = full_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        strokes = self.full_dataset.strokes_list[real_idx]

        # Apply augmentation
        from step6_augmentation import augment_strokes
        strokes = augment_strokes(strokes, intensity=1.0)

        # Extract features
        from step5b_dataset_augmented import extract_features, normalize_features, pad_or_truncate
        features = extract_features(strokes)
        features = normalize_features(features)
        features, actual_len = pad_or_truncate(features, self.full_dataset.max_len)

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(self.full_dataset.labels_list[real_idx], dtype=torch.long)
        return features, label, actual_len


# ============================================================
if __name__ == "__main__":
    main()
