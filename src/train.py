"""
=== Unified Training Script ===

One script to train all models. Config-driven, no code editing needed.

Usage:
  python train.py                           # train all enabled models
  python train.py --config config.yaml      # custom config
  python train.py --models lstm simple_cnn  # train specific models only
  python train.py --epochs 50              # override epochs
  python train.py --dry-run                # just show config, don't train
"""

import os
import sys
import time
import json
import random
import argparse
import math
import numpy as np
import yaml
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# DATA: STROKE PARSING + FEATURES
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
                if current_stroke:
                    strokes.append(current_stroke)
                current_stroke = []
            else:
                parts = line.split()
                current_stroke.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if current_stroke:
        strokes.append(current_stroke)
    return strokes


def extract_features(strokes):
    all_features = []
    for si, stroke in enumerate(strokes):
        for i, (x, y, t) in enumerate(stroke):
            if i == 0:
                dx, dy, speed, angle, curvature = 0, 0, 0, 0, 0
            else:
                px, py, pt = stroke[i-1]
                dx, dy, dt = x-px, y-py, t-pt
                speed = math.sqrt(dx**2+dy**2) / dt if dt > 0 else 0
                angle = math.atan2(dy, dx)
                if i >= 2:
                    p2x, p2y, _ = stroke[i-2]
                    pa = math.atan2(py-p2y, px-p2x)
                    curvature = angle - pa
                    while curvature > math.pi: curvature -= 2*math.pi
                    while curvature < -math.pi: curvature += 2*math.pi
                else:
                    curvature = 0
            all_features.append([x, y, dx, dy, speed, angle, curvature, 1.0])
        if si < len(strokes)-1:
            nx, ny, _ = strokes[si+1][0]
            all_features.append([nx, ny, 0, 0, 0, 0, 0, 0.0])
    return np.array(all_features, dtype=np.float32)


def normalize_features(features):
    f = features.copy()
    for c in [0, 1]:
        mn, mx = f[:, c].min(), f[:, c].max()
        r = mx - mn
        f[:, c] = (f[:, c] - mn) / r if r > 0 else 0.5
    for c in [2, 3]:
        ma = np.abs(f[:, c]).max()
        if ma > 0: f[:, c] /= ma
    sc = f[:, 4]
    p95 = np.percentile(sc[sc > 0], 95) if np.any(sc > 0) else 1.0
    f[:, 4] = np.clip(sc, 0, p95) / (p95 + 1e-6)
    f[:, 5] /= math.pi
    f[:, 6] = np.clip(f[:, 6], -math.pi, math.pi) / math.pi
    return f


def pad_or_truncate(features, max_len):
    n = len(features)
    if n >= max_len:
        return features[:max_len], max_len
    padded = np.zeros((max_len, features.shape[1]), dtype=np.float32)
    padded[:n] = features
    return padded, n


# ============================================================
# DATA: STROKE AUGMENTATION
# ============================================================

def augment_strokes(strokes, cfg):
    intensity = cfg.get('intensity', 1.0)
    s = [[(x, y, t) for x, y, t in st] for st in strokes]

    if random.random() < 0.8:
        sigma = cfg.get('jitter_sigma', 2.0) * intensity
        s = [[(x+random.gauss(0, sigma), y+random.gauss(0, sigma), t) for x, y, t in st] for st in s]

    if random.random() < 0.5:
        ax = [x for st in s for x, y, t in st]
        ay = [y for st in s for x, y, t in st]
        cx, cy = sum(ax)/len(ax), sum(ay)/len(ay)
        sig = cfg.get('scale_sigma', 0.15) * intensity
        sx, sy = 1+random.gauss(0, sig), 1+random.gauss(0, sig)
        s = [[(cx+(x-cx)*sx, cy+(y-cy)*sy, t) for x, y, t in st] for st in s]

    if random.random() < 0.5:
        ax = [x for st in s for x, y, t in st]
        ay = [y for st in s for x, y, t in st]
        cx, cy = sum(ax)/len(ax), sum(ay)/len(ay)
        ang = random.uniform(-1, 1) * cfg.get('rotation_max', 15) * intensity
        ar = math.radians(ang)
        ca, sa = math.cos(ar), math.sin(ar)
        s = [[(cx+(x-cx)*ca-(y-cy)*sa, cy+(x-cx)*sa+(y-cy)*ca, t) for x, y, t in st] for st in s]

    if random.random() < 0.4:
        sig = cfg.get('translate_sigma', 10) * intensity
        dx, dy = random.gauss(0, sig), random.gauss(0, sig)
        s = [[(x+dx, y+dy, t) for x, y, t in st] for st in s]

    if random.random() < 0.3:
        wf = max(0.5, min(2.0, 1+random.gauss(0, cfg.get('speed_warp_sigma', 0.3)*intensity)))
        ns = []
        for st in s:
            if not st: ns.append(st); continue
            t0 = st[0][2]
            ns.append([(x, y, t0+(t-t0)*wf) for x, y, t in st])
        s = ns

    if random.random() < 0.2:
        dr = cfg.get('point_dropout_rate', 0.1) * intensity
        ns = []
        for st in s:
            if len(st) <= 3: ns.append(st); continue
            ns.append([st[0]] + [p for p in st[1:-1] if random.random() > dr] + [st[-1]])
        s = ns

    return s


# ============================================================
# DATASETS
# ============================================================

import glob

class StrokeDataset(Dataset):
    def __init__(self, data_dir, labels_file=None, max_len=300, augment=False, aug_cfg=None):
        self.max_len = max_len
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        self.labels = []
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                self.labels = [l.strip() for l in f if l.strip()]

        self.strokes_list = []
        self.labels_list = []
        file_info = []
        for fp in glob.glob(os.path.join(data_dir, "*.txt")):
            fn = os.path.basename(fp)
            if "-" not in fn: continue
            try:
                cid = int(fn.replace(".txt", "").split("-")[0])
                file_info.append((cid, fp))
            except: continue

        if not file_info:
            self.num_classes = 0; return
        file_info.sort(key=lambda x: x[0])
        uq = sorted(set(c for c, _ in file_info))
        self.c2i = {c: i for i, c in enumerate(uq)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.num_classes = len(uq)

        for cid, fp in file_info:
            if cid in self.c2i:
                try:
                    st = parse_stroke_file(fp)
                    if st:
                        self.strokes_list.append(st)
                        self.labels_list.append(self.c2i[cid])
                except: pass

    def __len__(self): return len(self.strokes_list)

    def __getitem__(self, idx):
        strokes = self.strokes_list[idx]
        if self.augment:
            strokes = augment_strokes(strokes, self.aug_cfg)
        features = normalize_features(extract_features(strokes))
        features, length = pad_or_truncate(features, self.max_len)
        return torch.tensor(features, dtype=torch.float32), \
               torch.tensor(self.labels_list[idx], dtype=torch.long), length

    def get_syllable(self, ci):
        cid = self.i2c.get(ci, -1)
        return self.labels[cid-1] if 0 < cid <= len(self.labels) else f"class_{ci}"


class ImageDataset(Dataset):
    def __init__(self, data_dir, labels_file=None, img_size=64, augment=False, aug_cfg=None):
        self.img_size = img_size
        self.augment = augment
        ac = aug_cfg or {}

        self.train_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomAffine(degrees=ac.get('img_rotation', 15),
                           translate=(ac.get('img_translate', 0.1),)*2,
                           scale=(ac.get('img_scale_min', 0.85), ac.get('img_scale_max', 1.15)),
                           fill=255),
            T.RandomPerspective(distortion_scale=ac.get('img_perspective', 0.2), p=0.3, fill=255),
            T.ToTensor(),
            T.RandomErasing(p=0.1, scale=(0.02, 0.08), value=1.0),
            T.Lambda(lambda x: 1.0-x),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.val_tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Lambda(lambda x: 1.0-x),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.labels = []
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, "r", encoding="utf-8") as f:
                self.labels = [l.strip() for l in f if l.strip()]

        self.samples = []
        file_info = []
        for fp in glob.glob(os.path.join(data_dir, "*.png")):
            fn = os.path.basename(fp)
            if "-" not in fn: continue
            try:
                cid = int(fn.replace(".png", "").split("-")[0])
                file_info.append((cid, fp))
            except: continue

        if not file_info:
            self.num_classes = 0; return
        file_info.sort(key=lambda x: x[0])
        uq = sorted(set(c for c, _ in file_info))
        self.c2i = {c: i for i, c in enumerate(uq)}
        self.i2c = {i: c for c, i in self.c2i.items()}
        self.num_classes = len(uq)
        self.samples = [(fp, self.c2i[c]) for c, fp in file_info if c in self.c2i]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fp, ci = self.samples[idx]
        img = Image.open(fp).convert('L')
        img = self.train_tf(img) if self.augment else self.val_tf(img)
        return img, torch.tensor(ci, dtype=torch.long)


# ============================================================
# MODELS
# ============================================================

from ncps.torch import LTC
from ncps.wirings import AutoNCP
import torchvision.models as models


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(8, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(hidden_size*2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_size*2, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        if lengths is not None:
            lengths = torch.clamp(torch.tensor(lengths) if not isinstance(lengths, torch.Tensor) else lengths, min=1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(x)
        aw = self.attention(out).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1).to(x.device)
            aw = aw.masked_fill(mask, float('-inf'))
        aw = torch.softmax(aw, dim=1).unsqueeze(-1)
        ctx = (out * aw).sum(dim=1)
        return self.classifier(ctx)


class LiquidClassifier(nn.Module):
    def __init__(self, num_classes, ltc_units=128, dropout=0.3):
        super().__init__()
        mn = min(64, ltc_units // 2)
        self.proj = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.ltc = LTC(32, AutoNCP(ltc_units, mn), batch_first=True)
        self.attention = nn.Sequential(nn.Linear(mn, 32), nn.Tanh(), nn.Linear(32, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(mn, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))
        self.mn = mn

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        out, _ = self.ltc(self.proj(x))
        aw = self.attention(out).squeeze(-1)
        if lengths is not None:
            lengths_t = torch.tensor(lengths) if not isinstance(lengths, torch.Tensor) else lengths
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths_t.unsqueeze(1).to(x.device)
            aw = aw.masked_fill(mask, float('-inf'))
        aw = torch.softmax(aw, dim=1).unsqueeze(-1)
        return self.classifier((out * aw).sum(dim=1))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
                nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(True),
                nn.MaxPool2d(2), nn.Dropout2d(dropout*0.5))
        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(256, 512), nn.ReLU(True),
            nn.Dropout(dropout), nn.Linear(512, num_classes))

    def forward(self, x, lengths=None):
        return self.classifier(self.features(x))


class ResNetClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3, pretrained=True, freeze_layers=False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        orig = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.resnet.conv1.weight = nn.Parameter(orig.weight.mean(dim=1, keepdim=True))
        self.resnet.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

        if freeze_layers:
            for p in self.resnet.parameters(): p.requires_grad = False
            for p in self.resnet.layer3.parameters(): p.requires_grad = True
            for p in self.resnet.layer4.parameters(): p.requires_grad = True
            for p in self.resnet.fc.parameters(): p.requires_grad = True

    def forward(self, x, lengths=None):
        return self.resnet(x)


class TransformerClassifier(nn.Module):
    """
    Small Transformer encoder for stroke sequence classification.

    How it works:
      1. Linear projection: 8 features → d_model dimensions
      2. Positional encoding: tells Transformer the order of points
      3. Transformer encoder: self-attention captures relationships
         between ALL points simultaneously (unlike LSTM which is sequential)
      4. CLS token pooling: a learnable token aggregates the full sequence
      5. FC classifier → 1000 classes

    Key difference from LSTM:
      - LSTM processes left→right (or bidirectional)
      - Transformer sees ALL points at once via self-attention
      - Faster training (parallelizable) but needs positional encoding
    """
    def __init__(self, num_classes, d_model=128, nhead=4, num_layers=3, dropout=0.3, max_len=400):
        super().__init__()

        # Project 8 input features to d_model dimensions
        self.input_proj = nn.Sequential(
            nn.Linear(8, d_model),
            nn.LayerNorm(d_model),
        )

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)

        # Learnable CLS token (aggregates the whole sequence)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths=None):
        B, T, _ = x.shape

        # Project input
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :T+1, :]

        # Create padding mask (True = ignore)
        if lengths is not None:
            lengths_t = torch.tensor(lengths) if not isinstance(lengths, torch.Tensor) else lengths
            # +1 for CLS token (never masked)
            mask = torch.zeros(B, T+1, dtype=torch.bool, device=x.device)
            for i in range(B):
                mask[i, lengths_t[i]+1:] = True  # mask after actual length + CLS
        else:
            mask = None

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=mask)

        # Take CLS token output (first position)
        cls_out = x[:, 0, :]  # (B, d_model)

        return self.classifier(cls_out)


# ============================================================
# TRAINER
# ============================================================

class Trainer:
    def __init__(self, model, name, device, lr, grad_clip=0.5, save_dir="checkpoints"):
        self.model = model.to(device)
        self.name = name
        self.device = device
        self.grad_clip = grad_clip
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10)

        self.history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'train_top5':[], 'val_top5':[], 'lr':[]}
        self.best_val_acc = 0
        self.best_epoch = 0
        self.no_improve = 0

    def _run_epoch(self, loader, train=True):
        self.model.train() if train else self.model.eval()
        total_loss = correct = correct5 = total = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                if len(batch) == 3:
                    x, y, lengths = batch
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x, lengths)
                else:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x)

                loss = self.criterion(out, y)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                total_loss += loss.item() * x.size(0)
                total += x.size(0)
                correct += out.argmax(1).eq(y).sum().item()
                k = min(5, out.size(1))
                correct5 += out.topk(k, 1)[1].eq(y.unsqueeze(1)).any(1).sum().item()

        return total_loss/total, correct/total, correct5/total

    def train(self, train_loader, val_loader, epochs=100, patience=25):
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())

        print(f"\n{'='*70}")
        print(f"  {self.name.upper()} | {trainable:,} trainable / {total:,} total params | {self.device}")
        print(f"{'='*70}")

        for epoch in range(1, epochs+1):
            t0 = time.time()
            tl, ta, t5 = self._run_epoch(train_loader, train=True)
            vl, va, v5 = self._run_epoch(val_loader, train=False)
            self.scheduler.step(vl)
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(tl)
            self.history['val_loss'].append(vl)
            self.history['train_acc'].append(ta)
            self.history['val_acc'].append(va)
            self.history['train_top5'].append(t5)
            self.history['val_top5'].append(v5)
            self.history['lr'].append(lr)

            star = ''
            if va > self.best_val_acc:
                self.best_val_acc = va
                self.best_epoch = epoch
                self.no_improve = 0
                star = ' ★'
                torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'val_acc': va, 'val_top5': v5},
                           os.path.join(self.save_dir, f"{self.name}_best.pt"))
            else:
                self.no_improve += 1

            dt = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs} ({dt:.1f}s) | "
                  f"Train: loss={tl:.4f} acc={ta:.3f} top5={t5:.3f} | "
                  f"Val: loss={vl:.4f} acc={va:.3f} top5={v5:.3f} | "
                  f"LR={lr:.6f}{star}")

            if self.no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

        with open(os.path.join(self.save_dir, f"{self.name}_history.json"), 'w') as f:
            json.dump(self.history, f)

        print(f"  Best: {self.best_val_acc:.4f} (epoch {self.best_epoch})")
        return self.history


# ============================================================
# AUGMENTED SUBSET WRAPPERS
# ============================================================

class AugStrokeSubset(Dataset):
    def __init__(self, ds, indices, aug_cfg):
        self.ds, self.idx, self.cfg = ds, indices, aug_cfg
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        ri = self.idx[i]
        strokes = augment_strokes(self.ds.strokes_list[ri], self.cfg)
        f = normalize_features(extract_features(strokes))
        f, l = pad_or_truncate(f, self.ds.max_len)
        return torch.tensor(f, dtype=torch.float32), \
               torch.tensor(self.ds.labels_list[ri], dtype=torch.long), l

class AugImageSubset(Dataset):
    def __init__(self, ds, indices):
        self.ds, self.idx = ds, indices
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        fp, ci = self.ds.samples[self.idx[i]]
        img = self.ds.train_tf(Image.open(fp).convert('L'))
        return img, torch.tensor(ci, dtype=torch.long)


# ============================================================
# PLOTTING
# ============================================================

def plot_results(save_dir):
    colors = {'lstm':'#FF4444','liquid':'#4444FF','transformer':'#9333ea','simple_cnn':'#44AA44','resnet18':'#FF8800'}
    histories = {}
    for name in colors:
        p = os.path.join(save_dir, f"{name}_history.json")
        if os.path.exists(p):
            with open(p) as f: histories[name] = json.load(f)

    if not histories: return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('All Models Comparison', fontsize=14, fontweight='bold')

    for name, h in histories.items():
        c = colors[name]
        ep = range(1, len(h['val_loss'])+1)
        axes[0].plot(ep, h['val_loss'], color=c, label=name, linewidth=2)
        axes[1].plot(ep, h['val_acc'], color=c, label=name, linewidth=2)
        axes[2].plot(ep, h['val_top5'], color=c, label=name, linewidth=2)

    axes[0].set_title('Val Loss'); axes[1].set_title('Val Accuracy'); axes[2].set_title('Val Top-5')
    for ax in axes: ax.legend(); ax.grid(True, alpha=0.3); ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot: {save_dir}/comparison.png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train Myanmar HWR models")
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--models', nargs='+', help='Models to train (overrides config)')
    parser.add_argument('--epochs', type=int, help='Override num_epochs')
    parser.add_argument('--dry-run', action='store_true', help='Show config and exit')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs:
        cfg['training']['num_epochs'] = args.epochs

    # Determine which models to train
    model_names = args.models or [m for m, v in cfg['models'].items() if v]

    if args.dry_run:
        print("Config:", json.dumps(cfg, indent=2))
        print(f"\nModels to train: {model_names}")
        return

    # Setup
    seed = cfg['training']['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Models: {model_names}")

    paths = cfg['paths']
    tcfg = cfg['training']
    aug_cfg = cfg.get('augmentation', {})
    save_dir = paths['save_dir']

    # ---- Train each model ----
    for model_name in model_names:
        mcfg = cfg.get(model_name, {})
        is_image = model_name in ('simple_cnn', 'resnet18')

        print(f"\n{'#'*70}")
        print(f"  PREPARING: {model_name.upper()}")
        print(f"{'#'*70}")

        # Load dataset
        if is_image:
            full_ds = ImageDataset(paths['images_dir'], paths['labels_file'],
                                   img_size=mcfg.get('img_size', 64), augment=False, aug_cfg=aug_cfg)
        else:
            full_ds = StrokeDataset(paths['strokes_dir'], paths['labels_file'],
                                    max_len=mcfg.get('max_len', 300), augment=False, aug_cfg=aug_cfg)

        if len(full_ds) == 0:
            print(f"  No data found, skipping {model_name}")
            continue

        num_classes = full_ds.num_classes
        total = len(full_ds)
        val_size = max(1, int(total * tcfg['val_ratio']))
        train_size = total - val_size
        train_sub, val_sub = random_split(full_ds, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(seed))

        # Augmented train
        if is_image:
            train_aug = AugImageSubset(full_ds, train_sub.indices)
        else:
            train_aug = AugStrokeSubset(full_ds, train_sub.indices, aug_cfg)

        oversample = tcfg.get('oversample', 5)
        if oversample > 1:
            train_aug = ConcatDataset([train_aug] * oversample)

        bs = mcfg.get('batch_size', 32)
        train_loader = DataLoader(train_aug, batch_size=bs, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=bs, shuffle=False, num_workers=0)

        print(f"  Samples: {train_size} train, {val_size} val | {num_classes} classes | x{oversample} oversample")

        # Create model
        if model_name == 'lstm':
            model = LSTMClassifier(num_classes, mcfg.get('hidden_size', 128),
                                   mcfg.get('num_layers', 2), mcfg.get('dropout', 0.3))
        elif model_name == 'liquid':
            model = LiquidClassifier(num_classes, mcfg.get('ltc_units', 128), mcfg.get('dropout', 0.3))
        elif model_name == 'simple_cnn':
            model = SimpleCNN(num_classes, mcfg.get('dropout', 0.3))
        elif model_name == 'resnet18':
            model = ResNetClassifier(num_classes, mcfg.get('dropout', 0.3),
                                     mcfg.get('pretrained', True), mcfg.get('freeze_layers', False))
        elif model_name == 'transformer':
            model = TransformerClassifier(num_classes, d_model=mcfg.get('d_model', 128),
                                          nhead=mcfg.get('nhead', 4), num_layers=mcfg.get('num_layers', 3),
                                          dropout=mcfg.get('dropout', 0.3), max_len=mcfg.get('max_len', 400))
        else:
            print(f"  Unknown model: {model_name}"); continue

        trainer = Trainer(model, model_name, device, mcfg.get('lr', 0.001),
                          tcfg.get('grad_clip', 0.5), save_dir)
        trainer.train(train_loader, val_loader, tcfg['num_epochs'], tcfg['patience'])

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    for name in ['lstm', 'liquid', 'transformer', 'simple_cnn', 'resnet18']:
        cp = os.path.join(save_dir, f"{name}_best.pt")
        if os.path.exists(cp):
            c = torch.load(cp, map_location='cpu', weights_only=False)
            print(f"  {name:12s}  acc={c['val_acc']:.4f}  top5={c['val_top5']:.4f}  epoch={c['epoch']}")

    plot_results(save_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()