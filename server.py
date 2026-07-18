"""
=== Server v4: Compare All Models Side by Side ===

Single /predict_all endpoint runs all available models and returns
results for comparison.
"""

import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = Flask(__name__, static_folder='web')
CORS(app)

# ============================================================
# CONFIG
# ============================================================
CHECKPOINTS_DIR = "checkpoints"
LABELS_FILE     = "data/label/syl.txt"
MAX_LEN         = 300
IMG_SIZE        = 64
DEVICE          = "cpu"


# ============================================================
# INLINE MODELS (no import dependency issues)
# ============================================================
from ncps.torch import LTC
from ncps.wirings import AutoNCP
import torchvision.models as models


class LSTMClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=128, num_layers=2, dropout=0.0):
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
            lengths = torch.clamp(lengths, min=1)
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
        return self.classifier((out * aw).sum(dim=1))


class LiquidClassifier(nn.Module):
    def __init__(self, num_classes, ltc_units=128, dropout=0.0):
        super().__init__()
        mn = min(64, ltc_units // 2)
        self.input_proj = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        self.ltc = LTC(32, AutoNCP(ltc_units, mn), batch_first=True)
        self.attention = nn.Sequential(nn.Linear(mn, 32), nn.Tanh(), nn.Linear(32, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(mn, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x, lengths=None):
        B, T, _ = x.shape
        out, _ = self.ltc(self.input_proj(x))
        aw = self.attention(out).squeeze(-1)
        if lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1).to(x.device)
            aw = aw.masked_fill(mask, float('-inf'))
        aw = torch.softmax(aw, dim=1).unsqueeze(-1)
        return self.classifier((out * aw).sum(dim=1))


class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.0):
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
    def __init__(self, num_classes, dropout=0.0, pretrained=False, freeze_layers=False):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

    def forward(self, x, lengths=None):
        return self.resnet(x)


# ============================================================
# MODEL CONFIGS
# ============================================================
MODEL_CONFIGS = {
    "lstm":       {"type": "stroke", "class": LSTMClassifier,
                   "params": {"hidden_size": 128, "num_layers": 2}},
    "liquid":     {"type": "stroke", "class": LiquidClassifier,
                   "params": {"ltc_units": 128}},
    "simple_cnn": {"type": "image",  "class": SimpleCNN, "params": {}},
    "resnet18":   {"type": "image",  "class": ResNetClassifier,
                   "params": {"pretrained": False}},
}


# ============================================================
# LOAD LABELS
# ============================================================
labels = []
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f if l.strip()]
    print(f"Loaded {len(labels)} labels")


# ============================================================
# FEATURE EXTRACTION (for stroke models)
# ============================================================

def extract_features(strokes):
    all_f = []
    for si, stroke in enumerate(strokes):
        for i, (x, y, t) in enumerate(stroke):
            if i == 0:
                dx, dy, speed, angle, curv = 0, 0, 0, 0, 0
            else:
                px, py, pt = stroke[i-1]
                dx, dy, dt = x-px, y-py, t-pt
                speed = math.sqrt(dx**2+dy**2)/dt if dt > 0 else 0
                angle = math.atan2(dy, dx)
                if i >= 2:
                    p2x, p2y, _ = stroke[i-2]
                    pa = math.atan2(py-p2y, px-p2x)
                    curv = angle - pa
                    while curv > math.pi: curv -= 2*math.pi
                    while curv < -math.pi: curv += 2*math.pi
                else:
                    curv = 0
            all_f.append([x, y, dx, dy, speed, angle, curv, 1.0])
        if si < len(strokes)-1:
            nx, ny, _ = strokes[si+1][0]
            all_f.append([nx, ny, 0, 0, 0, 0, 0, 0.0])
    return np.array(all_f, dtype=np.float32)


def normalize_features(f):
    f = f.copy()
    for c in [0, 1]:
        mn, mx = f[:, c].min(), f[:, c].max()
        r = mx - mn
        f[:, c] = (f[:, c]-mn)/r if r > 0 else 0.5
    for c in [2, 3]:
        ma = np.abs(f[:, c]).max()
        if ma > 0: f[:, c] /= ma
    sc = f[:, 4]
    p95 = np.percentile(sc[sc > 0], 95) if np.any(sc > 0) else 1.0
    f[:, 4] = np.clip(sc, 0, p95)/(p95+1e-6)
    f[:, 5] /= math.pi
    f[:, 6] = np.clip(f[:, 6], -math.pi, math.pi)/math.pi
    return f


def pad_or_truncate(f, max_len):
    n = len(f)
    if n >= max_len: return f[:max_len], max_len
    p = np.zeros((max_len, f.shape[1]), dtype=np.float32)
    p[:n] = f
    return p, n


# ============================================================
# STROKES → IMAGE (for CNN models)
# ============================================================

def strokes_to_image(stroke_data, img_size=128, line_width=3):
    all_x = [p['x'] for s in stroke_data for p in s]
    all_y = [p['y'] for s in stroke_data for p in s]
    if not all_x: return None

    pad = 15
    mn_x, mx_x, mn_y, mx_y = min(all_x), max(all_x), min(all_y), max(all_y)
    size = max(mx_x-mn_x+2*pad, mx_y-mn_y+2*pad, 10)
    off_x = (size-(mx_x-mn_x))/2 + pad - mn_x
    off_y = (size-(mx_y-mn_y))/2 + pad - mn_y

    img = Image.new('L', (int(size), int(size)), 255)
    draw = ImageDraw.Draw(img)
    for stroke in stroke_data:
        if len(stroke) < 2: continue
        pts = [(p['x']+off_x, p['y']+off_y) for p in stroke]
        draw.line(pts, fill=0, width=line_width)
    return img.resize((img_size, img_size), Image.LANCZOS)


img_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Lambda(lambda x: 1.0-x),
    T.Normalize(mean=[0.5], std=[0.5]),
])


# ============================================================
# MODEL LOADING
# ============================================================
loaded_models = {}
model_info = {}

def load_model(name):
    if name in loaded_models:
        return loaded_models[name], model_info[name]

    config = MODEL_CONFIGS.get(name)
    if not config: return None, "Unknown model"

    cp_path = os.path.join(CHECKPOINTS_DIR, f"{name}_best.pt")
    if not os.path.exists(cp_path): return None, "No checkpoint"

    try:
        checkpoint = torch.load(cp_path, map_location=DEVICE, weights_only=False)
        sd = checkpoint['model_state_dict']

        # Detect num_classes from last 2D weight tensor
        last_weight = [k for k in sd.keys() if 'weight' in k and sd[k].dim() == 2][-1]
        num_classes = sd[last_weight].shape[0]

        model = config["class"](num_classes=num_classes, **config["params"])
        model.load_state_dict(sd)
        model.eval().to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        info = {
            "epoch": checkpoint.get("epoch", "?"),
            "val_acc": round(checkpoint.get("val_acc", 0)*100, 2),
            "val_top5": round(checkpoint.get("val_top5", 0)*100, 2),
            "num_classes": num_classes,
            "type": config["type"],
            "params": f"{total_params:,}",
        }
        loaded_models[name] = model
        model_info[name] = info
        print(f"  {name.upper()}: acc={info['val_acc']}% | {config['type']} | {info['params']} params")
        return model, info
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, str(e)

print("\nLoading models...")
for name in MODEL_CONFIGS:
    m, r = load_model(name)
    if m is None: print(f"  {name.upper()}: {r}")


# ============================================================
# PREDICTION
# ============================================================

def predict_single(model, model_type, stroke_data, top_k=5):
    if model_type == "stroke":
        strokes = [(p['x'], p['y'], p['t']) for s in stroke_data for p in s]
        strokes_parsed = []
        for s in stroke_data:
            pts = [(p['x'], p['y'], p['t']) for p in s]
            if len(pts) > 1: strokes_parsed.append(pts)
        if not strokes_parsed: return []

        f = normalize_features(extract_features(strokes_parsed))
        f, length = pad_or_truncate(f, MAX_LEN)
        inp = torch.tensor(f, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        length_t = torch.tensor([length])
        with torch.no_grad():
            out = model(inp, length_t)
    else:
        img = strokes_to_image(stroke_data)
        if img is None: return []
        inp = img_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp)

    probs = torch.softmax(out, dim=1)
    tp, ti = probs.topk(min(top_k, probs.size(1)), dim=1)

    results = []
    for i in range(tp.size(1)):
        ci = ti[0, i].item()
        results.append({
            'syllable': labels[ci] if ci < len(labels) else f"class_{ci}",
            'confidence': round(tp[0, i].item()*100, 2),
            'class_id': ci + 1,
        })
    return results


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')


@app.route('/predict_all', methods=['POST'])
def predict_all():
    """Run ALL available models and return compared results"""
    try:
        data = request.get_json()
        strokes = data.get('strokes', [])
        if not strokes:
            return jsonify({'error': 'No strokes'}), 400

        results = {}
        for name in MODEL_CONFIGS:
            if name not in loaded_models:
                continue
            t0 = time.time()
            preds = predict_single(loaded_models[name], MODEL_CONFIGS[name]['type'], strokes)
            elapsed = round((time.time()-t0)*1000, 1)
            results[name] = {
                'predictions': preds,
                'inference_ms': elapsed,
                'type': MODEL_CONFIGS[name]['type'],
                'info': model_info.get(name, {}),
            }

        return jsonify(results)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """Single model prediction (backwards compatible)"""
    try:
        data = request.get_json()
        strokes = data.get('strokes', [])
        model_name = data.get('model', 'lstm')
        if not strokes: return jsonify({'error': 'No strokes'}), 400
        if model_name not in loaded_models:
            return jsonify({'error': f'{model_name} not loaded'}), 400

        preds = predict_single(loaded_models[model_name], MODEL_CONFIGS[model_name]['type'], strokes)
        return jsonify({'predictions': preds, 'model_type': MODEL_CONFIGS[model_name]['type']})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/models')
def get_models():
    available = {}
    for name, config in MODEL_CONFIGS.items():
        if name in model_info:
            available[name] = model_info[name]
        else:
            available[name] = {"status": "not loaded", "type": config["type"]}
    return jsonify(available)


@app.route('/syllables')
def get_syllables():
    return jsonify([{'class_id': i+1, 'syllable': s} for i, s in enumerate(labels)])


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Myanmar HWR — Compare All Models")
    print("  http://localhost:7080")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=7080, debug=False)