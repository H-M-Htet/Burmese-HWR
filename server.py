"""
=== Server v3: All 4 Models ===

Supports LSTM, Liquid (stroke-based) and SimpleCNN, ResNet18 (image-based).
For CNN models, strokes from the canvas are rendered into an image first.
"""

import os
import json
import math
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from step7_models import LSTMClassifier, LiquidClassifier
from step9_cnn import SimpleCNN, ResNetClassifier
from step5b_dataset_augmented import extract_features, normalize_features, pad_or_truncate

app = Flask(__name__, static_folder='web')
CORS(app)

# ============================================================
# CONFIG
# ============================================================
CHECKPOINTS_DIR = "checkpoints"
LABELS_FILE     = "data/syl.txt"
MAX_LEN         = 300
IMG_SIZE        = 64
DEVICE          = "cpu"

MODEL_CONFIGS = {
    "lstm": {
        "type": "stroke",
        "class": LSTMClassifier,
        "params": {"input_size": 8, "hidden_size": 128, "num_layers": 2, "dropout": 0.0},
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "lstm_best.pt"),
    },
    "liquid": {
        "type": "stroke",
        "class": LiquidClassifier,
        "params": {"input_size": 8, "ltc_units": 128, "dropout": 0.0},
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "liquid_best.pt"),
    },
    "simple_cnn": {
        "type": "image",
        "class": SimpleCNN,
        "params": {"dropout": 0.0},
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "simple_cnn_best.pt"),
    },
    "resnet18": {
        "type": "image",
        "class": ResNetClassifier,
        "params": {"dropout": 0.0, "pretrained": False},
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "resnet18_best.pt"),
    },
}


# ============================================================
# LOAD LABELS
# ============================================================
labels = []
if os.path.exists(LABELS_FILE):
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(labels)} syllable labels")


# ============================================================
# MODEL LOADING
# ============================================================
loaded_models = {}
model_info = {}

def load_model(model_name):
    if model_name in loaded_models:
        return loaded_models[model_name], model_info[model_name]

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return None, f"Unknown model: {model_name}"

    if not os.path.exists(config["checkpoint"]):
        return None, f"Checkpoint not found: {config['checkpoint']}"

    try:
        checkpoint = torch.load(config["checkpoint"], map_location=DEVICE, weights_only=False)

        # Detect num_classes
        classifier_weight_key = [k for k in checkpoint['model_state_dict'].keys()
                                  if 'classifier' in k and 'weight' in k][-1]
        num_classes = checkpoint['model_state_dict'][classifier_weight_key].shape[0]

        model = config["class"](num_classes=num_classes, **config["params"])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        info = {
            "epoch": checkpoint.get("epoch", "?"),
            "val_acc": round(checkpoint.get("val_acc", 0) * 100, 2),
            "val_top5": round(checkpoint.get("val_top5", 0) * 100, 2),
            "num_classes": num_classes,
            "type": config["type"],
            "params": f"{total_params:,}",
        }

        loaded_models[model_name] = model
        model_info[model_name] = info

        print(f"  Loaded {model_name.upper()}: epoch={info['epoch']}, "
              f"acc={info['val_acc']}%, type={config['type']}")

        return model, info
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)


# Pre-load
print("\nLoading models...")
for name in MODEL_CONFIGS:
    model, result = load_model(name)
    if model is None:
        print(f"  {name.upper()}: not available ({result})")


# ============================================================
# STROKE → IMAGE CONVERSION (for CNN models)
# ============================================================

def strokes_to_image(stroke_data, img_size=128, line_width=3):
    """
    Render web canvas strokes into a grayscale image.
    Matches the format of your training .png images.
    """
    # Find bounding box
    all_x = [p['x'] for s in stroke_data for p in s]
    all_y = [p['y'] for s in stroke_data for p in s]

    if not all_x:
        return None

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    # Add padding
    padding = 15
    width = max_x - min_x + 2 * padding
    height = max_y - min_y + 2 * padding

    # Make square
    size = max(width, height, 10)
    offset_x = (size - width) / 2 + padding - min_x
    offset_y = (size - height) / 2 + padding - min_y

    # Create white image
    img = Image.new('L', (int(size), int(size)), 255)
    draw = ImageDraw.Draw(img)

    # Draw strokes in black
    for stroke in stroke_data:
        if len(stroke) < 2:
            continue
        points = [(p['x'] + offset_x, p['y'] + offset_y) for p in stroke]
        draw.line(points, fill=0, width=line_width)

    # Resize to target size
    img = img.resize((img_size, img_size), Image.LANCZOS)

    return img


# ============================================================
# PREDICTION
# ============================================================

# Image transform (must match training)
img_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Lambda(lambda x: 1.0 - x),       # invert: strokes=white, bg=black
    T.Normalize(mean=[0.5], std=[0.5]),
])


def predict_from_strokes(stroke_data, model_name="lstm", top_k=10):
    model, info = load_model(model_name)
    if model is None:
        return None, info

    config = MODEL_CONFIGS[model_name]

    if config["type"] == "stroke":
        # --- Stroke-based models (LSTM, Liquid) ---
        strokes = []
        for stroke in stroke_data:
            points = [(p['x'], p['y'], p['t']) for p in stroke]
            if len(points) > 1:
                strokes.append(points)

        if not strokes:
            return None, "No valid strokes"

        features = extract_features(strokes)
        features = normalize_features(features)
        features, actual_len = pad_or_truncate(features, MAX_LEN)

        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        length_tensor = torch.tensor([actual_len])

        with torch.no_grad():
            output = model(input_tensor, length_tensor)

    else:
        # --- Image-based models (SimpleCNN, ResNet18) ---
        img = strokes_to_image(stroke_data)
        if img is None:
            return None, "Could not render image"

        input_tensor = img_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

    # Get top-k predictions
    probabilities = torch.softmax(output, dim=1)
    top_probs, top_indices = probabilities.topk(min(top_k, probabilities.size(1)), dim=1)

    results = []
    for i in range(top_probs.size(1)):
        class_idx = top_indices[0, i].item()
        confidence = top_probs[0, i].item()
        syllable = labels[class_idx] if class_idx < len(labels) else f"class_{class_idx}"
        results.append({
            'syllable': syllable,
            'confidence': round(confidence * 100, 2),
            'class_id': class_idx + 1,
        })

    return results, None


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    return send_from_directory('web', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        strokes = data.get('strokes', [])
        model_name = data.get('model', 'lstm')

        if not strokes:
            return jsonify({'error': 'No strokes provided'}), 400

        results, error = predict_from_strokes(strokes, model_name)
        if error:
            return jsonify({'error': error}), 400

        return jsonify({'predictions': results, 'model_type': MODEL_CONFIGS[model_name]['type']})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    available = {}
    for name, config in MODEL_CONFIGS.items():
        if name in model_info:
            available[name] = model_info[name]
        elif os.path.exists(config["checkpoint"]):
            available[name] = {"status": "available", "type": config["type"]}
        else:
            available[name] = {"status": "no checkpoint", "type": config["type"]}
    return jsonify(available)

@app.route('/syllables')
def get_syllables():
    return jsonify([{'class_id': i+1, 'syllable': s} for i, s in enumerate(labels)])


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Myanmar Handwriting Recognition Server")
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)