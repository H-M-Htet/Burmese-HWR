"""
=== Improved Server: Model Switching + Syllable Reference ===

Features:
  - Switch between LSTM and Liquid models
  - Serve syllable reference list
  - Better error handling
"""

import os
import json
import math
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from step7_models import LSTMClassifier, LiquidClassifier
from step5b_dataset_augmented import extract_features, normalize_features, pad_or_truncate

app = Flask(__name__, static_folder='web')
CORS(app)

# ============================================================
# CONFIG
# ============================================================
CHECKPOINTS_DIR = "checkpoints"
LABELS_FILE     = "data/label/syl.txt"
MAX_LEN         = 300
DEVICE          = "cpu"

# Model configs (must match training)
MODEL_CONFIGS = {
    "lstm": {
        "class": LSTMClassifier,
        "params": {
            "input_size": 8,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.0,
        },
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "lstm_best.pt"),
    },
    "liquid": {
        "class": LiquidClassifier,
        "params": {
            "input_size": 8,
            "ltc_units": 128,
            "dropout": 0.0,
        },
        "checkpoint": os.path.join(CHECKPOINTS_DIR, "liquid_best.pt"),
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
else:
    print(f"WARNING: Labels file not found at {LABELS_FILE}")


# ============================================================
# MODEL LOADING
# ============================================================
loaded_models = {}   # cache: {"lstm": model, "liquid": model}
model_info = {}      # cache: {"lstm": {epoch, val_acc, ...}}

def load_model(model_name):
    """Load a model by name, return (model, info) or (None, error)"""
    if model_name in loaded_models:
        return loaded_models[model_name], model_info[model_name]

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        return None, f"Unknown model: {model_name}"

    if not os.path.exists(config["checkpoint"]):
        return None, f"Checkpoint not found: {config['checkpoint']}"

    try:
        checkpoint = torch.load(config["checkpoint"], map_location=DEVICE, weights_only=False)

        # Detect num_classes from saved weights
        classifier_weight_key = [k for k in checkpoint['model_state_dict'].keys()
                                  if 'classifier' in k and 'weight' in k][-1]
        num_classes = checkpoint['model_state_dict'][classifier_weight_key].shape[0]

        # Create model
        model = config["class"](num_classes=num_classes, **config["params"])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model.to(DEVICE)

        info = {
            "epoch": checkpoint.get("epoch", "?"),
            "val_acc": round(checkpoint.get("val_acc", 0) * 100, 2),
            "val_top5": round(checkpoint.get("val_top5", 0) * 100, 2),
            "num_classes": num_classes,
        }

        # Cache it
        loaded_models[model_name] = model
        model_info[model_name] = info

        print(f"Loaded {model_name.upper()}: epoch={info['epoch']}, "
              f"val_acc={info['val_acc']}%, top5={info['val_top5']}%")

        return model, info

    except Exception as e:
        return None, str(e)


# Pre-load available models
print("\nLoading models...")
for name in MODEL_CONFIGS:
    model, result = load_model(name)
    if model is None:
        print(f"  {name.upper()}: not available ({result})")


# ============================================================
# PREDICTION
# ============================================================

def predict_from_strokes(stroke_data, model_name="lstm", top_k=10):
    model, info = load_model(model_name)
    if model is None:
        return None, info  # info contains error message

    # Convert web format to internal format
    strokes = []
    for stroke in stroke_data:
        points = [(p['x'], p['y'], p['t']) for p in stroke]
        if len(points) > 1:
            strokes.append(points)

    if not strokes:
        return None, "No valid strokes"

    # Extract features
    features = extract_features(strokes)
    features = normalize_features(features)
    features, actual_len = pad_or_truncate(features, MAX_LEN)

    # Predict
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    length_tensor = torch.tensor([actual_len])

    with torch.no_grad():
        output = model(features_tensor, length_tensor)
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
            'class_id': class_idx + 1,  # 1-based for display
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

        return jsonify({'predictions': results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    """Return available models and their info"""
    available = {}
    for name in MODEL_CONFIGS:
        if name in model_info:
            available[name] = model_info[name]
        elif os.path.exists(MODEL_CONFIGS[name]["checkpoint"]):
            available[name] = {"status": "available but not loaded"}
        else:
            available[name] = {"status": "no checkpoint found"}
    return jsonify(available)

@app.route('/syllables')
def get_syllables():
    """Return all syllable labels"""
    syllable_list = []
    for i, syl in enumerate(labels):
        syllable_list.append({
            'class_id': i + 1,
            'syllable': syl,
        })
    return jsonify(syllable_list)

# ============================================================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Burmese Handwriting Recognition Server")
    print("  Open http://localhost:8000 in your browser")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=8000, debug=False)