# Myanmar Handwriting Recognition

Online handwriting recognition for 1000 Myanmar (Burmese) syllable blocks using stroke sequence data.

Compares **LSTM** (bidirectional with attention) vs **Liquid Neural Network** (LTC) architectures.

## Results

| Model  | Parameters | Val Accuracy | Val Top-5 | Best Epoch |
|--------|-----------|-------------|-----------|------------|
| LSTM   | 875,881   | 87.3%     | 94.2%   | 50         | converged
| Liquid | 358,537   | 51.0%      | 78.6%    | 50         | still learning

## Project Structure

```
burmese-hwr/
├── src/                          # Source code
│   ├── step5b_dataset_augmented.py  # PyTorch Dataset
│   ├── step6_augmentation.py        # Data augmentation
│   ├── step7_models.py              # LSTM + Liquid models
│   └── step8_train.py               # Training loop
├── web/
│   └── index.html                # Drawing UI
├── data/
│   ├── strokes/                  # Stroke .txt files (not in git)
│   └── labels.txt                # 1000 syllable labels
├── checkpoints/                  # Trained models (not in git)
├── server.py                     # Flask prediction server
├── Dockerfile                    # Server container
├── Dockerfile.train              # Training container (GPU)
├── docker-compose.yml            # Container orchestration
└── .github/workflows/ci.yml     # CI/CD pipeline
```

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Train (from project root)
cd src && python step8_train.py

# Serve
python server.py
# Open http://localhost:5000
```

### Docker

```bash
# Build and run server
docker compose up serve

# Run training with GPU
docker compose run train
```

### RunPod (Cloud GPU)

```bash
# This projct was trained on RunPod GPU ___ Using RTXA4000___Upload project
scp -P <port> burmese-hwr.zip root@<ip>:/workspace/

# On RunPod
cd /workspace && unzip burmese-hwr.zip
pip install -r requirements.txt
cd src && python step8_train.py
```

## Data Format

Each handwriting sample is a `.txt` file with stroke sequences:

```
STROKE 1
x y timestamp
x y timestamp
...
STROKE 2
x y timestamp
...
```

Handwritng Collected Files'f format `{class_id}-{sample_id}.txt` (e.g., `901-1.txt`).

## Features

- **8 input features per timestep**: x, y, dx, dy, speed, angle, curvature, pen_state
- **6 augmentation types**: jitter, scale, rotate, translate, speed warp, point dropout
- **Web UI**: draw with mouse/touch, switch models, browse all 1000 syllables
- **Dockerized**: separate containers for training (GPU) and serving (CPU)

## Architecture

**LSTM Classifier**: Bidirectional LSTM → Attention Pooling → FC → 1000 classes

**Liquid Classifier**: Input Projection → LTC (Liquid Time-Constant Network) → Attention Pooling → FC → 1000 classes

## License

MIT