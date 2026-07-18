# Myanmar Handwriting Recognition

Online handwriting recognition for 1,000 Myanmar (Burmese) syllable blocks using stroke sequence and image data.

Compares **5 neural network architectures** across **2 data modalities** (stroke sequences and images) with limited training data (~4 samples per class).

## Results

| Model | Input | Parameters | Accuracy | Top-5 | Status |
|-------|-------|-----------|----------|-------|--------|
| SimpleCNN | Image | 1.8M | **91.1%** | 95.0% | Best |
| Transformer | Stroke | 938K | 13.6% | 34.5% | Overfitting |
| LSTM | Stroke | 875K | 88.2% | 94.1% | Converged |
| ResNet18 | Image | 11.7M | 83.7% | 94.0% | Overfitting |
| Liquid (LTC) | Stroke | 358K | 51.0% | 78.6% | Not converged |

- **Transformer fails with limited data** — self-attention has no sequential inductive bias, needs far more examples to learn meaningful patterns (13.6% vs LSTM's 88.2%)

## Key Findings

- **Custom CNN beats pretrained ResNet18** — model size must match data scale; 1.8M params outperformed 11.7M with only ~4 samples per class
- **Stroke temporal info is competitive** — LSTM (88.2%) nearly matches image-based SimpleCNN (91.1%), showing writing order and speed carry useful signal
- **Bigger is not always better** — ResNet18 with 11.7M parameters overfits severely (train 99.9% vs val 83.7%)
- **Liquid Neural Networks need more compute** — novel ODE-based architecture is 60x slower per epoch than LSTM and hadn't converged after 100 epochs

## Project Structure

```
burmese-hwr/
├── train.py                  # All models, datasets, training (self-contained)
├── config.yaml               # All hyperparameters (no code editing needed)
├── server.py                 # Flask server for live prediction
├── web/
│   └── index.html            # Drawing UI - compare all models side by side
├── data/
│   ├── strokes/              # Stroke .txt files (not in git)
│   ├── images/               # 128x128 grayscale PNGs (not in git)
│   └── labels.txt            # 1,000 syllable labels
├── checkpoints/              # Trained model weights (not in git)
├── Dockerfile                # Server container (CPU)
├── Dockerfile.train          # Training container (GPU)
├── docker-compose.yml        # Container orchestration
├── .github/workflows/ci.yml  # CI/CD pipeline
└── requirements.txt
```

## Quick Start

### Training

```bash
pip install -r requirements.txt

# Train all enabled models
python train.py

# Train specific models
python train.py --models transformer simple_cnn

# Quick test
python train.py --models lstm --epochs 10

# Check config
python train.py --dry-run
```

### Web Demo

```bash
python server.py
# Open http://localhost:5000
# Draw a syllable - all models predict simultaneously
```

### Docker

```bash
docker compose up serve          # start web server
docker compose run train         # train with GPU
```

## Models

### Stroke-based (sequence input)

| Model | Architecture | Key Innovation |
|-------|-------------|---------------|
| LSTM | Bidirectional LSTM - Attention - FC | Sequential processing, strong baseline |
| Transformer | Positional encoding - Self-attention - CLS token - FC | Parallel processing, long-range dependencies |
| Liquid (LTC) | Input projection - LTC Network - Attention - FC | ODE-based, adaptive time constants, compact |

### Image-based (spatial input)

| Model | Architecture | Key Innovation |
|-------|-------------|---------------|
| SimpleCNN | 4x Conv blocks - Global avg pool - FC | Custom, right-sized for limited data |
| ResNet18 | Pretrained ImageNet - Fine-tuned | Transfer learning, 3ch to 1ch adaptation |

## Data

- **1,000 classes**: Myanmar syllable blocks (grapheme clusters)
- **3,934 samples**: ~4 handwritten samples per class, drawn with mouse
- **Dual format**: stroke sequences (x, y, timestamp) + 128x128 grayscale images

### Stroke Features (8 per timestep)

x, y, dx, dy, speed, angle, curvature, pen_state

### Data Augmentation (on-the-fly)

Critical with ~4 samples per class. 5x oversampling per epoch.

- Stroke: jitter, rotation, scale, translate, speed warp, point dropout
- Image: rotation, translate, scale, perspective warp, random erasing

## Configuration

All hyperparameters in config.yaml:

```yaml
models:
  lstm: true
  transformer: true
  simple_cnn: true
  resnet18: true
  liquid: false

transformer:
  d_model: 128
  nhead: 4
  num_layers: 3
  lr: 0.0005
```

## Tech Stack

PyTorch, ncps, torchvision, Flask, Docker, GitHub Actions, RunPod

## License

MIT