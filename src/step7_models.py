"""
=== Step 7: Models — LSTM vs Liquid Neural Network ===

Two models that take the SAME input and produce the SAME output:
  Input:  (batch_size, max_len, 8)  →  sequence of 8 features
  Output: (batch_size, num_classes)  →  prediction scores for each class

The only difference is HOW they process the sequence internally.
"""

import torch
import torch.nn as nn
from ncps.torch import LTC
from ncps.wirings import AutoNCP


# ============================================================
# MODEL 1: LSTM (Baseline)
# ============================================================

class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM with attention pooling.

    How it works:
      1. LSTM reads the sequence forward AND backward
      2. Attention layer learns WHICH time steps matter most
      3. Fully connected layers classify into num_classes

    Architecture:
      Input (batch, time, 8)
        → LSTM (bidirectional)
        → Attention pooling
        → Dropout
        → FC layer
        → Output (batch, num_classes)
    """

    def __init__(self, input_size=8, hidden_size=128, num_layers=2,
                 num_classes=1000, dropout=0.3):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM: processes sequence step by step
        # bidirectional=True means it reads forward AND backward
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,       # input shape: (batch, time, features)
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention: learns which time steps are important
        # bidirectional → output is 2 * hidden_size
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, max_len, 8) - padded feature sequences
            lengths: actual lengths (before padding), used to create mask
        """
        batch_size = x.size(0)
        max_len = x.size(1)

        # Pack padded sequences for efficient LSTM processing
        if lengths is not None:
            # Sort by length (required for pack_padded_sequence)
            lengths = torch.clamp(lengths, min=1)  # ensure no zero lengths
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=max_len
            )
        else:
            lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)

        # === Attention pooling ===
        # Instead of just taking the last hidden state,
        # let the model learn which time steps matter
        attn_weights = self.attention(lstm_out)     # (batch, time, 1)
        attn_weights = attn_weights.squeeze(-1)     # (batch, time)

        # Mask out padding positions
        if lengths is not None:
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch, time)
        attn_weights = attn_weights.unsqueeze(-1)           # (batch, time, 1)

        # Weighted sum of LSTM outputs
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        # Classify
        output = self.classifier(context)  # (batch, num_classes)
        return output


# ============================================================
# MODEL 2: Liquid Neural Network (LTC)
# ============================================================

class LiquidClassifier(nn.Module):
    """
    Liquid Time-Constant Network for sequence classification.

    How it works:
      1. Linear layer projects input to match LTC input size
      2. LTC processes the sequence with adaptive time constants
         (neurons adjust their speed based on input — key innovation!)
      3. Attention pooling over LTC outputs
      4. Fully connected layers classify

    Key advantage: much fewer parameters than LSTM,
    and naturally handles variable-speed input (like handwriting!)

    Architecture:
      Input (batch, time, 8)
        → Linear projection
        → LTC (Liquid Time-Constant network)
        → Attention pooling
        → Dropout
        → FC layer
        → Output (batch, num_classes)
    """

    def __init__(self, input_size=8, ltc_units=128, num_classes=1000,
                 dropout=0.3):
        super().__init__()

        self.ltc_units = ltc_units

        # Project input features to a good size for LTC
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
        )

        # Wiring: defines how LTC neurons connect to each other
        # AutoNCP automatically creates a good wiring pattern
        #   units: total number of neurons
        #   output_size: number of output neurons (motor neurons)
        self.wiring = AutoNCP(
            units=ltc_units,
            output_size=64,    # motor neurons that produce output
        )

        # LTC: the liquid neural network layer
        self.ltc = LTC(
            input_size=32,     # must match input_proj output
            units=self.wiring,
            batch_first=True,
        )

        # Attention pooling (same idea as LSTM)
        self.attention = nn.Sequential(
            nn.Linear(64, 32),  # 64 = wiring output_size
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, max_len, 8)
            lengths: actual sequence lengths
        """
        batch_size = x.size(0)
        max_len = x.size(1)

        # Project input
        x = self.input_proj(x)  # (batch, time, 32)

        # LTC processes the sequence
        # Returns: (output, hidden_state)
        ltc_out, _ = self.ltc(x)  # (batch, time, 64)

        # === Attention pooling ===
        attn_weights = self.attention(ltc_out)      # (batch, time, 1)
        attn_weights = attn_weights.squeeze(-1)     # (batch, time)

        # Mask padding
        if lengths is not None:
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(mask, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1)

        # Weighted sum
        context = (ltc_out * attn_weights).sum(dim=1)  # (batch, 64)

        # Classify
        output = self.classifier(context)  # (batch, num_classes)
        return output


# ============================================================
# HELPER: Count parameters
# ============================================================

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# TEST IT
# ============================================================
if __name__ == "__main__":

    NUM_CLASSES = 1000
    BATCH_SIZE = 4
    MAX_LEN = 300
    INPUT_SIZE = 8

    # Create dummy input
    dummy_input = torch.randn(BATCH_SIZE, MAX_LEN, INPUT_SIZE)
    dummy_lengths = torch.tensor([300, 250, 200, 150])

    print("=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)

    # --- LSTM ---
    print("\n--- LSTM Classifier ---")
    lstm_model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    )
    lstm_out = lstm_model(dummy_input, dummy_lengths)
    total, trainable = count_parameters(lstm_model)
    print(f"  Output shape:          {lstm_out.shape}")
    print(f"  Total parameters:      {total:,}")
    print(f"  Trainable parameters:  {trainable:,}")

    # --- Liquid ---
    print("\n--- Liquid (LTC) Classifier ---")
    liquid_model = LiquidClassifier(
        input_size=INPUT_SIZE,
        ltc_units=128,
        num_classes=NUM_CLASSES,
        dropout=0.3,
    )
    liquid_out = liquid_model(dummy_input, dummy_lengths)
    total_l, trainable_l = count_parameters(liquid_model)
    print(f"  Output shape:          {liquid_out.shape}")
    print(f"  Total parameters:      {total_l:,}")
    print(f"  Trainable parameters:  {trainable_l:,}")

    # --- Comparison ---
    print(f"\n--- Size Comparison ---")
    print(f"  LSTM parameters:    {trainable:,}")
    print(f"  Liquid parameters:  {trainable_l:,}")
    ratio = trainable / trainable_l
    print(f"  LSTM is {ratio:.1f}x larger than Liquid")

    # --- Verify both produce valid output ---
    print(f"\n--- Output Check ---")
    print(f"  LSTM predictions:   {torch.argmax(lstm_out, dim=1).tolist()}")
    print(f"  Liquid predictions: {torch.argmax(liquid_out, dim=1).tolist()}")
    print(f"\n✓ Both models working! Ready for training.")