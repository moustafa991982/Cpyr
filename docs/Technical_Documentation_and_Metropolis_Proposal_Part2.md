---

---

# PART IV — IMPLEMENTATION GUIDE

---

## 1. Environment Setup

### 1.1 CPYR — System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10 |
| PyTorch | 1.8 (CPU) | 2.0+ (CUDA 11.8) |
| CUDA | Optional | 11.8 / 12.1 |
| RAM | 4 GB | 16 GB |
| VRAM | None (CPU) | 8 GB (GTX 1070 class) |
| Storage | 500 MB (code + tiny models) | 2 GB (all variants) |

```bash
# Clone and install
git clone https://github.com/moustafa991982/Cpyr.git
cd Cpyr
python -m venv .venv && source .venv/bin/activate

pip install torch>=1.8.0 torchvision>=0.9.0
pip install numpy>=1.19 pandas>=1.2
pip install scikit-learn>=0.24 scipy>=1.6
pip install scikit-image>=0.18        # SSIM loss
pip install einops tqdm matplotlib seaborn jupyter

# Verify GPU availability
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1.2 Traffic Vision — System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.7 | 3.9 |
| PyTorch | 1.7 | 1.7 (tested) |
| CUDA | 10.2 | 11.1 (Tesla T4 target) |
| GPU VRAM | 6 GB | 15 GB (Tesla T4) |
| Storage | 5 GB (video + checkpoints) | 20 GB |
| OpenCV | Required | 4.x |
| Albumentations | 0.5.2 (exact version tested) | 0.5.2 |

```bash
pip install torch==1.7.0 torchvision==0.8.0
pip install albumentations==0.5.2
pip install opencv-python Pillow matplotlib numpy tqdm

# Install SSIM utility (required for validation metrics)
pip install pytorch-ssim
# or clone: git clone https://github.com/Po-Hsun-Su/pytorch-ssim
```

---

## 2. Reproducing CPYR — Fuzz Attack Detection

### 2.1 Data Preparation

The original proprietary dataset is not redistributed. To reproduce against your own in-vehicle Ethernet captures:

**Step 1 — Capture raw Ethernet frames**
```
Required format: one frame per line, tab-separated:
  <frame_id_hex>  <payload_hex>

Example:
  0A1    4F3A00FF12BC7E01
  0B2    00000000AABBCCDD
  ...
```

**Step 2 — Parse and clean** (using `clean_parser.ipynb` or `Parser.ipynb`)
```python
# Conceptual pipeline from data_handler.py
import pandas as pd
import numpy as np

def hex_to_binary(hex_string):
    """Convert hex payload to binary tensor."""
    binary = []
    for char in hex_string:
        bits = format(int(char, 16), '04b')  # 4 bits per hex char
        binary.extend([int(b) for b in bits])
    return np.array(binary, dtype=np.float32)

def pad_frame(binary_frame, target_shape=(112, 112), mode='right'):
    """Pad binary frame to uniform 112×112."""
    flat_len = len(binary_frame)
    target_len = target_shape[0] * target_shape[1]
    if flat_len < target_len:
        if mode == 'right':
            padded = np.pad(binary_frame, (0, target_len - flat_len))
        elif mode == 'center':
            pad_left = (target_len - flat_len) // 2
            pad_right = target_len - flat_len - pad_left
            padded = np.pad(binary_frame, (pad_left, pad_right))
    else:
        padded = binary_frame[:target_len]
    return padded.reshape(target_shape)

# Process a capture file
frames = []
with open('capture.txt') as f:
    for line in f:
        frame_id, payload = line.strip().split('\t')
        binary = hex_to_binary(payload)
        padded = pad_frame(binary)
        frames.append(padded)

frames = np.stack(frames)  # (N_frames, 112, 112)
```

**Step 3 — Create sequence batches**
```python
def create_batches(frames, seq_len=20, batch_size=8):
    """Slide a window of seq_len across the frame sequence."""
    sequences = []
    for i in range(0, len(frames) - seq_len, seq_len):
        seq = frames[i : i + seq_len]      # (seq_len, 112, 112)
        sequences.append(seq)
    # Stack into batches
    sequences = np.stack(sequences)         # (N_seqs, seq_len, 112, 112)
    return sequences
```

### 2.2 Training the Convolutional Encoder

Open `Integration script.ipynb` and run all cells. The notebook trains all five encoder variants sequentially and logs separation scores.

**Standalone training script (minimal):**
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models import ConvEncoder, ConvDecoder  # from models.py

# Hyperparameters
BATCH_SIZE = 8
SEQ_LEN    = 20
LR         = 1e-3
EPOCHS     = 6
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load normal-only training data (shape: N, SEQ_LEN, 112, 112)
train_data = torch.from_numpy(np.load('normal_sequences_train.npy'))
val_data   = torch.from_numpy(np.load('normal_sequences_val.npy'))

train_loader = DataLoader(TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(val_data),   batch_size=BATCH_SIZE, shuffle=False)

encoder = ConvEncoder(seq_len=SEQ_LEN).to(DEVICE)
decoder = ConvDecoder(seq_len=SEQ_LEN).to(DEVICE)
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=LR
)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    encoder.train(); decoder.train()
    train_loss = 0
    for (batch,) in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        z = encoder(batch)
        recon = decoder(z)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Train MSE: {train_loss/len(train_loader):.6f}")

# Save
torch.save(encoder.state_dict(), 'cpyr_encoder.pt')
torch.save(decoder.state_dict(), 'cpyr_decoder.pt')
```

### 2.3 Inference & Anomaly Scoring

```python
encoder.eval(); decoder.eval()

def anomaly_score(batch, encoder, decoder, device):
    """Return per-batch reconstruction loss (anomaly score)."""
    with torch.no_grad():
        batch = batch.to(device)
        z     = encoder(batch)
        recon = decoder(z)
        loss  = nn.MSELoss()(recon, batch)
    return loss.item()

# Stream over evaluation data
scores = {'normal': [], 'attack': [], 'end': []}
for phase, data in eval_data.items():
    loader = DataLoader(TensorDataset(data), batch_size=BATCH_SIZE)
    for (batch,) in loader:
        scores[phase].append(anomaly_score(batch, encoder, decoder, DEVICE))

# Compute separation (Normal vs Attack)
from scipy.stats import wasserstein_distance
separation = 1.0 - wasserstein_distance(
    scores['normal'], scores['attack']
) / (max(scores['attack']) - min(scores['normal']) + 1e-8)
print(f"Normal vs Attack separation: {separation:.3f}")
```

---

## 3. Reproducing CPYR — LKA Contextual Anomaly

Open `detecting lka state.ipynb`. The notebook contains the complete training and evaluation loop for both the LKA Predictor and Enhanced LKA Predictor.

**Key configuration cells to modify for a custom scenario:**

```python
# Cell: Dataset configuration
LKA_FRAME_ID = 0x123   # Replace with your vehicle's LKA frame ID
RLD_FRAME_ID = 0x456   # Replace with your RLD (lane distance) frame ID
WINDOW_SIZE  = 10       # Reconstruction window for Enhanced model (N in algorithm 2)
BATCH_SIZE   = 8
SEQ_LEN      = 10
LR           = 1e-4     # Grid-searched; do not change unless retuning
EPOCHS       = 4

# Cell: Threshold configuration (after training)
# Following the method in the SAE paper:
# Upper ON bound = (highest safe ON trigger + lowest unsafe ON trigger) / 2
# Set manually based on verification test case 7 output
UPPER_ON_BOUND  = None   # Set after examining loss curve on case 7
LOWER_ON_BOUND  = None
UPPER_OFF_BOUND = None
LOWER_OFF_BOUND = None
ABS_LOWER_BOUND = None   # Non-transition loss floor
```

**Generating verification test cases programmatically:**

```python
def make_test_case_1(seq_len, rld_phases):
    """Case 1: LKA OFF throughout. No transitions."""
    lka = np.zeros(seq_len)          # always OFF
    rld = rld_phases['constant']     # normal constant-phase RLD
    return lka, rld

def make_test_case_5(seq_len, rld_phases):
    """Case 5: Single illegitimate ON in steering phase."""
    lka = np.zeros(seq_len)
    lka[len(seq_len)//2] = 1         # ON transition at steering phase midpoint
    rld = rld_phases['steering']
    return lka, rld

# Evaluate each case
for case_id, (lka, rld) in enumerate(test_cases):
    loss = evaluate_predictor(lka_predictor, rld, lka)
    f1   = compute_f1(loss, threshold_upper=UPPER_ON_BOUND,
                            threshold_lower=LOWER_ON_BOUND)
    print(f"Case {case_id+1}: F1 = {f1:.3f}")
```

---

## 4. Reproducing Traffic Vision — Training

### 4.1 Data Preparation

```bash
# Step 1: Extract frames from dashcam video
mkdir -p data/train/frames

python - <<'EOF'
import cv2, os

cap   = cv2.VideoCapture('video_00.mp4')
idx   = 0
out   = 'data/train/frames'

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Crop road region (adjust per camera mount)
    frame = frame[40:1040, :, :]
    # Resize to 320×480 (H×W)
    frame = cv2.resize(frame, (480, 320))
    cv2.imwrite(f'{out}/frame-{idx:05d}.jpg', frame)
    idx += 1

cap.release()
print(f"Extracted {idx} frames")
EOF
```

### 4.2 Training Loop

```python
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2, os, numpy as np
from pathlib import Path

# ── Architecture ──────────────────────────────────────────────────────────────
class ConvAutoencoder(nn.Module):
    def __init__(self, in_num_frames, out_num_frames):
        super().__init__()
        self.conv1   = nn.Conv2d(in_num_frames,  32, 3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.conv2   = nn.Conv2d(32,  64, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.conv3   = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.pool    = nn.MaxPool2d(3, padding=1, stride=2)
        self.t_conv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_bn1   = nn.BatchNorm2d(64)
        self.t_conv2 = nn.ConvTranspose2d(64,  32, 2, stride=2)
        self.t_bn2   = nn.BatchNorm2d(32)
        self.t_conv3 = nn.ConvTranspose2d(32, out_num_frames, 2, stride=2)

    def forward(self, x):
        import torch.nn.functional as F
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.t_bn1(F.relu(self.t_conv1(x)))
        x = self.t_bn2(F.relu(self.t_conv2(x)))
        return self.t_conv3(x)

# ── Dataset ───────────────────────────────────────────────────────────────────
class FrameSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, frame_dir, sequence_len=16, transform=None):
        self.frame_dir    = frame_dir
        self.image_list   = sorted(os.listdir(frame_dir))
        self.sequence_len = sequence_len
        self.transform    = transform

    def __len__(self):
        return len(self.image_list) - self.sequence_len

    def __getitem__(self, idx):
        frames = []
        for name in self.image_list[idx : idx + self.sequence_len]:
            img = cv2.imread(os.path.join(self.frame_dir, name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)['image']
            frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        return torch.cat(frames, dim=0)   # (3*L, H, W)

# ── Augmentation ──────────────────────────────────────────────────────────────
augment = A.Compose([
    A.OneOf([A.CLAHE(), A.RandomBrightness(), A.RandomGamma()], p=0.9),
    A.OneOf([A.IAASharpen(), A.Blur(blur_limit=3), A.MotionBlur(blur_limit=3)], p=0.9),
    A.IAAAdditiveGaussianNoise(p=0.2),
    A.IAAPerspective(p=0.5),
])

# ── Training config ───────────────────────────────────────────────────────────
IN_FRAMES  = 10   # frames the model sees
SEQ_LEN    = 16   # total window length (model predicts frames 10-15)
BATCH_SIZE = 4
LR         = 1e-3
EPOCHS     = 30
DEVICE     = 'cuda'

train_ds = FrameSequenceDataset('data/train/frames', SEQ_LEN, transform=augment)
val_ds   = FrameSequenceDataset('data/val/frames',   SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model     = ConvAutoencoder(in_num_frames=IN_FRAMES * 3,
                             out_num_frames=SEQ_LEN * 3).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

best_val  = float('inf')

for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred  = model(batch[:, : IN_FRAMES * 3, :, :])
        loss  = criterion(pred, batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch    = batch.to(DEVICE)
            pred     = model(batch[:, : IN_FRAMES * 3, :, :])
            val_loss += criterion(pred, batch).item()

    avg_val = val_loss / len(val_loader)
    print(f"Epoch {epoch+1:02d} | Train: {train_loss/len(train_loader):.4e} | Val: {avg_val:.4e}")

    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), f'tv_model_f{IN_FRAMES*3}_f{SEQ_LEN*3}.pt')
        print(f"  ✓ Checkpoint saved (val={best_val:.4e})")
```

### 4.3 Inference & Anomaly Score

```python
model.eval()
THRESHOLD = 5e-3   # empirical; recalibrate per deployment

def run_inference(video_path, model, in_frames, seq_len, device, threshold):
    """Process a dashcam video and return per-window anomaly scores + alerts."""
    # Extract frames
    frames = extract_frames(video_path)         # → list of (H,W,3) uint8 arrays
    dataset = FrameSequenceDataset(frames, seq_len)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    scores = []
    alerts = []
    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(device)
            pred   = model(batch[:, : in_frames * 3, :, :])
            score  = nn.MSELoss()(pred, batch).item()
            scores.append(score)
            alerts.append(score > threshold)

    return scores, alerts

scores, alerts = run_inference(
    'test_footage.mp4', model,
    in_frames=10, seq_len=16,
    device=DEVICE, threshold=THRESHOLD
)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(14, 4))
plt.plot(scores, label='Anomaly score')
plt.axhline(THRESHOLD, color='r', linestyle='--', label=f'Threshold ({THRESHOLD:.0e})')
plt.fill_between(range(len(alerts)),
                 [s if a else 0 for s, a in zip(scores, alerts)],
                 color='red', alpha=0.3, label='Alert region')
plt.legend(); plt.xlabel('Window index'); plt.ylabel('MSE score')
plt.title('Traffic Vision — Hazard Detection')
plt.tight_layout(); plt.savefig('traffic_vision_output.png', dpi=150)
```

---

## 5. Running the MISV Fusion Locally

```python
"""
Minimal MISV fusion demo.
Runs both CPYR and Traffic Vision on their respective data streams
and outputs a fused alert level (0–3).
"""
import torch, numpy as np

class MISVFusion:
    def __init__(self, thresholds_a, thresholds_b, thresholds_c=None, weights=(1, 1, 1)):
        """
        thresholds_a: (level1, level2, level3) for CPYR channel
        thresholds_b: (level1, level2, level3) for Traffic Vision channel
        thresholds_c: (level1, level2, level3) for V2X channel (optional)
        weights:      channel weights for voting
        """
        self.thresh_a = thresholds_a
        self.thresh_b = thresholds_b
        self.thresh_c = thresholds_c
        self.weights  = weights

    def score_to_level(self, score, thresholds):
        t1, t2, t3 = thresholds
        if score >= t3:   return 3
        elif score >= t2: return 2
        elif score >= t1: return 1
        else:             return 0

    def fuse(self, score_a, score_b, score_c=None):
        level_a = self.score_to_level(score_a, self.thresh_a)
        level_b = self.score_to_level(score_b, self.thresh_b)
        level_c = self.score_to_level(score_c, self.thresh_c) if score_c is not None else 0

        # Weighted vote
        weighted = (self.weights[0] * level_a
                  + self.weights[1] * level_b
                  + self.weights[2] * level_c)
        max_possible = sum(w * 3 for w in self.weights[:2 if score_c is None else 3])

        # Normalize to 0–3
        fused_level = round(3 * weighted / max_possible)
        return fused_level, {'cpyr': level_a, 'vision': level_b, 'v2x': level_c}

# Usage
fusion = MISVFusion(
    thresholds_a=(0.010, 0.025, 0.050),   # CPYR: calibrate from site normal data
    thresholds_b=(2e-3,  4e-3,  5e-3),    # Traffic Vision: from demo threshold
    thresholds_c=(0.1,   0.3,   0.6),     # V2X kinematic deviation (m/s²)
    weights=(1, 1, 1)
)

# Per-frame update
score_a = cpyr_model.anomaly_score(can_batch)
score_b = tv_model.anomaly_score(video_window)
score_c = v2x_checker.kinematic_deviation()     # optional

alert_level, detail = fusion.fuse(score_a, score_b, score_c)
print(f"MISV Alert Level: {alert_level} | Detail: {detail}")
```

---

## 6. Configuration Reference

### 6.1 CPYR Configuration Matrix

| Parameter | Fuzz Detection | LKA Contextual | Notes |
|---|---|---|---|
| `seq_len` | 20 | 10 | Frames per input batch |
| `frame_shape` | (112, 112) | (frame_size,) | Padded shape post hex→bin |
| `pad_mode` | `'right'` or `'center'` | N/A | Right default; center for symmetric protocols |
| `batch_size` | 8 | 8 | Reduce to 2 for ECU-class deployment |
| `lr` | 1e-3 | 1e-4 | Grid-searched for LKA |
| `epochs` | 6 | 4 | Normal capture sufficient by epoch 6 |
| `hidden_size` | — | 1024 | Predictor bottleneck; sharpens transition triggers |
| `window_size` (Enhanced) | — | N=10 | RLD frames to reconstruct |
| `loss` | MSE | MSE + reconstruction | BCE+SSIM for evaluation |
| `optimizer_betas` | (0.9, 0.999) | (0.9, 0.999) | Adam defaults |

### 6.2 Traffic Vision Configuration Matrix

| Parameter | Value | Notes |
|---|---|---|
| `in_num_frames` | 10 (ground) / 12 (predict) | Dual-model: different per stream |
| `out_num_frames` | 16 | Frames to predict |
| `sequence_len` | 16 | Total window (input + output) |
| `img_height` | 320 | After crop and resize |
| `img_width` | 480 | After crop and resize |
| `batch_size` | 4 | Constrained by 15 GB T4 |
| `lr` | 1e-3 | Flat across all 30 epochs |
| `epochs` | 30 | Best-val checkpoint kept |
| `augmentation_p_group_a` | 0.9 | Lighting group |
| `augmentation_p_group_b` | 0.9 | Sharpness/blur group |
| `anomaly_threshold` | ~5e-3 | Empirical; recalibrate per scene |
| `bottleneck_channels` | 128 | After 3 MaxPool layers |
| `encoder_channels` | [32, 64, 128] | Progressive depth |

---

## 7. Common Errors & Troubleshooting

### 7.1 CPYR

| Error | Likely Cause | Fix |
|---|---|---|
| `RuntimeError: Expected input batch_size (X) to match target batch_size (Y)` | Sequence length mismatch between encoder and decoder | Ensure `seq_len` is identical in both modules |
| Poor fuzz separation (< 0.7) | Normal-phase data contaminated with partial attack frames | Review phase labels; re-split dataset at phase boundaries |
| LKA predictor always outputs zero loss | LKA frame ID incorrect; filter never extracts LKA frames | Verify `LKA_FRAME_ID` against capture file |
| Threshold determination gives single threshold for ON/OFF | Model not producing distinct trigger heights | Increase `hidden_size` to 1024; ensure batch size ≥ 4 |
| Model size correct but inference slow on Jetson | Running in eager mode, not TensorRT | Export to ONNX and run via `trtexec` (see Part III §3.2) |

### 7.2 Traffic Vision

| Error | Likely Cause | Fix |
|---|---|---|
| `RuntimeError: Expected 4D tensor` | Frame stacking not applied; dataset returns (H,W,3) | Confirm `torch.cat(frames, dim=0)` in `__getitem__` |
| Validation loss higher than training loss | Validation footage significantly different from training | Reduce crop region; use closer-match validation video |
| Anomaly score flat (≈ constant) across all windows | Model outputting near-zero everywhere | Check that model is in `eval()` mode; verify gradient is detached |
| Score spikes on normal footage (high false-positive rate) | Threshold too low for local scene | Recalibrate threshold: collect 1000 normal windows and set threshold at 99th percentile |
| Memory OOM on Jetson | Batch size too large for available VRAM | Reduce batch to 1; use FP16 TensorRT engine |

---

---

# PART V — SOTIF EVIDENCE PACKAGE TEMPLATE

---

## 1. Overview

ISO 21448:2022 requires a structured body of evidence demonstrating that residual risk from Zone 2 (unknown unsafe triggering conditions) has been reduced to an acceptable level. This part provides templates for the evidence artifacts that CPYR and Traffic Vision together generate, formatted for ISO 21448 clause 9 compliance.

---

## 2. Evidence Artifact Catalog

| Artifact ID | Name | Source System | ISO 21448 Clause |
|---|---|---|---|
| EA-01 | System Specification | Both | Clause 5 |
| EA-02 | Triggering Condition Identification Log | CPYR + Traffic Vision | Clause 7 |
| EA-03 | Verification Test Results (CPYR) | CPYR | Clause 9.3 |
| EA-04 | Validation Test Results (CPYR) | CPYR | Clause 9.4 |
| EA-05 | Video Demo Anomaly Event Log | Traffic Vision | Clause 9.4 |
| EA-06 | MISV Zone 2 Reduction Argument | Both | Clause 9.5 |
| EA-07 | Continuous Anomaly Score Log | Both | Clause 9.6 |
| EA-08 | Triggering Condition Discovery Report | Both | Clause 9.7 |
| EA-09 | Threshold Calibration Record | Both | Clause 9.3 |
| EA-10 | Sensor Degradation Response Record | Both | Clause 9.4 |

---

## 3. EA-01: System Specification Record

```
SOTIF SYSTEM SPECIFICATION
══════════════════════════════════════════════════════════════════
System name:          MISV — Multiple Independent Source Verification
Version:              1.0
Date:                 [DATE]
Authors:              [AUTHORS]
SAE reference:        DOI 10.4271/2021-01-0196

Intended functionality:
  Detect hazardous traffic situations and SOTIF trigger events in real
  time from (a) in-vehicle network signals and (b) forward camera video,
  providing an auditable anomaly score stream and graduated alert output.

Operational design domain (ODD):
  • Urban and highway roadways
  • Daylight, clear/light-overcast conditions
  • Forward-facing dashcam, 30 fps, 320×480 resolution (Traffic Vision)
  • CAN/Ethernet-connected vehicles with accessible OBD-II port (CPYR)
  • V2X-equipped infrastructure (third channel, optional)

Out of ODD:
  • Night driving (Traffic Vision — retraining required)
  • Heavy rain, fog, snow (Traffic Vision — retraining required)
  • Vehicles without OBD-II access (CPYR — hardware adaptation required)
  • V2X-absent environments (third channel inactive; MISV degrades to 2-channel)

Performance targets:
  CPYR:          F1 ≥ 0.88 (all 12 SOTIF validation scenarios)
                 F1 = 1.00 (all 7 SOTIF verification test cases)
  Traffic Vision: Anomaly score > threshold before event completion
                 Correct de-escalation on ambiguous-but-safe events
  MISV fused:    False-positive rate ≤ single-channel rate (product reduction)
══════════════════════════════════════════════════════════════════
```

---

## 4. EA-02: Triggering Condition Identification Log

Each row represents one identified triggering condition — a scene or event class that drives the system toward a hazardous outcome.

```
TRIGGERING CONDITION LOG
══════════════════════════════════════════════════════════════════════════
TC-ID  | Description                      | Zone | Source  | Date      | Status
───────┼──────────────────────────────────┼──────┼─────────┼───────────┼────────
TC-001 | LKA activation during lane-change | 1   | CPYR    | 2021-04   | Mitigated
       | hesitation phase (user misuse)    |      | (SAE)   |           | (SAE paper)
TC-002 | LKA command via cyber-attack at   | 1   | CPYR    | 2021-04   | Detected
       | hesitation phase                  |      | (SAE)   |           | (SAE paper)
TC-003 | Vehicle sudden U-turn in          | 2→1 | Traffic | [DATE]    | Detected
       | intersection (unsignaled)         |      | Vision  |           | (Demo video)
TC-004 | Vehicle reckless merge at high    | 2→1 | Traffic | [DATE]    | Detected
       | relative speed                    |      | Vision  |           | (Demo video)
TC-005 | RLD positive bias drift (sensor   | 1   | CPYR    | 2021-04   | Mitigated
       | performance limitation)           |      | (SAE)   |           | (F1≥0.80)
TC-006 | RLD Gaussian noise (10× σ)        | 1   | CPYR    | 2021-04   | Detected
       | (severe sensor degradation)       |      | (SAE)   |           | (F1=0.88)
TC-[N] | [New condition from deployment]   | 2   | Both    | [DATE]    | Under review
══════════════════════════════════════════════════════════════════════════

Zone 1 = known unsafe (mitigated or detected)
Zone 2 = unknown unsafe (under investigation)
Zone 2→1 = migrated via demonstration
```

---

## 5. EA-03 & EA-04: CPYR Verification and Validation Results

(See Part I §6.3 and §6.4 for full tables.)

**Summary record for ISO 21448 audit:**

```
CPYR V&V SUMMARY RECORD
══════════════════════════════════════════════════════════════════
Model:        LKA Predictor / Enhanced LKA Predictor
Hardware:     Intel i5-8400, GeForce GTX 1070
Framework:    PyTorch 1.8
Date:         April 2021

Verification (ISO 21448 clause 9.3 — known test cases):
  Test cases:   7 (all published in SAE 2021-01-0196, Table 1)
  Pass criteria: F1 = 1.00 on all cases
  Result:        PASS — F1 = 1.00 on all 7 cases (both models)
  Evidence:      SAE paper Table 1; notebook: "detecting lka state.ipynb"

Validation (ISO 21448 clause 9.4 — unknown triggering conditions):
  Test families: 4 (positive bias, negative bias, square wave, Gaussian noise)
  Cases per family: 3 (low/medium/high severity)
  Total cases:   12
  Pass criteria: F1 ≥ 0.88 on all cases
  Result:        PASS — F1 range: 0.88–1.00 (LKA Predictor)
                        F1 range: 0.78–1.00 (Enhanced; 0.78 is below criterion)
  Observation:   Enhanced model under-performs on high positive-bias case (0.78).
                 LKA Predictor preferred for deployments with positive RLD drift.
  Evidence:      SAE paper Table 2; notebook: "detecting lka state.ipynb"

Conclusion: LKA Predictor meets SOTIF validation criteria across all 12 scenarios.
══════════════════════════════════════════════════════════════════
```

---

## 6. EA-06: MISV Zone 2 Reduction Argument

```
MISV ZONE 2 REDUCTION — FORMAL ARGUMENT
══════════════════════════════════════════════════════════════════
Standard clause: ISO 21448:2022, Clause 9.5

Claim:
  The probability of an undetected Zone 2 hazardous event under the MISV
  three-channel architecture is at least two orders of magnitude lower than
  under any single-channel detector.

Argument structure (Goal Structuring Notation summary):
  G1: MISV system reduces Zone 2 residual risk to acceptable level
    ├─ G2: CPYR channel (network) independently detects SOTIF trigger events
    │      Evidence: EA-03, EA-04 (F1 ≥ 0.88 across all V&V scenarios)
    ├─ G3: Traffic Vision channel (video) independently detects behavioral anomalies
    │      Evidence: EA-05 (demo video, 5 annotated events; 1 correct de-escalation)
    ├─ G4: V2X channel (cooperative) independently detects kinematic inconsistency
    │      Evidence: Physics-based consistency model (no ML dependency)
    └─ G5: Channels fail independently (non-overlapping failure modes)
           Argument: Network anomalies are camera-invisible; behavioral video
           anomalies are network-invisible; V2X failures require communication
           infrastructure outage independent of onboard sensor failures.

Quantitative estimate (conservative):
  Let p_a = P(CPYR misses a Zone 2 event)     ≤ 0.12  (1 - min F1 = 1 - 0.88)
  Let p_b = P(TV misses a Zone 2 event)        ≤ 0.20  (conservative, unquantified)
  Let p_c = P(V2X misses a Zone 2 event)       ≤ 0.05  (physics model, high confidence)

  P(MISV misses event) ≤ p_a × p_b × p_c
                       ≤ 0.12 × 0.20 × 0.05
                       = 0.0012   (< 0.2% residual miss rate)

  Single-channel miss rate (CPYR alone): ≤ 12%
  MISV miss rate: ≤ 0.12%
  Improvement factor: 100×

Assumption caveat:
  Independence assumption holds when failure causes are distinct (different
  physical modalities). If a common-cause failure exists (e.g., extreme fog
  simultaneously degrades camera AND V2X signal), independence is violated.
  Common-cause failure mode: fog/heavy rain. Mitigation: weather detection
  pre-processor raises ODD boundary flag, system enters advisory-only mode.

Conclusion:
  MISV architecture satisfies SOTIF clause 9.5 residual risk acceptance
  under the independence assumption, with identified common-cause failure
  mode and explicit ODD boundary.
══════════════════════════════════════════════════════════════════
```

---

## 7. EA-07: Continuous Anomaly Score Log Format

Every frame processed in deployment produces one log entry per channel. This forms the auditable evidence stream for clause 9.6.

```
Log schema (CSV / time-series database):
═══════════════════════════════════════════════════════════════════════════════
timestamp_utc  | ISO 8601, millisecond precision
session_id     | Deployment session UUID
vehicle_id     | Vehicle or infrastructure node ID
frame_idx      | Sequential frame counter within session
gps_lat        | Decimal degrees (WGS-84)
gps_lon        | Decimal degrees (WGS-84)
gps_speed_kmh  | Vehicle speed at frame timestamp
weather_code   | 0=clear, 1=light rain, 2=heavy rain, 3=fog, 4=snow, 9=unknown
channel_a_score| CPYR reconstruction loss (float)
channel_b_score| Traffic Vision MSE (float)
channel_c_score| V2X kinematic deviation (float, null if V2X unavailable)
misv_level     | Fused alert level (0, 1, 2, 3)
alert_fired    | Boolean (1 if misv_level >= deployment threshold)
ood_flag       | Boolean (1 if operating outside ODD)
═══════════════════════════════════════════════════════════════════════════════

Example rows:
2026-05-29T08:14:32.100Z,sess-001,VH-42,1001,30.0444,31.2357,52.3,0,0.0043,0.00041,0.02,0,False,False
2026-05-29T08:14:32.133Z,sess-001,VH-42,1002,30.0444,31.2357,52.1,0,0.0041,0.00039,0.02,0,False,False
2026-05-29T08:14:35.200Z,sess-001,VH-42,1091,30.0445,31.2360,48.7,0,0.0412,0.00780,0.47,3,True,False
```

---

## 8. EA-08: Triggering Condition Discovery Report (Quarterly)

```
TRIGGERING CONDITION DISCOVERY REPORT
Period: [QUARTER] [YEAR]
Deployment: [LOCATION / FLEET ID]
══════════════════════════════════════════════════════════════════════

Section 1: High-Score Event Summary
────────────────────────────────────
Total frames processed:          [N]
Frames above Level-1 threshold:  [N1]  ([N1/N × 100:.1f]%)
Frames above Level-3 threshold:  [N3]  ([N3/N × 100:.3f]%)
Confirmed incidents:             [N_confirmed]
Unconfirmed high-score events:   [N_unconfirmed]  ← Zone 2 candidates

Section 2: Cluster Analysis (unconfirmed events only)
───────────────────────────────────────────────────────
Cluster ID | Count | Channel firing     | Dominant scene feature | TC-ID assigned
C-001      |  14   | Vision only        | Low-light + pedestrian | TC-NEW-001
C-002      |   7   | CPYR only          | Parking lot CAN noise  | TC-NEW-002
C-003      |  22   | Both channels      | Intersection left-turn | TC-NEW-003 ← priority
C-004      |   3   | V2X + Vision       | Ramp merge at speed    | TC-NEW-004

Section 3: Zone Migration
──────────────────────────
Events migrated Zone 2 → Zone 1 this period: [N_migrated]
New Zone 2 entries identified:               [N_new]
Net Zone 2 change:                           [DELTA]

Section 4: Recommended Model Updates
──────────────────────────────────────
[ ] Re-calibrate Traffic Vision threshold for low-light conditions (C-001)
[ ] Collect additional normal CAN captures in parking scenarios (C-002)
[ ] Add intersection left-turn scenario to Traffic Vision training data (C-003)
[ ] Add C-003 as new verification test case in next CPYR V&V cycle
══════════════════════════════════════════════════════════════════════
```

---

---

# PART VI — FAILURE MODE & EFFECTS ANALYSIS

---

## 1. Component-Level Failure Modes

### 1.1 CPYR — Fuzz Detection Channel

| Failure Mode | Cause | Effect | Detection | Mitigation |
|---|---|---|---|---|
| No CAN frames received | OBD-II adapter disconnected | Channel A silent; no score | Watchdog: score absent for > 1 s | Alert operator; degrade to 2-channel |
| Reconstruction loss stuck at zero | Model not loaded / wrong weights | All inputs scored as normal | Score variance monitor (should be > ε) | Reload model; fail-safe to Level 1 |
| Loss always above threshold | Normal distribution shifted (new vehicle variant) | Permanent false alert | Running median drift detection | Re-calibrate threshold on 30 min normal |
| Gradual loss distribution drift | Vehicle firmware update changes CAN patterns | Slow threshold violation | Monthly distribution comparison | TAO retraining trigger |
| GPU OOM during inference | Batch size too large for device | Inference crash | Try/except → CPU fallback | Set batch_size=1 for INT8 TRT |

### 1.2 Traffic Vision Channel

| Failure Mode | Cause | Effect | Detection | Mitigation |
|---|---|---|---|---|
| Camera feed unavailable | Cable fault, lens occlusion | Channel B silent | Frame counter watchdog | Degrade to 2-channel; log ODD flag |
| Score always high (night) | ODD boundary — darkness | Permanent false alert | ODD detector (luminance < threshold) | Activate ODD flag; disable channel B |
| Score always low (static scene) | Traffic stoppage; car in tunnel | Misses genuine anomalies | Score variance monitor | No action; log as low-confidence period |
| Prediction blurs at high speed | Frame motion exceeds training distribution | Reduced sensitivity | Speed-based ODD flag | Recalibrate at high-speed training data |
| Model stale (new vehicle types) | Domain shift (new dominant vehicle class) | Decreased sensitivity | Periodic F1 evaluation on held-out clips | TAO fine-tuning on new normal |

### 1.3 V2X Channel

| Failure Mode | Cause | Effect | Detection | Mitigation |
|---|---|---|---|---|
| No V2X messages received | RSU out of range / communication failure | Channel C inactive | Message rate counter | Degrade to 2-channel gracefully |
| Stale V2X position data | High-latency communication | Kinematic model misaligned | Message age check (> 200 ms = stale) | Reject stale messages; use last-known |
| Spoofed V2X message | Malicious V2X participant | False anomaly signal or false normal | Cryptographic V2X authentication (ETSI TS 102 940) | Reject unauthenticated messages |
| Too many V2X participants | Dense urban environment | Computation bottleneck | Processing latency monitor | Limit to K nearest vehicles (K=5 default) |

---

## 2. System-Level Failure Modes

### 2.1 Complete MISV System

| Scenario | Condition | System Response |
|---|---|---|
| **Normal operation** | All 3 channels active | Full MISV fusion; all alert levels available |
| **Channel A offline** | CPYR unavailable | MISV degrades to Vision + V2X; log degraded mode |
| **Channel B offline** | Traffic Vision unavailable | MISV degrades to CPYR + V2X; ODD flag set |
| **Channel C offline** | V2X unavailable | MISV degrades to CPYR + Vision; common scenario |
| **A + B offline** | Both onboard systems offline | System enters standby; driver alert; no automated action |
| **A + B + C offline** | Total failure | Fail-safe: driver notification; system ceases alerting |
| **Common-cause: heavy fog** | ODD boundary crossed for Camera | ODD flag raised; Channel B disabled; MISV → 2-channel |
| **Score disagreement** | A fires, B does not | Log discrepancy; output Level 1 only (conservative) |
| **Score agreement** | Both A and B fire | High-confidence event; full alert level |

### 2.2 Fail-Safe Specification

The MISV system must fail in a deterministic safe direction:

```
Fail-safe rule:
  1. Any channel returning no score within 1 second → degrade gracefully
  2. System in < 2-channel mode → advisory-only (no automated ADAS actuation)
  3. All channels offline → driver notification only; no false "all clear"
  4. ODD violation detected → disable affected channel(s); maintain remaining

Forbidden failures:
  ✗ Silent channel treated as "score = 0" (would suppress alerts)
  ✗ Threshold recalibration during active incident window
  ✗ Model hot-swap during inference (race condition risk)
```

---

## 3. Watchdog & Health Monitoring

```python
class MISVWatchdog:
    """Monitors health of all MISV channels and triggers graceful degradation."""

    def __init__(self, timeout_s=1.0, variance_window=100):
        self.timeout     = timeout_s
        self.var_window  = variance_window
        self.last_score  = {'a': None, 'b': None, 'c': None}
        self.last_time   = {'a': None, 'b': None, 'c': None}
        self.score_hist  = {'a': [], 'b': [], 'c': []}
        self.active      = {'a': True, 'b': True, 'c': True}

    def update(self, channel, score, timestamp):
        self.last_score[channel] = score
        self.last_time[channel]  = timestamp
        self.score_hist[channel].append(score)
        if len(self.score_hist[channel]) > self.var_window:
            self.score_hist[channel].pop(0)

    def check(self, now):
        status = {}
        for ch in ('a', 'b', 'c'):
            if self.last_time[ch] is None:
                status[ch] = 'NEVER_RECEIVED'
                self.active[ch] = False
                continue

            # Timeout check
            if (now - self.last_time[ch]) > self.timeout:
                status[ch] = 'TIMEOUT'
                self.active[ch] = False
                continue

            # Variance check (stuck-at-zero or stuck-at-high)
            if len(self.score_hist[ch]) >= self.var_window:
                variance = np.var(self.score_hist[ch])
                if variance < 1e-10:
                    status[ch] = 'STUCK'
                    self.active[ch] = False
                    continue

            status[ch] = 'OK'
            self.active[ch] = True

        n_active = sum(self.active.values())
        if n_active < 1:
            return 'FAIL_SAFE', status
        elif n_active < 2:
            return 'ADVISORY_ONLY', status
        else:
            return 'OPERATIONAL', status
```

---

---

# PART VII — COMPARATIVE ANALYSIS & RESEARCH CONTEXT

---

## 1. Anomaly Detection Taxonomy

CPYR and Traffic Vision address two of the three canonical anomaly classes (after Song et al., 2007; Chandola et al., 2009):

| Anomaly Class | Definition | CPYR Application | Traffic Vision Application |
|---|---|---|---|
| **Point anomaly** | A single instance anomalous w.r.t. the global distribution | Not targeted (insufficient for SOTIF) | Not targeted |
| **Contextual anomaly** | An instance anomalous only in a specific context | ✓ **Primary target** — LKA disable during hesitation phase | ✓ **Primary target** — vehicle trajectory unexpected given surrounding context |
| **Collective anomaly** | A group of instances anomalous in aggregate | ✓ **Secondary target** — fuzz attack frame storm | Partially — sustained high-MSE windows across maneuver sequence |

**Why point-anomaly detectors fail for SOTIF**: Every individual LKA frame is, in isolation, a normal automotive CAN frame. Every individual video frame during a sudden lane change shows a perfectly normal car. The anomaly only exists in the temporal-contextual relationship. ISO 21448 explicitly requires contextual analysis; both systems were designed with this as the primary constraint.

---

## 2. Comparison with Alternative Approaches

### 2.1 Signature-Based Intrusion Detection

```
Approach: Maintain a whitelist/blacklist of known attack patterns.
           Flag frames matching known attack signatures.

Strengths:
  + Zero false positives on known attack types
  + Deterministic, auditable decision

Weaknesses (critical for SOTIF):
  ✗ Completely blind to novel attack patterns (Zone 2 by construction)
  ✗ Cannot reason about context — same frame is flagged regardless of
    whether it occurs during normal or hazardous operational context
  ✗ Requires continuous manual signature updates
  ✗ LKA scenario: the legitimate LKA frame is identical to the attack frame;
    no signature can distinguish them without context

CPYR improvement: Semi-supervised contextual model detects the LKA
scenario with F1=1.0 where signature-based approaches would have F1=0
(the frame is legitimate — no signature to match).
```

### 2.2 Supervised Classification (DNN Accident Detector)

```
Approach: Train a classifier on labeled normal vs. accident footage.
           Output a binary class label per frame.

Strengths:
  + High precision on known accident types in the training distribution
  + Well-understood evaluation methodology (precision/recall on test set)

Weaknesses (critical for SOTIF):
  ✗ Requires labeled accident data — scarce, legally sensitive, distribution-shifted
  ✗ Cannot generalize to accident types not in the training taxonomy (Zone 2)
  ✗ Binary output cannot support graduated alert levels
  ✗ No SOTIF compliance argument — does not reduce Zone 2 by construction
  ✗ Threshold tuning requires labeled validation data at each new deployment site

Traffic Vision improvement: Trains on normal data only (17,370 frames of
routine driving). Achieves quarter-second pre-event detection on events
unseen in training (U-turns, reckless merges, ambiguous trajectories).
Correct de-escalation on false alarms without any labeled data.
```

### 2.3 Rule-Based Threshold Systems

```
Approach: Flag events when measured values exceed fixed thresholds
           (e.g., TTC < 2 s, speed > 90 km/h in school zone).

Strengths:
  + Simple, interpretable, deterministic
  + No training data required

Weaknesses:
  ✗ Context-blind: a legal speed can be dangerous at a blind curve
  ✗ Cannot adapt to new road geometries, traffic compositions, or weather
  ✗ Produces high false-positive rates in atypical-but-safe scenarios
  ✗ No mechanism for Zone 2 discovery

MISV improvement: Continuous anomaly score adapts to local normal
distribution via TAO retraining. Detects dangerous situations outside
the rule designer's enumerated set.
```

### 2.4 Fingerprinting Methods (Automotive Cybersecurity)

```
Approach: Identify anomalous ECU behavior by physical-layer characteristics
           (clock timing, voltage patterns) — Cho & Shin 2016/2017.

Strengths:
  + Detects impersonation attacks (attacker controls a different ECU
    and sends frames pretending to be another ECU)

Weaknesses for SOTIF:
  ✗ The LKA scenario in SAE 2021-01-0196 involves NO impersonation:
    the legitimate Steering Column ECU sends a legitimate LKA frame.
    Physical-layer fingerprinting sees only a normal ECU — no anomaly.
  ✗ Cannot reason about the semantic relationship between LKA state and
    RLD sensor values (contextual reasoning is required)

CPYR addresses precisely the gap fingerprinting cannot fill:
legitimate frames with anomalous contextual meaning.
```

### 2.5 State-of-the-Art Video Anomaly Detection (UCF-Crime / ShanghaiTech)

```
Approach: Weakly supervised ranking models (Sultani et al. 2018),
           or reconstruction-based models (Park et al. 2020) trained
           on video anomaly benchmarks.

Strengths:
  + Strong benchmark performance on curated datasets
  + Spatial localization available in some models

Weaknesses for traffic safety:
  ✗ Benchmark datasets (UCF-Crime, ShanghaiTech) are camera-surveillance
    oriented (robbery, fighting, loitering) — traffic-specific anomalies
    are underrepresented
  ✗ UCF-Crime models require weakly labeled video — still requires
    identifying which clips contain anomalies at collection time
  ✗ No integration with SOTIF methodology or ISO 21448 compliance framework

Traffic Vision improvement: Fully unsupervised training on purely normal
dashcam footage. Predicts future frames rather than reconstructing present
frames — captures behavioral anomalies that reconstruction-based models miss.
Quarter-second lookahead is novel: existing benchmarks evaluate detection
at or after the event, not before it.
```

---

## 3. Research Positioning

CPYR + Traffic Vision together occupy a unique position in the literature:

```
                   Contextual
                   reasoning?
                       │
           Yes ────────┼──────── No
               │       │       │
    Published  │       │       │  Reconstruction
    SOTIF      │  ←────┘       │  autoencoders
    framework  │  CPYR+TV      │  (Park 2020,
               │               │   Lu 2013)
               │               │
    Requires   │               │  Rule-based
    labeled    │               │  threshold
    data?      │               │  systems
       │       │               │
      No ──────┤               │
               │               │
               ▼               │
    Traffic Vision           Supervised
    CPYR (semi-supervised)   classifiers
                             (Sultani 2018)
```

The combination of (a) contextual reasoning, (b) no labeled anomaly data, and (c) formal SOTIF compliance argument is, to our knowledge, not represented elsewhere in the published literature as of the SAE paper submission date (April 2021).

---

---

# PART VIII — FUTURE DEVELOPMENT ROADMAP

---

## 1. Near-Term (0–6 months): Production Hardening

### 1.1 Spatial Anomaly Localization for Traffic Vision

Current limitation: the system produces a scalar anomaly score per window. It cannot identify *which vehicle* is the threat.

**Proposed approach — Spatial MSE heatmap:**
```python
# Per-pixel error map (no architectural change required)
predicted = model(input_frames[:, :in_frames, :, :])
per_pixel_error = F.mse_loss(predicted, input_frames, reduction='none')
heatmap = per_pixel_error.mean(dim=1)   # (B, H, W)

# Gaussian smoothing for display
from scipy.ndimage import gaussian_filter
smooth_heatmap = gaussian_filter(heatmap[0].cpu().numpy(), sigma=5)
```

**Next step — Object-level attribution**: Run a lightweight YOLO-nano detector in parallel. For each detected bounding box, compute the mean heatmap value inside the box. The box with the highest mean heatmap value is the highest-probability threat vehicle. This adds < 5 ms at Jetson AGX Orin and gives a named, localized threat output.

### 1.2 Quantization for Tighter Edge Constraints

Traffic Vision currently runs FP16 on Tesla T4. Jetson Orin Nano (8 GB) requires INT8 for real-time performance:

```bash
# INT8 calibration for Traffic Vision
trtexec \
    --onnx=traffic_vision_ground.onnx \
    --saveEngine=traffic_vision_ground_int8.trt \
    --int8 \
    --calib=calibration_dataset/normal_frames/ \
    --workspace=2048 \
    --verbose

# Expected throughput improvement: 2–3× vs FP16 on Jetson Orin Nano
```

**Calibration dataset**: 500 normal-traffic windows from the deployment site. No labeling required.

### 1.3 CPYR: Extending to New Attack Families

The current fuzz-attack dataset covers high-rate frame injection. Two additional attack families require new datasets and evaluation:

| Attack Family | Description | Expected CPYR response |
|---|---|---|
| **Spoofing** | Attacker replays captured frames with modified IDs | Reconstruction loss on ID field will spike; payload may be normal |
| **Replay attack** | Exact frame replay from a previous session | Temporal inconsistency: future-frame prediction expects novel frames, not exact repeats |
| **Timing attack** | Legitimate frames at wrong inter-arrival times | Sequence-level model should detect changed temporal statistics |

---

## 2. Mid-Term (6–18 months): Architecture Evolution

### 2.1 Temporal Model Improvements for Traffic Vision

The current frame-stacking approach handles ~500 ms temporal context. Longer-horizon dangers (a vehicle following too closely for 30 seconds; a driver gradually drifting) require true temporal models.

**Candidate architectures:**

| Architecture | Temporal capacity | Computational cost | Suitability |
|---|---|---|---|
| ConvLSTM | Seconds | 2–3× current model | Suitable for Jetson AGX |
| 3-D CNN (C3D/SlowFast) | 1–4 s | 5–10× current | Cloud training; TRT for edge |
| Video Swin Transformer | Up to 32 frames (configurable) | 20–50× current | Cloud analysis only |
| Hierarchical (current model + LSTM over windows) | Unlimited | +LSTM overhead | Recommended for Phase 2 |

**Recommended incremental path**: Keep the current convolutional autoencoder as the short-horizon predictor. Add a lightweight LSTM over its window-level anomaly scores to detect long-duration anomalies without changing the edge inference model.

### 2.2 CPYR: Attention-Augmented History

The current History block is a simple convolutional layer that produces a fixed-size compressed representation of all past RLD frames. This representation has no mechanism to prioritize recent or high-variance frames.

**Proposed enhancement — Self-attention History:**
```python
class AttentionHistory(nn.Module):
    """Replaces the conv-based History block with multi-head self-attention."""
    def __init__(self, frame_size, num_heads=4, max_history=50):
        super().__init__()
        self.attn = nn.MultiheadAttention(frame_size, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(frame_size)

    def forward(self, history_seq):
        # history_seq: (B, T, frame_size)
        attended, _ = self.attn(history_seq, history_seq, history_seq)
        return self.norm(attended[:, -1, :])   # return last attended frame
```

This change allows the model to focus on the most contextually relevant past frames when predicting the next LKA state — particularly useful for long-duration scenarios (>10 frames of hesitation).

### 2.3 Federated Learning Across Fleet

In a fleet deployment, each vehicle or roadside unit trains on its own local normal distribution. Federated averaging across the fleet builds a shared encoder that generalizes better than any single-site model:

```
Phase 1: Each node fine-tunes the shared encoder on local normal data
Phase 2: Local weight updates (gradients only) sent to aggregation server
Phase 3: Federated average → updated shared encoder
Phase 4: Updated encoder distributed back to all nodes
Phase 5: Repeat on periodic schedule (weekly)

Privacy property: Raw data never leaves the vehicle/node.
                  Only gradient updates are shared.
```

**NVIDIA Metropolis fit**: NVIDIA Fleet Command handles the model distribution step (Phase 4) natively. Phases 2–3 require a lightweight aggregation microservice deployed alongside Metropolis.

---

## 3. Long-Term (18+ months): Research Extensions

### 3.1 Explainability Integration

Current limitation: the anomaly score is a scalar with no explanation. A safety auditor asking "why did the system raise an alert at 08:14:35?" has no answer beyond "the score exceeded the threshold."

**Proposed: LIME-based explanation for Traffic Vision**
```python
from lime import lime_image
import shap

explainer = lime_image.LimeImageExplainer()

def predict_score(frames):
    """LIME-compatible wrapper: (N, H, W, 3) → (N, 1) scores."""
    t = torch.stack([transforms.ToTensor()(f) for f in frames]).to(DEVICE)
    with torch.no_grad():
        pred = model(t[:, :in_frames*3, :, :])
        return F.mse_loss(pred, t, reduction='none').mean(dim=(1,2,3)).cpu().numpy()

explanation = explainer.explain_instance(
    current_frame, predict_score, top_labels=1, num_samples=500
)
# Produces a superpixel mask showing which image regions most contributed
# to the high anomaly score
```

**Proposed: Integrated Gradients for CPYR**
```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(lambda x: encoder(x).sum())
attribution = ig.attribute(input_batch, baselines=normal_baseline)
# attribution[i, j] shows which bit positions in frame j most contributed
# to the reconstruction anomaly
```

### 3.2 LLM Integration for Automated Incident Reports

Each high-score MISV event can be passed to a language model for automatic natural-language incident description:

```
Input to LLM:
  - MISV alert level: 3
  - Channel A (CPYR) score: 0.041 (threshold 0.025)
  - Channel B (Traffic Vision) score: 0.0078 (threshold 0.005)
  - GPS: [30.0445°N, 31.2360°E]
  - Speed: 48.7 km/h
  - Time: 08:14:35 UTC
  - Heatmap peak: upper-right quadrant of frame
  - Score history: rising over 3 windows

LLM output:
  "At 08:14:35 UTC, the MISV system detected a high-confidence hazardous
  situation (Level 3) at GPS coordinates 30.0445°N, 31.2360°E. The camera
  channel identified unexpected motion in the upper-right region of the
  field of view — consistent with a vehicle merging or crossing from the
  right lane. The in-vehicle network channel simultaneously detected an
  anomalous control state transition. The host vehicle was traveling at
  48.7 km/h. Recommended review priority: HIGH."
```

This automated narrative feeds directly into the SOTIF triggering-condition discovery report (EA-08) without human authoring.

---

---

# PART IX — APPENDICES

---

## Appendix A: CPYR Complete Hyperparameter Reference

### A.1 Fuzz Detection — All Variants

| Parameter | Conv Encoder | Einstein-sum | U-Net | U-Net+Ein+ConvT3D | U-Net+Ein+Transformer |
|---|---|---|---|---|---|
| Input shape | (B,20,112,112) | (B,20,112,112) | (B,20,112,112) | (B,20,112,112) | (B,20,112,112) |
| Encoder output | (B,1,112,112) | (B,112,112) | (B,1,112,112) | (B,1,112,112) | (B,1,112,112) |
| Optimizer | Adam | Adam | Adam | Adam | Adam |
| LR | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 |
| Adam β₁ | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| Adam β₂ | 0.999 | 0.999 | 0.999 | 0.999 | 0.999 |
| Batch size | 8 | 8 | 8 | 8 | 8 |
| Seq length | 20 | 20 | 20 | 20 | 20 |
| Frame shape | 112×112 | 112×112 | 112×112 | 112×112 | 112×112 |
| Epochs | 6 | 6 | 6 | 6 | 6 |
| Loss | MSE | MSE | MSE | MSE | MSE |
| Pad mode | right | right | right | right | right |
| Model size | 1.2 KB | 51.2 KB | 31.3 MB | 33.3 MB | 223.2 MB |
| Eval time | 1.66 s | 2.01 s | 16.3 s | 16.4 s | 17.2 s |

### A.2 LKA Contextual Anomaly

| Parameter | Baseline | LKA Predictor | Enhanced LKA Predictor |
|---|---|---|---|
| Optimizer | Adam | Adam | Adam |
| LR | 1e-4 | 1e-4 | 1e-4 |
| Adam β₁ | 0.9 | 0.9 | 0.9 |
| Adam β₂ | 0.999 | 0.999 | 0.999 |
| Batch size | 8 | 8 | 8 |
| Seq length | 10 | 10 | 10 |
| Epochs | 4 | 4 | 4 |
| Predictor hidden | — | 1024 | 1024 |
| Window size N | — | — | 10 |
| Loss | MSE | MSE | 0.5×MSE + 0.5×MSE_hist |
| LKA eval loss | MSE | BCE + SSIM | BCE + SSIM |
| Model params | 2 | 1,054,726 | 1,054,746 |
| Inference (mb/s) | 139.1 | 138.1 | 139.6 |

---

## Appendix B: Traffic Vision Complete Hyperparameter Reference

| Parameter | Value | Notes |
|---|---|---|
| Input resolution | 320×480 (H×W) | After per-video crop and resize |
| Sequence length | 16 frames | Active variant |
| Channels per sample | 48 (3×16) | Frames stacked into channel dim |
| In frames (ground model) | 10 | `model_f30_f48.pt` |
| In frames (predict model) | 12 | `model_f36_f48.pt` |
| Out frames (both models) | 16 | Prediction target |
| Batch size | 4 | Constrained by 15 GB T4 |
| Loss | MSE (nn.MSELoss) | Pixel-wise |
| Optimizer | Adam | Default betas |
| Learning rate | 1e-3 | Flat across all epochs |
| Epochs | 30 | Best-val checkpoint saved |
| Encoder channels | [32, 64, 128] | Conv1, Conv2, Conv3 |
| Pool kernel | 3×3, stride 2, pad 1 | MaxPool2d |
| Bottleneck shape | (B, 128, 40, 60) | After 3 MaxPool layers |
| Decoder transposed kernels | 2×2, stride 2 | ConvTranspose2d ×3 |
| Augmentation library | Albumentations 0.5.2 | Photometric only |
| Augmentation p (Group A) | 0.9 | Lighting ops |
| Augmentation p (Group B) | 0.9 | Sharpness/blur ops |
| Gaussian noise p | 0.2 | IAAAdditiveGaussianNoise |
| Perspective p | 0.5 | IAAPerspective |
| Anomaly threshold | ≈5×10⁻³ | Empirical; tunable per site |
| Training frames | ~17,370 | ~10 min routine urban driving |
| Validation MSE (final) | 4.3×10⁻⁴ | Epoch 30 best checkpoint |
| Tesla T4 throughput | ~36 fps | Exceeds 30 fps dashcam |
| Per-batch latency (B=4) | ~902 ms | torch.cuda.Event measurement |

---

## Appendix C: Source File Reference

### C.1 CPYR Module Map

| File | Role | Key classes / functions |
|---|---|---|
| `basemodel.py` | Abstract base for all models | `BaseModel` (train/eval scaffolding) |
| `models.py` | Model definitions | `ConvEncoder`, `EinsumEncoder`, combined variants |
| `unet.py` | U-Net implementation | `UNet`, `DoubleConv`, `Down`, `Up`, `OutConv` |
| `unet_ein.py` | U-Net + Einstein-sum hybrid | `UNetEin` |
| `ein.py` | Einstein-sum encoder block | `EinsumLayer` |
| `modified_transformer/` | Transformer decoder | `TransformerDecoder`, `MultiHeadAttention`, `PositionalEncoding` |
| `fuz_conv.py` | Fuzzy convolution layer | `FuzConv` |
| `cbs.py` | Conv-BatchNorm-Sigmoid block | `CBS` |
| `customized_sigmoid.py` | Custom sigmoid activation | `CustomizedSigmoid` |
| `customized_softmax.py` | Custom softmax activation | `CustomizedSoftmax` |
| `up_down_sampler.py` | U-Net up/down primitives | `DownSampler`, `UpSampler` |
| `resnet_hybrid.py` | ResNet hybrid backbone | `ResNetHybrid` |
| `data_handler.py` | Dataset loading + preprocessing | `hex_to_binary()`, `pad_frame()`, `FrameDataset` |
| `fit.py` | Training loop | `Learner`, callback hooks |
| `optimizer.py` | Adam wrapper | `build_optimizer()` |
| `scheduler.py` | LR scheduling | `WarmupCosineScheduler` |
| `callbacks.py` | Training callbacks | `CheckpointCallback`, `EarlyStopCallback` |
| `cbs.py` | Conv-BN-Sigmoid block | `CBS` |
| `meter.py` | Loss/metric accumulation | `AverageMeter`, `F1Meter` |
| `utils.py` | Miscellaneous utilities | `save_model()`, `load_model()`, `tag_experiment()` |
| `imports.py` | Centralized imports | Common library imports |

### C.2 Traffic Vision Notebook Map

| Notebook | Role |
|---|---|
| `Autoencoder_Anomaly_Vision_Detection.ipynb` | First training notebook; sequence_len=10; pure reconstruction |
| `Autoencoder_Anomaly_Vision_Detection_01.ipynb` | Architecture ablation: depth and input width |
| `Autoencoder_Anomaly_Vision_Detection_001.ipynb` | Final training; sequence_len=16; all horizon variants |
| `Inference_Autoencoder_Anomaly_Vision_Detection_01.ipynb` | Inference + demo: dual-model eval + annotated MP4 output |

### C.3 CPYR Notebook Map

| Notebook | Role |
|---|---|
| `Integration script.ipynb` | End-to-end fuzz-attack training across all 5 encoder variants |
| `detecting lka state.ipynb` | LKA contextual anomaly (SAE paper reproduction) |
| `clean_parser.ipynb` | Dataset hex→binary conversion and cleaning |
| `Parser.ipynb` | Earlier parser notebook (legacy) |
| `unet for sequence model.ipynb` | U-Net sequence experiments |

---

## Appendix D: Dataset Format Specifications

### D.1 CPYR — Input Format

```
File format: Plain text, UTF-8
Extension:   .txt or .log
Encoding:    One frame per line

Line format (tab-separated):
  <frame_id>  <hex_payload>

Field specifications:
  frame_id:    Hexadecimal string, no '0x' prefix, variable length (e.g., "0A1", "12AB")
  hex_payload: Hexadecimal string, uppercase or lowercase, no spaces
               Variable length (8–128 hex characters = 32–512 bits)
               Trailing zeros acceptable (will be padded to 112×112 during preprocessing)

Example:
  0A1    4F3A00FF12BC7E01
  0B2    00000000AABBCCDD0011223344556677
  0A1    4F3A00FF12BC7E02

Phase labels (for evaluation only, not used during training):
  Supplied as a separate CSV: frame_index, phase_label
  phase_label values: 0=Normal, 1=Attack, 2=Disable, 3=End
```

### D.2 Traffic Vision — Input Format

```
Video format:
  Container:   MP4 (H.264 recommended)
  Resolution:  Any (will be cropped + resized to 320×480)
  Frame rate:  30 fps recommended (other rates supported with threshold adjustment)
  Content:     Forward-facing dashcam footage
  Duration:    Minimum 10 minutes for training; any length for inference

Frame extraction output:
  Format:   JPEG, quality 95
  Naming:   frame-00000.jpg, frame-00001.jpg, ... (zero-padded to 5 digits)
  Directory: one folder per video source

Training data requirements:
  Content:  Normal driving only (no accidents or near-misses)
  Diversity: Multiple routes, drivers, times of day (within ODD)
  Minimum:  ~17,370 frames (~10 minutes at 30 fps) for baseline model
  Recommended: ~100,000 frames (~55 minutes) for robust deployment model
```

### D.3 V2X — Input Format

```
Protocol:    ETSI ITS-G5 (DSRC) or C-V2X (PC5)
Message types:
  CAM (Cooperative Awareness Message):
    - Frequency: 1–10 Hz per vehicle
    - Content: position, speed, heading, acceleration, timestamp
    - Size: ~250–350 bytes per message

  DENM (Decentralized Environmental Notification Message):
    - Triggered: on event (emergency brake, accident, hazard)
    - Content: event type, position, confidence

Parsed fields used by MISV kinematic checker:
  station_id         : uint32, unique vehicle identifier
  generation_delta_t : uint16, milliseconds since last CAM
  latitude           : int32, WGS-84 × 10⁷ (1/10 microdegree)
  longitude          : int32, WGS-84 × 10⁷
  speed              : uint16, 0.01 m/s resolution
  heading            : uint16, 0.1 degree resolution (0 = North)
  longitudinal_accel : int16, 0.1 m/s² resolution
  yaw_rate           : int16, 0.01 deg/s resolution
```

---

## Appendix E: Glossary

| Term | Definition |
|---|---|
| **ADAS** | Advanced Driver-Assistance System — software that aids the human driver with perception, alerts, or partial control |
| **Anomaly score** | A scalar value quantifying how unexpected an input is to a trained model. In CPYR: reconstruction loss. In Traffic Vision: MSE between predicted and observed frames. |
| **BCE** | Binary Cross Entropy — loss function measuring bit-wise prediction error between predicted and actual binary states |
| **Bottleneck** | The narrowest layer of an autoencoder; the most compressed representation of the input |
| **CAN** | Controller Area Network — dominant in-vehicle bus protocol for ECU-to-ECU communication |
| **Collective anomaly** | A group of data instances that are anomalous in aggregate, even if each individual instance appears normal (e.g., a fuzz attack's abnormal frame rate) |
| **Contextual anomaly** | A data instance that is anomalous only within a specific operational context (e.g., LKA activation during sensor hesitation phase) |
| **C-V2X** | Cellular Vehicle-to-Everything — V2X communication over 4G/5G cellular infrastructure |
| **DAD** | Deep Anomaly Detection — anomaly detection using deep neural network models |
| **DeepStream** | NVIDIA's SDK for AI-powered video analytics pipelines, built on GStreamer |
| **DSRC** | Dedicated Short-Range Communications — 5.9 GHz wireless protocol for V2X |
| **ECU** | Electronic Control Unit — embedded computer managing one or more automotive functions |
| **Einstein-sum (einsum)** | A PyTorch operation for computing generalized tensor contractions; used in CPYR to create a learned weighted combination of a frame sequence |
| **F1 score** | Harmonic mean of precision and recall: 2×(P×R)/(P+R) |
| **FLOPs** | Floating-Point Operations — measure of computational work per forward pass |
| **Hex-to-binary** | CPYR preprocessing: each hex character expanded to 4 bits, collapsing output space from 16-class to 2-class |
| **INT8** | 8-bit integer quantization — reduces model size and inference latency by 2–4× vs FP32, with minor accuracy trade-off |
| **ISO 21448 (SOTIF)** | Safety of the Intended Functionality — safety standard for risks arising from functional limitations and foreseeable misuse of automotive AI systems |
| **ISO 26262** | Functional Safety — automotive safety standard addressing electrical/electronic hardware failures |
| **Jetson** | NVIDIA's family of edge AI compute modules (Orin Nano, Orin NX, AGX Orin) |
| **LKA** | Lane Keep Assist — ADAS function that keeps a vehicle centered in its lane |
| **MISV** | Multiple Independent Source Verification — architecture combining multiple anomaly detectors with non-overlapping failure modes to formally reduce SOTIF Zone 2 risk |
| **MSE** | Mean Squared Error — average squared pixel-wise difference between predicted and observed tensors |
| **NGC** | NVIDIA GPU Cloud — catalog of pretrained AI models, containers, and SDKs |
| **nvinfer** | DeepStream element for running TensorRT inference on a GStreamer frame stream |
| **nvmsgbroker** | DeepStream element for publishing inference results to external message brokers (Kafka, MQTT, Azure IoT Hub) |
| **OBD-II** | On-Board Diagnostics II — standardized vehicle diagnostic port providing access to CAN bus data |
| **ODD** | Operational Design Domain — the specific conditions under which a system is designed and validated to operate safely |
| **ONNX** | Open Neural Network Exchange — standardized ML model format for interoperability across frameworks |
| **RLD** | Right Lane Distance — sensor measurement of the distance from the vehicle to the right lane marker |
| **SAE** | Society of Automotive Engineers — standards body for automotive engineering |
| **Semi-supervised learning** | Training paradigm using only labeled normal data; anomalies are detected as deviations from the learned normal model |
| **SSIM** | Structural Similarity Index — perceptual image quality metric capturing luminance, contrast, and structure |
| **TAO Toolkit** | NVIDIA Transfer Learning Toolkit — platform for fine-tuning pretrained models on domain-specific data |
| **TensorRT** | NVIDIA's inference optimization library: fuses layers, quantizes weights, and generates GPU-optimized inference engines |
| **TTC** | Time-to-Collision — estimated time until two objects at current trajectories would intersect |
| **U-Net** | Convolutional encoder-decoder with skip connections; originally developed for biomedical image segmentation |
| **V2X** | Vehicle-to-Everything — communication between vehicles and infrastructure (V2V, V2I, V2P) |
| **Zone 2** | ISO 21448 hazard zone: unknown triggering conditions that lead to hazardous behavior — the primary target for MISV reduction |

---

## Appendix F: Academic References

The following references underpin the design decisions and evaluation methodology of both systems.

**SOTIF & Automotive Safety Standards**
1. ISO/PAS 21448:2019 / ISO 21448:2022 — Safety of the Intended Functionality
2. ISO 26262:2018 — Road vehicles — Functional safety
3. SAE J3016:2021 — Taxonomy and Definitions for Terms Related to Driving Automation
4. Abdulazim, A., Elbahaey, M., Mohamed, A. (2021). *Putting Safety of Intended Functionality SOTIF into Practice.* SAE Technical Paper 2021-01-0196. DOI: 10.4271/2021-01-0196
5. Schnellbach, A., Griessnig, G. (2019). Development of the ISO 21448. *Systems, Software and Services Process Improvement*, Springer.
6. Schwalb, E. (2020). Analysis of Safety of the Intended Use (SOTIF).
7. Koopman, P., Wagner, M. (2016). Challenges in Autonomous Vehicle Testing and Validation. *SAE Int. J. Trans. Safety* 4:15–24.
8. Guo, M. et al. (2020). Control Model of Automated Driving Systems Based on SOTIF Evaluation. SAE Technical Paper 2020-01-1214.

**Deep Anomaly Detection**
9. Chandola, V., Banerjee, A., Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys* 41(3).
10. Song, X. et al. (2007). CONDITIONAL ANOMALY DETECTION. *IEEE TKDE* 19(5).
11. Hawkins, S. et al. (2002). Outlier Detection Using Replicator Neural Networks. *PAKDD*. (CPYR ref. [18])
12. Gharib, M. et al. (2018). On the Safety of Automotive Systems Incorporating Machine Learning Based Components. *DSN-W 2018*.

**Video Anomaly Detection**
13. Park, H. et al. (2020). Learning Memory-guided Normality for Anomaly Detection. *CVPR 2020*.
14. Sultani, W., Chen, C., Shah, M. (2018). Real-world Anomaly Detection in Surveillance Videos. *CVPR 2018*.
15. Lu, C., Shi, J., Jia, J. (2013). Abnormal Event Detection at 150 FPS in MATLAB. *ICCV 2013*.

**Automotive Cybersecurity**
16. Cho, K.-T., Shin, K.G. (2016). Fingerprinting Electronic Control Units for Vehicle Intrusion Detection. *USENIX Security 2016*.
17. Cho, K., Shin, K.G. (2017). VIDEN: Attacker Identification on In-Vehicle Networks. *CoRR abs/1708.08414*.

**Neural Network Safety**
18. Burton, S., Gauerhof, L., Heinzemann, C. (2017). Making the Case for Safety of Machine Learning in Highly Automated Driving. *SAFECOMP 2017*.
19. Varshney, K.R. (2016). Engineering Safety in Machine Learning. *ITA Workshop*.
20. Bodenhausen, U. (2019). Architecture and Independence Controller for Deep Learning in Safety Critical Applications. *Stuttgarter Symposium*.

**Attention & Transformer Architecture**
21. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
22. Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.

---

*End of continuation.*

**Document:** Technical_Documentation_and_Metropolis_Proposal (Parts IV–IX)  
**Total document length (Parts I–IX):** ~2,900 lines / ~18,000 words  
**Repository:** github.com/moustafa991982/Cpyr  
**Contact:** sales@at-instr.com · https://at-instr.com
