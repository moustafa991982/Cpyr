# Technical Documentation & NVIDIA Metropolis Integration Proposal
## CPYR · Traffic Vision Hazardous Detection
### Multiple Independent Source Verification for Traffic Safety Under ISO 21448 SOTIF

**Authors:** Moustafa El Bahaey, Amr Abdulazim, Abduallah Mohamed  
**Organization:** AT Instruments / EVRaid  
**Contact:** sales@at-instr.com · https://at-instr.com  
**SAE Reference:** DOI 10.4271/2021-01-0196

---

## Table of Contents

**Part I — CPYR: Deep Anomaly Detection for In-Vehicle Networks**
1. Overview & Design Philosophy
2. SOTIF Methodology
3. Data Pipeline & Preprocessing
4. Model Architecture — Reconstruction-Based Fuzz Detection
5. Model Architecture — Contextual Anomaly Engine (SOTIF)
6. Training, Verification & Validation
7. Results
8. Deployment Characteristics

**Part II — Traffic Vision: Hazardous Situation Detection via Future-Frame Prediction**
1. Overview & Design Philosophy
2. Core Concept: Behavioral Anomaly via Predictive Error
3. System Architecture
4. Dataset & Preprocessing
5. Training Procedure
6. Dual-Model Inference Pipeline
7. Anomaly Scoring & Uncertainty
8. Results & Demonstrated Events
9. Performance Characteristics

**Part III — NVIDIA Metropolis Integration Proposal**
1. The Safety Gap MISV Fills
2. Unified System Architecture
3. CPYR → DeepStream Integration
4. Traffic Vision → DeepStream Integration
5. V2X as Third Channel
6. TAO Toolkit Continuous Learning
7. Edge Deployment on Jetson
8. Smart City Application Scenarios
9. SOTIF Compliance Argument
10. Phased Collaboration Roadmap
11. Business Case Summary

---

---

# PART I — CPYR: Deep Anomaly Detection for In-Vehicle Networks

---

## 1. Overview & Design Philosophy

CPYR is a semi-supervised deep learning framework for detecting anomalies in automotive in-vehicle networks. It was developed as the reference implementation for the SAE WCX 2021 paper *"Putting Safety of Intended Functionality SOTIF into Practice"* (SAE Technical Paper 2021-01-0196, DOI: 10.4271/2021-01-0196).

The project addresses two complementary safety problems that share a single codebase:

**Problem A — Cybersecurity (Collective Anomaly)**  
Malicious actors injecting frames into automotive Ethernet (fuzz attacks, denial-of-service, spoofing). Classical signature-based IDS approaches fail here: they cannot detect novel attack patterns, and they are typically too heavy for ECU-class deployment.

**Problem B — SOTIF (Contextual Anomaly)**  
Hazardous behaviors that emerge from correctly functioning components reacting to operational contexts the system was never designed to handle. The canonical example: Lane Keep Assist (LKA) being activated during a sensor hesitation phase while the driver has released the wheel — each individual action is legal; the combination is a SOTIF trigger event.

**Design principles driving every architectural decision:**

1. **Semi-supervised training**: Models train only on normal operational data. No labeled attack or anomaly samples are needed. The anomaly signal is the delta between expected and observed reconstruction or prediction quality.

2. **Minimize model size without sacrificing separation**: A preprocessing step (hex-to-binary conversion) collapses the reconstruction problem from a 16-class to a 2-class output space. This single decision allows a 1.2 KB model to match the separation quality of a 223 MB transformer.

3. **Contextual reasoning over point-in-time classification**: SOTIF trigger events cannot be detected by inspecting a single frame. The model must understand what a state change means *given* the surrounding sequence of context.

4. **SOTIF-driven verification and validation**: The evaluation methodology follows ISO/PAS 21448 guidelines, including separate verification (known test cases) and validation (unknown distortion families) phases.

---

## 2. SOTIF Methodology

ISO/PAS 21448 (SOTIF) addresses risks due to non-electrical, non-electronic failures — specifically the performance limitations of ML-based systems and foreseeable user misuse. Unlike ISO 26262, which deals with hardware faults, SOTIF targets the *functional* boundary where a correctly operating system produces hazardous outcomes.

**The SOTIF hazard zone model:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Zone 1 — Known Unsafe:   known triggers → hazardous behaviour  │
│  Zone 2 — Unknown Unsafe: unknown triggers → hazardous          │  ← The hard problem
│  Zone 3 — Known Safe:     known triggers → safe behaviour       │
│  Zone 4 — Unknown Safe:   unknown triggers → safe behaviour     │
└─────────────────────────────────────────────────────────────────┘
Goal of SOTIF V&V: shrink Zone 2 toward zero
```

**SOTIF scenario analysis workflow (applied in SAE 2021-01-0196):**

```
Step 1: Determine system specification
        └─► Define intended functionality (LKA: keep vehicle centered in lane)

Step 2: Identify acceptable triggering events
        └─► Normal LKA ON/OFF transitions in constant-phase RLD

Step 3: Evaluate known hazardous scenarios (Verification)
        └─► 7 structured test cases covering all LKA-RLD transition combinations

Step 4: Evaluate hazardous unknown scenarios (Validation)
        └─► 12 test cases with 4 families of sensor distortion
             (positive bias, negative bias, square wave, Gaussian noise)
```

**The safety-critical scenario demonstrated in the paper:**

The scenario involves a Lane Keep Assist (LKA) system and a Right Lane Distance (RLD) sensor on a vehicle using CAN protocol.

```
Time sequence 1 (Scene setup):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ LKA: OFF │ Driver switches lanes │ Car from behind accelerates      │
  │          │ Driver turns LKA ON   │ Driver releases wheel (misuse)   │
  │          │ RLD enters hesitation │                                  │
  └─────────────────────────────────────────────────────────────────────┘

Time sequence 2 (Hazardous event):
  ┌─────────────────────────────────────────────────────────────────────┐
  │ LKA: ON  │ LKA cannot determine correct lane (hesitation phase)    │
  │          │ LKA steers hard back to old lane                        │
  │          │ Collision with approaching vehicle                      │
  │          │ Neither driver can react — shock                        │
  └─────────────────────────────────────────────────────────────────────┘

Trigger event (SOTIF definition):
  LKA state transition to ON while RLD is in switching/steering phase
  ↑ This is the single contextual anomaly CPYR detects at zero-batch lag
```

**Cyber-security overlap**: The same scenario can be triggered by a cyber-attack — an attacker with control over the Steering Column ECU sends a legitimate-looking LKA frame at a critical moment. CPYR detects both origins through the same contextual mechanism, because the trigger event signature is identical regardless of whether the cause is user misuse or malicious injection.

---

## 3. Data Pipeline & Preprocessing

### 3.1 Input: Automotive Ethernet / CAN Frames

Raw input consists of a heterogeneous stream of variable-length Ethernet or CAN frames, each containing a frame ID and a hexadecimal payload.

```
Example raw frame:
  ID: 0x0A1   Data: 4F3A00FF12BC7E01...   Length: 8 bytes
```

### 3.2 Hex-to-Binary Conversion

**This is the single most consequential preprocessing step in CPYR.**

Each hexadecimal character is expanded to its 4-bit binary representation:

```python
# Conceptual mapping (from data_handler.py)
hex_char → 4-bit binary:
  '0' → [0,0,0,0]
  '1' → [0,0,0,1]
  ...
  'F' → [1,1,1,1]
```

**Why this matters for anomaly detection:**

A reconstruction model trained on hex characters must learn to reproduce one of 16 possible symbols per position. With binary expansion, each position has only 2 possible values (0 or 1). The output space collapses from 16-class to binary classification.

Consequences:
- The reconstruction objective becomes tractable for a very small model
- Bit-level statistical deviations are directly exposed in the tensor
- Fuzz attacks, which inject random or patterned bit sequences, create large binary reconstruction errors even with a 1.2 KB encoder
- The approach is modality-agnostic: any fixed-format binary stream can be processed identically

### 3.3 Frame Padding

After hex-to-binary conversion, frames are padded to a uniform 112×112 tensor. The dimensions are chosen to:
- Accommodate the longest observed Ethernet frame with safety margin
- Satisfy the U-Net architecture's downsampling requirements (divisible by 2^n)
- Support both center and right padding modes (configurable)

### 3.4 Sequence Assembly

The padded frames are assembled into batches of 20 sequential frames (configurable). The encoder receives a 3-D tensor of shape `(batch=8, seq_len=20, 112×112)`, treating the sequence dimension as the temporal context axis.

```
Frame stream:  [f1][f2][f3]...[f20][f21]...[f40]...
                └──────────────┘    └──────────────┘
                  Batch 1 (train)     Batch 2 (train)
```

---

## 4. Model Architecture — Reconstruction-Based Fuzz Detection

Five encoder variants are provided. All share a common decoder and training loop; only the encoder differs.

### 4.1 Convolutional Encoder (Stride 1×1) — Production Default

```
Input:  (B, seq_len=20, 112, 112)
  ↓
Conv2D(in_channels=seq_len, out_channels=1, kernel=3, stride=1, padding=1)
  ↓
Output: (B, 1, 112, 112)    ← compressed single-channel representation
```

**Design rationale**: The input channels are the sequence length (20 frames). Using a single output channel forces the model to compress the entire temporal sequence into one learned representation. The stride-1 convolution preserves spatial dimensions while creating a temporal relationship across stacked frames. The deliberate simplicity is justified empirically: this model matches the separation quality of the 223 MB transformer at 1/10th the speed and 5 orders of magnitude smaller on disk.

**Size**: 1.2 KB  
**Evaluation time**: 1.66 s per evaluation run  
**Separation scores**: Normal=1.00 / Attack=0.95 / End=1.00

### 4.2 Einstein-Sum Encoder

```
Input:  (B, seq_len, H, W)
  ↓
einsum('b s h w, s h w -> b h w', input_sequence, trainable_weight)
  ↓
Output: (B, H, W)    ← weighted combination of all frames via learned coefficients
```

**Design rationale**: The Einstein summation produces a compressed representation of the input sequence by contracting the sequence dimension using a trainable weight tensor of shape `(seq_len, H, W)`. Each position in the output is a learned linear combination of the corresponding position across all frames in the sequence. This creates a richer temporal abstraction than the stride-1 convolution because the combination weights are position-specific — different parts of the frame contribute differently to the compressed representation.

**Size**: 51.2 KB  
**Evaluation time**: 2.01 s  
**Separation scores**: Normal=1.00 / Attack=0.95 / End=1.00  
**Key property**: Wider Normal-vs-Attack margin under batch-size reduction (batch=2: separation stays at 0.943 without retraining)

### 4.3 U-Net Encoder

A full U-Net contracting path with skip connections. Used as a baseline for comparing simple vs. complex architectures.

```
Input: (B, seq_len, 112, 112)
  ↓ DoubleConv (3×3, BN, ReLU ×2)
  ↓ Down1 (MaxPool + DoubleConv) → 56×56
  ↓ Down2 (MaxPool + DoubleConv) → 28×28
  ↓ Down3 (MaxPool + DoubleConv) → 14×14
  ↓ Down4 (MaxPool + DoubleConv) → 7×7   ← bottleneck
  ↑ Up1 (ConvTranspose + concat skip + DoubleConv) → 14×14
  ↑ Up2 → 28×28
  ↑ Up3 → 56×56
  ↑ Up4 → 112×112
  ↓ OutConv (1×1 conv → 1 channel)
```

**Size**: 31.3 MB  
**Evaluation time**: 16.3 s  
**Separation scores**: Normal=1.00 / Attack=0.95 / End=1.00

### 4.4 U-Net + Einstein-Sum + Conv-Transpose 3-D Hybrid

Combines the U-Net contracting path with an Einstein-sum auxiliary encoder and a Conv-Transpose 3-D decoder. The Einstein-sum encoder runs in parallel with U-Net and its output is concatenated at the bottleneck, adding sequence-level context to the spatial features.

**Size**: 33.3 MB  
**Evaluation time**: 16.4 s  
**Separation scores**: Normal=1.00 / Attack=0.95 / End=1.00

### 4.5 U-Net + Einstein-Sum + Transformer Decoder

The most complex variant. Replaces the ConvTranspose 3-D decoder with a full Transformer decoder (2-layer multi-head self-attention, causal masking, positional encoding). The Transformer decoder autoregressively reconstructs the sequence from the bottleneck representation.

```
Bottleneck representation
  ↓
[Transformer Decoder]
  Layer 1: MultiHeadAttention(heads=8) + FFN + LayerNorm
  Layer 2: MultiHeadAttention(heads=8) + FFN + LayerNorm
  ↓
Reconstructed sequence
```

**Size**: 223.2 MB  
**Evaluation time**: 17.2 s  
**Separation scores**: Normal=1.00 / Attack=0.95 / End=1.00

### 4.6 Decoder — Conv-Transpose 3-D (shared across variants)

```python
# From models.py / fuz_conv.py
ConvTranspose3d(
    in_channels=1,
    out_channels=seq_len,
    kernel_size=(seq_len, 3, 3),
    stride=(seq_len, 1, 1),
    padding=(0, 1, 1)
)
```

The 3-D transposed convolution natively expands the temporal dimension from 1 back to `seq_len`, reconstructing the full frame sequence from the compressed single-channel bottleneck. This is intentionally simple: reconstruction **quality** is not the goal. The anomaly signal is the **delta** between reconstruction quality on normal data and reconstruction quality on anomalous data.

### 4.7 Loss Function — Fuzz Detection

**Primary**: Mean Squared Error (MSE) between input sequence and reconstructed sequence  
**Custom variant**: `customized_sigmoid.py` introduces a sigmoid-weighted MSE that down-weights near-zero bit positions (which are trivially easy to reconstruct) and emphasizes positions with high bit entropy

**Anomaly score**: Reconstruction loss per frame, averaged over the batch. The distribution of this score on normal data is used to set a detection threshold. Attack-phase reconstruction loss is separated from normal-phase loss by 0.95–1.00 separation score (Bhattacharyya-distance-like separation metric).

---

## 5. Model Architecture — Contextual Anomaly Engine (SOTIF)

The contextual engine is the core SOTIF contribution of CPYR. It detects the safety-critical LKA scenario described in Section 2.

### 5.1 The SOTIF Detection Problem

The challenge: an LKA frame being toggled OFF is not anomalous. An LKA frame being toggled OFF *while the RLD sensor is in the steering or switching phase* is a SOTIF trigger event. No point-in-time classifier can distinguish these cases. The model must reason about the history of sensor values.

```
Frame-level inspection (insufficient):
  LKA toggle → "is this value normal?" → YES (any state is permitted)

Context-level inspection (required):
  [RLD history] + [current RLD] + [LKA history] → "is this transition safe
  given the surrounding sensor context?" → NO (transition during hesitation phase)
```

### 5.2 Baseline Model

```
Filter ──► RLD(t) ──────────────────────► Predictor ──► LKA'(t)
              LKA(t) ──► Memory Unit ──► LKA(t-1) ──►
```

The baseline is a non-contextual model: a single linear layer with two inputs (current RLD, previous LKA) predicting the next LKA state. It has no access to RLD history. Result: overlapping loss distributions for safe and unsafe transitions; F1 = 0.42–0.80 across verification test cases.

**Purpose**: The baseline establishes that *non-contextual* models cannot solve SOTIF trigger detection, motivating the architecture of the full models.

### 5.3 LKA Predictor

```
Data stream ──► Filter ──► RLD(t) ──────────────────────────────────────────► Predictor ──► LKA'(t)
                    │                                                              ▲
                    ├──► LKA(t) ──► Memory Unit ──► LKA(t-1) ────────────────────┤
                    │                                                              │
                    └──► RLD history ──► History block (Conv layers) ─────────────┘
                                         compressed RLD representation
```

**Components in detail:**

**Filter block**: Extracts LKA and RLD frames from the heterogeneous data stream by frame ID. When augmented data is used, also injects configurable noise into RLD values to simulate sensor performance limitations.

**Memory unit**: A one-step delay register storing `LKA(t-1)`. Provides the model with the previous LKA state without requiring recurrence.

**History block**: A convolutional layer that stacks the existing compressed representation with the current RLD frame, producing an output of the same size as a single frame. This is a compressed rolling summary of all past RLD values — the model's "memory" of how the lane distance has been changing. Formally, it represents the full sequence of past frames in the shape of one frame.

**Predictor block**: Three linear layers:
```
Input: [current RLD, History representation, LKA(t-1)]
  ↓ Linear(input_size → 1024)
  ↓ Linear(1024 → 1024)      ← hidden layer
  ↓ Linear(1024 → LKA_frame_size)
Output: LKA'(t)              ← predicted next LKA state
```
The 1024-dimensional hidden layer was found to sharpen the output triggers at state transitions, producing clear spike-and-decay patterns that make threshold determination tractable.

**Detection logic**: The model is trained to predict what the LKA *should* be, given the RLD history and context. When the actual LKA contradicts the contextual prediction (e.g., LKA toggles ON while RLD history indicates a lane switch is in progress), the MSE loss spikes — this is the anomaly signal.

**Key question the model answers:** *"Given that the vehicle is currently switching lanes (RLD history), and LKA was last seen in state X, what should the next LKA state be?"*  
Correct answer: *"It should remain in state X."*  
Observed answer: *"It has transitioned to the opposite state."*  
Consequence: Large prediction error → SOTIF trigger detected.

**Trigger lag**: 0 batches at the anomaly batch. The model fires the instant the context-state combination becomes anomalous.

### 5.4 Enhanced LKA Predictor

Adds a **reconstruction objective** to the History block, ensuring the History representation genuinely carries temporal information rather than acting as a shortcut.

```
Data stream ──► Filter ──► RLD(t) ──────────────────────────────────────────► Predictor ──► LKA'(t)
                    │                                                              ▲
                    ├──► LKA(t) ──► Memory Unit ──► LKA(t-1) ────────────────────┤
                    │                                                              │
                    └──► RLD history ──► History block ──────────────────────────┤
                                              │                                    │
                                              └──► Reconstructor ──► RLD[t-N:t]   │
                                                   (ConvTranspose2D)               │
                                                   ← reconstruction loss ─────────┘
                                                   (auxiliary objective)
```

**Reconstructor**: A trainable `ConvTranspose2D` layer with `in_channels=1, out_channels=N`. It takes the History representation and reconstructs the last N RLD frames. Adding this reconstruction objective as an auxiliary loss (weight 0.5 each) guarantees the History block encodes temporal content — it cannot simply relay the most recent frame.

**Combined loss**:
```
loss_lka  = MSE(LKA(t), LKA'(t))
loss_hist = MSE(RLD[t-N:t], reconstructed_RLD[t-N:t])
loss      = 0.5 * loss_lka + 0.5 * loss_hist
```

**Trade-off**: The reconstruction objective slightly reduces sensitivity to positive-bias sensor distortion (model becomes more tolerant of gradual drift) while improving robustness to Gaussian noise. The overall F1 profile across 12 validation scenarios is comparable to LKA Predictor.

### 5.5 Algorithm Pseudocode

**LKA Predictor (Algorithm 1 from SAE paper):**
```
lka_id   ← LKAframe_ID
RLD_id   ← RLDframe_ID
lka_state ← zero(OFF)
RLD_t    ← normal RLD tensor
hist_t   ← zero

for each minibatch in dataset:
    if minibatch has frame of ID lka_id:
        lka_t ← xor(lka_id, 1)          # toggle
    if minibatch has frame of ID RLD_id:
        RLD_t ← RLD_frame
    hist_t      ← History(hist_t, RLD_t)
    lka_predicted ← Predictor(lka_{t-1}, RLD_t, hist_t)
    lka_{t-1}   ← lka_t

loss = MSE(lka_t, lka_predicted)
```

**Enhanced LKA Predictor (Algorithm 2)** — adds:
```
ws ← N (windowsize)
RLD_list ← empty list

# Inside loop, after prediction:
RLD_reconstructed ← Reconstructor(hist_t)
append(RLD_t in RLD_list)

loss_lka  = MSE(lka_t, lka_predicted)
loss_hist = MSE(hist_{t-ws:t}, hist_reconstructed_{t-ws:t})
loss      = 0.5 * loss_lka + 0.5 * loss_hist
```

---

## 6. Training, Verification & Validation

### 6.1 Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 1×10⁻³ (fuzz detection) / 1×10⁻⁴ (LKA, grid-searched) |
| Batch size | 8 |
| Sequence length | 20 frames (fuzz) / 10 frames (LKA) |
| Epochs | 6 (fuzz) / 4 (LKA) |
| Training batches | 368 (fuzz) = 58,880 frames |
| Validation batches | 92 (fuzz) = 14,720 frames |
| Hardware | Intel i5-8400, GeForce GTX 1070, 16 GB RAM |
| Framework | PyTorch ≥ 1.8 |

### 6.2 Dataset

**Fuzz attack dataset**: Proprietary in-vehicle Ethernet captures from a real vehicle. Four labeled phases (used only for evaluation, not training):

```
Phase labels (evaluation only):
  Normal   → baseline traffic, no attack
  Attack   → high-rate fuzz frame injection
  Disable  → host ECU halts due to ingest overload
  End      → frame rate returns to normal
```

**LKA dataset — Original**: Recorded on a real vehicle. Normal driving + one real safety-critical LKA scenario. Contains full frame flow including LKA, RLD, and all other frames.

**LKA dataset — Augmented RLD**: Synthesized from the distribution of the original dataset:
```
Phase A (constant): Sine wave + noise (small amplitude)
Phase B (steering): |x| for x ∈ [-1, 1] — absolute value curve
Phase C (switching/hesitation): |x| × square_wave — models the RLD oscillation
                                 during lane line crossing
```

**Augmented LKA**: LKA transitions randomized within each epoch to constant-phase positions only. This prevents the model from learning the specific timing of the training anomaly and instead learning the general principle: *transitions are unsafe during non-constant RLD phases*.

### 6.3 Verification Test Cases (7 cases)

Following SOTIF guidelines, structured test cases target the trigger event directly. All use clean RLD (no distortion).

| Case | Scenario | Expected | Validates |
|---|---|---|---|
| 1 | LKA OFF throughout, no transitions | No anomaly | Model tolerates all 3 RLD phases during LKA OFF |
| 2 | Single legitimate ON transition | Anomaly at switching phase | LKA cannot go ON during lane departure |
| 3 | 3 legitimate + 1 illegitimate OFF in steering phase | Anomaly at illegitimate only | Distinguishes transition timing |
| 4 | 1 legitimate ON + 1 illegitimate OFF in steering | Anomaly at OFF | ON→OFF cannot occur in steering phase |
| 5 | 1 illegitimate ON in steering phase | Anomaly at ON | OFF→ON cannot occur in steering phase |
| 6 | 1 illegitimate ON at local minimum RLD | Anomaly at ON | Same for local minimum scenario |
| 7 | Train of transitions (constant=safe, others=unsafe) | Anomaly at non-constant transitions | Threshold determination and precision/recall |

**Results:**

| Model | Params | Speed (mb/s) | Case 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 2 | 139.1 | 0.8 | 0.53 | — | 0.49 | 0.6 | 0.42 | 0.8 |
| LKA Predictor | 1,054,726 | 138.1 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |
| Enhanced LKA Pred. | 1,054,746 | 139.6 | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |

*F = true positive zero; F1 undefined (infinite) — counted as pass.*

### 6.4 Validation Test Cases (12 cases — 4 distortion families)

SOTIF validation tests unknown scenarios. RLD is distorted; LKA pattern is fixed to Case 7 (rich transition train).

| Family | Cases | Distortion | Range |
|---|---|---|---|
| Positive bias | 1–3 | Additive positive offset to RLD | 0.1→1.0 |
| Negative bias | 4–6 | Additive negative offset | 0.1→1.0 |
| Square wave | 7–9 | Square wave interference | 0.08, 0.04, 0.02 Hz |
| Gaussian noise | 10–12 | Zero-mean Gaussian | σ = 1×, 2×, 10× dataset σ |

**Results (F1 score):**

| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LKA Predictor | 1.0 | 1.0 | 0.80 | 1.0 | 1.0 | 0.80 | 1.0 | 0.98 | 0.96 | 0.98 | 0.95 | 0.88 |
| Enhanced | 0.98 | 0.96 | 0.78 | 1.0 | 1.0 | 0.80 | 1.0 | 0.98 | 0.98 | 0.96 | 0.95 | 1.00 |

---

## 7. Results

### 7.1 Fuzz Attack Detection — Separation Scores

Higher separation score = greater distance between the reconstruction loss distributions of normal and anomalous phases. Score of 1.0 means zero overlap; 0.95 means near-complete separation.

| Model | Size | Eval time | Normal | Attack/Disable | End |
|---|---|---|---|---|---|
| U-Net + Einstein + Transformer | 223.2 MB | 17.2 s | 1.00 | 0.95 | 1.00 |
| U-Net + Einstein + ConvTranspose3D | 33.3 MB | 16.4 s | 1.00 | 0.95 | 1.00 |
| U-Net | 31.3 MB | 16.3 s | 1.00 | 0.95 | 1.00 |
| **Convolutional Encoder** | **1.2 KB** | **1.66 s** | **1.00** | **0.95** | **1.00** |
| **Einstein-sum** | **51.2 KB** | **2.01 s** | **1.00** | **0.95** | **1.00** |

**Key finding**: The 1.2 KB convolutional encoder and 51.2 KB Einstein-sum encoder match the largest architectures on every separation metric, at 10× the speed and orders of magnitude smaller on disk. The hex-to-binary preprocessing is the reason: it collapses the problem to the point where a single 2-D convolution layer captures the relevant signal.

### 7.2 Ablation Under Reduced Batch Size

| Model | Batch=8: Normal / Attack / End | Batch=2 (no retrain): Normal / Attack / End | Batch=2 (retrained): Normal / Attack / End |
|---|---|---|---|
| Conv Encoder | 1.000 / 0.95 / 1.000 | 1.000 / 0.941 / 0.999 | 1.000 / **0.991** / 0.999 |
| Einstein-sum | 1.000 / 0.95 / 1.000 | 1.000 / 0.943 / 0.996 | 1.000 / **0.943** / 0.996 |

Einstein-sum provides a wider and more stable margin at small batch sizes — important for constrained ECU-class deployment where batch size may be forced to 2 or even 1.

### 7.3 LKA Contextual Anomaly — Trigger Lag

Anomaly injected at batch 24.

| Model | BCE trigger batch | BCE lag | SSIM trigger batch | SSIM lag |
|---|---|---|---|---|
| **LKA Predictor** | **24** | **0 batches** | **24** | **0 batches** |
| VIS Predictor | 416–430 | 393 batches | 416–426 | 393 batches |

The LKA predictor achieves **instantaneous** detection at the exact anomaly batch. The VIS predictor provides a sustained 393-batch confirmation window. Together, they implement a surgical early-warning + confirmation alarm strategy.

---

## 8. Deployment Characteristics

### 8.1 Hardware Requirements

| Component | Minimum | Tested |
|---|---|---|
| CPU | Any modern automotive MCU | Intel i5-8400 |
| GPU | Optional (CPU inference for 1.2 KB model) | GeForce GTX 1070 |
| RAM | 64 MB (conv encoder) | 16 GB system |
| Storage | 1.2 KB (conv encoder weights) | — |
| Framework | PyTorch ≥ 1.8 or ONNX runtime | PyTorch 1.8 |

### 8.2 Export Path for Production Deployment

```
PyTorch state dict (.pt)
    ↓ torch.onnx.export()
ONNX model (.onnx)
    ↓ trtexec --onnx=model.onnx --saveEngine=model.trt --int8
TensorRT engine (.trt)
    ↓ TensorRT C++ / Python runtime
On-device inference (Jetson Orin, DRIVE AGX, or AUTOSAR-compliant ECU)
```

### 8.3 Computational Budget Summary

| Operation | Conv Encoder | Einstein-sum |
|---|---|---|
| Parameters | 576 | 26,112 |
| FLOPs per batch | ~0.1 MFLOP | ~0.5 MFLOP |
| Inference (batch=8) | 1.66 s (GTX 1070) | 2.01 s (GTX 1070) |
| TensorRT estimate (Jetson Orin) | <100 ms | <200 ms |
| Memory footprint | 1.2 KB weights + frame buffer | 51.2 KB weights + frame buffer |

---

---

# PART II — Traffic Vision: Hazardous Situation Detection via Future-Frame Prediction

---

## 1. Overview & Design Philosophy

Traffic Vision is an unsupervised deep learning system that monitors the road ahead through a forward-facing dashcam and raises a hazard alert approximately **250 milliseconds before** a dangerous maneuver fully unfolds.

The system is built around a fully convolutional spatio-temporal autoencoder trained exclusively on normal driving footage. At inference time, it forecasts the next frames of video. When the forecast diverges sharply from what actually happens, the divergence is the hazard signal.

**No labeled accidents. No bounding boxes. No per-vehicle tracking. No fixed hazard taxonomy.**

**Design goals:**

| Goal | Rationale |
|---|---|
| Unsupervised training | Accident labels are rare, legally sensitive, and distribution-shifted. Normal footage is abundant and continuously available. |
| Predictive (not reconstructive) | Reconstruction error catches appearance anomalies. Prediction error catches *behavioral* anomalies — unexpected motion — which is what dangerous driving actually is. |
| Class-agnostic | No fixed list of hazard categories. Any motion that violates the learned normal model is flagged, regardless of whether it matches a known label. |
| Real-time | Must exceed dashcam framerate (30 fps) on commodity GPU hardware. Achieves 36 fps on NVIDIA Tesla T4. |
| Continuous score output | A scalar anomaly score per window enables graduated alert levels without retraining. |

**Hardware platform**: NVIDIA Tesla T4 GPU (Google Colab). The system was developed, trained, and evaluated on NVIDIA infrastructure — it is not an aspirational GPU target but the actual development platform.

---

## 2. Core Concept: Behavioral Anomaly via Predictive Error

### 2.1 Reconstruction vs. Prediction

Classical anomaly-detection autoencoders reconstruct their own input. The reconstruction error flags appearance anomalies — unusual objects, unseen textures, atypical colors. This is useful but insufficient for traffic safety:

> A car executing a sudden U-turn in front of the host vehicle looks perfectly normal in any individual frame. Each pixel is a plausible pixel of a car on a road. Reconstruction error will be near zero. The car will not be flagged.

Future-frame prediction forces the model to internalize *how things ought to move*. The same car, having been tracked across the last 10 frames moving straight, should — 250 ms from now — be slightly further along the same trajectory. If instead it swings across the lane, every predicted pixel near that car is wrong. The MSE between prediction and observation spikes.

```
Reconstruction error: f(appearance)    → misses behavioral anomalies
Prediction error:     f(appearance AND motion) → catches both
```

### 2.2 The Quarter-Second Lookahead

At 30 fps, each frame spans ~33 ms. The model is configured to predict M output frames from N input frames. The gap between the last input frame and the last predicted frame gives the lookahead horizon:

```
Configuration: 10 input frames + 12 output frames at 30 fps
  Lookahead = 12 frames × (1/30 s) ≈ 400 ms
  Effective warning window ≈ 250 ms (conservative estimate used in demo)
```

This is a meaningful head-start for a human driver: average human reaction time to an unexpected event is 1.5–2.5 seconds; a 250 ms pre-warning reduces the required reaction time by 10–17%.

### 2.3 Uncertainty as a Safety Signal

Because the model outputs a continuous MSE score rather than a binary class, it behaves like a confidence measure:

```
Low MSE  → model's forecast matched reality → scene is routine → no alert
High MSE → model is surprised → scene deviates from normal → candidate hazard
```

The score is not a calibrated probability, but it is monotone with surprise: larger scores correspond reliably to scenes the model has not learned to expect. This is sufficient for threshold-based alerting, and for the graduated alert levels required by safety-system human-machine interfaces.

---

## 3. System Architecture

### 3.1 Four-Stage Pipeline

```
[Raw dashcam video (.mp4)]
     ↓ Stage 1: Frame extraction
[Cropped 320×480 RGB frames]
     ↓ Stage 2: Sequence assembly
[Tensor: (B, 3N, 320, 480)]   ← N frames × 3 RGB channels, stacked into channel dim
     ↓ Stage 3: Predictive autoencoder
[Tensor: (B, 3M, 320, 480)]   ← M predicted future frames
     ↓ Stage 4: Anomaly scoring
[Per-window MSE score + alert]
```

**Why frame stacking instead of 3-D convolutions or LSTMs?**

Folding the temporal axis into the channel dimension lets the entire model use plain 2-D convolutions. This choice sacrifices some temporal modeling capacity but gains:
- Compatibility with any 2-D CNN inference runtime (TensorRT, CoreML, ONNX)
- Full utilization of GPU SIMD lanes (2-D conv is the most optimized operation on any GPU)
- Faster-than-real-time throughput on a Tesla T4 with a simple 3-block encoder

Empirically, the ≈500 ms temporal window provided by 16 stacked frames at 30 fps is sufficient for the prediction task.

### 3.2 Encoder

The encoder progressively downsamples spatial dimensions while expanding channel count.

```python
class ConvAutoencoder(nn.Module):
    def __init__(self, in_num_frames, out_num_frames):
        super().__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_num_frames, 32,  3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,             64,  3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,             128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(3, padding=1, stride=2)
        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.t_bn1   = nn.BatchNorm2d(64)
        self.t_conv2 = nn.ConvTranspose2d(64,  32,  2, stride=2)
        self.t_bn2   = nn.BatchNorm2d(32)
        self.t_conv3 = nn.ConvTranspose2d(32,  out_num_frames, 2, stride=2)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.t_bn1(F.relu(self.t_conv1(x)))
        x = self.t_bn2(F.relu(self.t_conv2(x)))
        x = self.t_conv3(x)
        return x
```

**Tensor flow for the production configuration (10 input + 12 output frames):**

```
Input:      (B,  30, 320, 480)   # 10 frames × 3 channels
Conv+Pool:  (B,  32, 160, 240)
Conv+Pool:  (B,  64,  80, 120)
Conv+Pool:  (B, 128,  40,  60)   # bottleneck
t_conv:     (B,  64,  80, 120)
t_conv:     (B,  32, 160, 240)
t_conv:     (B,  36, 320, 480)   # 12 frames × 3 channels = 36
```

**Why 3 blocks and not deeper?**

Deeper prototypes (5–7 blocks, 256-channel bottleneck) reconstructed individual frames more crisply but memorized appearance detail, dampening their sensitivity to motion anomalies. The 3-block design generalizes better as a future-frame predictor: it is forced to compress to the 128-channel bottleneck, discarding texture detail and retaining only structural motion patterns.

### 3.3 Spatial Cropping

Each video source has a per-video crop window applied before resizing, removing the vehicle hood, sky, and mirrors so the model focuses on the drivable area:

```python
if videopath == "video_02.mp4":        # inference footage
    frame = frame[30:1045, 0:1890, :]  # tight road crop
elif videopath in ("video_00.mp4", "video_000.mp4"):   # training
    frame = frame[40:1040, :, :]

frame = cv2.resize(frame, (480, 320))  # (W, H) → (480, 320) = (H=320, W=480)
```

### 3.4 Model Variant Checkpoints

Six checkpoints with different input/output horizons were trained:

| Checkpoint | Input frames | Output frames | Role |
|---|---|---|---|
| `model_f24_f36.pt` | 8 | 12 | Short-horizon predictor |
| `model_f27_f36.pt` | 9 | 12 | Medium-short horizon |
| `model_f30_f36.pt` | 10 | 12 | Medium horizon |
| `model_f33_f36.pt` | 11 | 12 | Long-short horizon |
| `model_f30_f48.pt` | 10 | 16 | **Ground model** (context-rich) |
| `model_f36_f48.pt` | 12 | 16 | **Predict model** (far-horizon) |

The last two are used together in the dual-model inference configuration.

---

## 4. Dataset & Preprocessing

### 4.1 Dataset Split

| Split | Frames | Source |
|---|---|---|
| Training | ~17,370 (≈90%) | `video_00.mp4` / `video_000.mp4` (head) |
| Validation | ~1,000 (≈5%) | Video tail (held out) |
| Test | ~1,000 (≈5%) | Video tail (held out) |
| Inference demo | ~7,466 | `video_02.mp4` (fully unseen footage) |

### 4.2 Sequence Dataset

Frames are not consumed individually. A sliding window of N consecutive frames is stacked along the channel axis:

```python
class Images_Dataset(Dataset):
    def __getitem__(self, idx):
        seq = []
        for name in self.image_list[idx : idx + self.sequence_len]:
            img = cv2.imread(self.image_folder + name)
            if self.transform:
                img = self.transform(image=img)['image']
            seq.append(transforms.ToTensor()(img))
        return torch.cat(seq, dim=0)   # shape: (3 × L, H, W)
```

### 4.3 Augmentation

Photometric-only augmentation (Albumentations library). Geometric augmentation intentionally disabled — mirroring traffic direction or random spatial crops would corrupt the temporal motion semantics.

| Augmentation group | Operations | Probability |
|---|---|---|
| Lighting (Group A) | CLAHE, RandomBrightness, RandomGamma | 0.9 |
| Sharpness/Blur (Group B) | IAASharp, Blur(limit=3), MotionBlur(limit=3) | 0.9 |
| Noise | IAAAdditiveGaussianNoise | 0.2 |
| Perspective | IAAPerspective | 0.5 |

---

## 5. Training Procedure

### 5.1 Configuration

| Parameter | Value |
|---|---|
| Loss function | Pixel-wise MSE (nn.MSELoss) |
| Optimizer | Adam (β₁=0.9, β₂=0.999) |
| Learning rate | 1×10⁻³ (flat, no scheduler) |
| Batch size | 4 |
| Epochs | 30 (best-validation checkpoint kept) |
| Hardware | 1× NVIDIA Tesla T4 (15 GB) |
| Framework | PyTorch 1.7 + Albumentations 0.5.2 |

### 5.2 The Predictive Training Loop

```python
def training(model, train_loader, in_num_frames, epochs):
    model.train()
    running_loss = 0.0
    for input_ in tqdm(train_loader):
        input_ = input_.to(device)
        optimizer.zero_grad()
        # Model only sees first in_num_frames channels
        # But must predict the FULL sequence
        outputs = model(input_[:, 0:in_num_frames, :, :])
        loss = criterion(outputs, input_)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)
```

The model receives only `in_num_frames` channels (the recent past) and is asked to reproduce the entire `sequence_len` window, including the frames it did not see. The MSE is computed over the full window, dominated by the unseen future frames.

### 5.3 Training Curves

| Epoch | Train MSE | Val MSE |
|---|---|---|
| 1 | 8.8×10⁻⁴ | 2.18×10⁻³ |
| 5 | 7.8×10⁻⁴ | 2.00×10⁻³ |
| 10 | 7.2×10⁻⁴ | 1.65×10⁻³ |
| 15 | 7.0×10⁻⁴ | 9.1×10⁻⁴ |
| 20 | 6.8×10⁻⁴ | 5.3×10⁻⁴ |
| 25 | 6.7×10⁻⁴ | 4.5×10⁻⁴ |
| 30 | 6.7×10⁻⁴ | **4.3×10⁻⁴** |

Validation loss below training loss indicates the validation split is drawn from a quieter section of the route (less motion variation), not overfitting.

---

## 6. Dual-Model Inference Pipeline

### 6.1 Two-Stream Evaluation

The deployed inference configuration runs two predictors concurrently:

```python
in_num_frames_ground  = 10 * 3   # 10 frames of context
in_num_frames_predict = 12 * 3   # 12 frames of context

loss_list = evaluate(model_30_48, model_36_48,
                     images_loader,
                     in_num_frames_ground,
                     in_num_frames_predict)
```

| Stream | Model | Context | Horizon | Role |
|---|---|---|---|---|
| Ground | `model_f30_f48.pt` | 10 frames (330 ms) | 16 frames | "What is happening right now" |
| Predict | `model_f36_f48.pt` | 12 frames (400 ms) | 16 frames | "What is about to happen" |

The ground stream detects anomalies that are already unfolding. The predict stream anticipates anomalies before they fully develop. Together:
- If predict fires but ground does not: early warning (event forming, not yet confirmed)
- If both fire: high-confidence alert (event confirmed from two independent temporal windows)
- If ground fires but predict does not: event already happening (late detection)

This dual-stream architecture is structurally identical to CPYR's dual-predictor (instantaneous LKA + sustained VIS). Both systems independently converged on the same design principle: **pair a fast surgical detector with a confirmatory window**.

### 6.2 Anomaly Score Computation

```
s_t = (1 / (C × H × W)) × Σ_{c,h,w} (x_pred[c,h,w] - x_obs[c,h,w])²

C = 3 × M  (3 channels × M predicted frames)
H × W = 320 × 480
```

### 6.3 Threshold

Empirical detection threshold: **≈ 5×10⁻³**  
This threshold is a deployment parameter, not a model constant. Adjusting it requires no retraining:
- Lower threshold → higher recall, more false positives (safety-conservative product)
- Higher threshold → fewer interruptions, more missed events (comfort-conservative product)

---

## 7. Anomaly Scoring & Uncertainty

### 7.1 Score Interpretation

```
Score range (observed):
  Routine traffic:   0.5×10⁻³ – 2×10⁻³   (well below threshold)
  Ambiguous event:   2×10⁻³  – 5×10⁻³   (approaching threshold — soft alert)
  Dangerous event:   > 5×10⁻³             (above threshold — hard alert)
```

### 7.2 Continuous vs. Binary

The continuous score enables graduated alerts:

```
Level 0 (< 2×10⁻³):  No action
Level 1 (2–4×10⁻³):  Log event; driver-monitoring system note
Level 2 (4–5×10⁻³):  Amber HMI indicator; haptic feedback
Level 3 (> 5×10⁻³):  Active alert; ADAS pre-charge; V2X broadcast
```

This graduated response is directly relevant to SOTIF validation: the model's response to a gradually degrading situation (sensor noise, weather onset) produces a smoothly rising score rather than a snap from safe to unsafe — enabling controlled, proportional interventions.

---

## 8. Results & Demonstrated Events

### 8.1 Demo Event Table

(From the demonstration video, `youtube.com/watch?v=sudJ5_wccdI`)

| Timestamp | Predicted (250 ms ahead) | Observed | Alert behavior |
|---|---|---|---|
| 0:13–0:15 | White car trajectory crosses host path | Car stops to park across lane | Alert fires before parking maneuver completes |
| 0:20–0:23 | KIA van forecast to intersect host | Van merges in front | Alert fires before merge completes |
| 1:16–1:25 | Motorcycles and minivans flagged | Multiple unsafe lane changes | Sustained alert across sequence |
| 1:37–1:38 | Oncoming car anomalous trajectory | Car executes sudden U-turn | Alert fires before U-turn completes |
| 2:09–2:11 | Small Hyundai flagged as ambiguous | Hyundai stays in lane | **Alert correctly clears** — soft alert, no action needed |

The 2:09 event is the most diagnostically valuable: the system raises a soft alert when the trajectory becomes ambiguous, then de-escalates when the ambiguity resolves into safe behavior. This is the behavior required of a SOTIF-compliant system — it responds proportionally and correctly withdraws when the hazard does not materialize.

### 8.2 Visual Reconstruction Quality

On routine footage: predicted future frames are visually indistinguishable from observed frames. On anomalous footage: predicted frames contain motion artifacts — vehicles appear at the model's expected position, not where they actually moved. These artifacts are the numerically captured MSE spike.

---

## 9. Performance Characteristics

| Metric | Value | Notes |
|---|---|---|
| Throughput | ~36 frames/second | NVIDIA Tesla T4, torch.cuda.Event timer |
| Source video framerate | 30 fps | Standard dashcam |
| Real-time margin | ~20% | Sufficient for post-processing and alert routing |
| Per-batch latency (B=4) | ~902 ms | Cold first batch; warm batches faster |
| GPU memory footprint | Well within 15 GB T4 | Comfortable for production deployment |
| Validation MSE | 4.3×10⁻⁴ | Best checkpoint after 30 epochs |
| Training dataset size | ~17,370 frames | ~10 minutes of normal urban driving |

---

---

# PART III — NVIDIA METROPOLIS INTEGRATION PROPOSAL

---

## 1. The Safety Gap MISV Fills

NVIDIA Metropolis provides the infrastructure for city-scale intelligent video analytics: DeepStream SDK for real-time pipeline orchestration, TAO Toolkit for model transfer learning, Jetson hardware for edge inference, and a microservices layer for alert routing and dashboard integration.

What Metropolis does not natively provide — and what CPYR and Traffic Vision together offer — is a **safety argument** for the detectors it deploys.

Current traffic AI deployments face three unsolved problems:

**Problem 1 — Labeled data scarcity**: Supervised accident detectors require labeled near-miss and accident footage. Such data is rare, legally sensitive, and distribution-shifted between cities, seasons, and traffic compositions. Every new deployment requires new labeling.

**Problem 2 — Unknown hazard classes**: A supervised detector trained on known accident types cannot generalize to novel dangerous behaviors (new vehicle categories, new intersection geometries, new mixed-autonomy scenarios). It is by construction limited to its training taxonomy.

**Problem 3 — No SOTIF safety argument**: An anomaly detector that cannot demonstrate ISO 21448 compliance cannot be deployed in safety-critical applications (V2X alerting, emergency pre-charge, autonomous intervention). Without a formal safety argument, Metropolis partners are limited to advisory roles with no liability-grade evidence.

**CPYR + Traffic Vision + V2X resolves all three:**

| Problem | Solution |
|---|---|
| Labeled data scarcity | Both systems train on normal data only. No accident labels needed at any stage. |
| Unknown hazard classes | Semi-supervised anomaly detection automatically flags any deviation from normal — including novel events with no training label. |
| No SOTIF argument | MISV architecture provides formal Zone 2 reduction: product-probability argument across independent channels. Continuous per-frame scores feed directly into ISO 21448 clause 9 evidence. |

---

## 2. Unified System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                              MISV SYSTEM — HIGH-LEVEL ARCHITECTURE                          │
│                                                                                              │
│  ┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────────────────────┐   │
│  │   Data Sources   │    │   NVIDIA Metropolis  │    │   Output / Integration           │   │
│  │                  │    │   Edge Node (Jetson) │    │                                  │   │
│  │  📷 Camera ──────┼───►│  Traffic Vision      ├───►│  Score A (video anomaly)         │   │
│  │                  │    │  (DeepStream plugin) │    │                                  │   │
│  │  🚗 CAN/OBD ─────┼───►│  CPYR               ├───►│  Score B (network anomaly)       │   │
│  │                  │    │  (DeepStream plugin) │    │                                  │   │
│  │  📡 V2X RSU ─────┼───►│  V2X consistency    ├───►│  Score C (kinematic anomaly)     │   │
│  └──────────────────┘    │  checker             │    │                                  │   │
│                           │                      │    │  ┌──────────────────────────┐   │   │
│                           │  MISV Fusion Plugin  ├───►│  │  Alert Level 0/1/2/3     │   │   │
│                           │  (weighted vote)     │    │  │  ISO 21448 evidence log  │   │   │
│                           └──────────────────────┘    │  │  Zone 2 discovery report │   │   │
│                                                        │  │  V2X alert broadcast     │   │   │
│                                                        │  │  Dashboard / eCall       │   │   │
│                                                        │  └──────────────────────────┘   │   │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. CPYR → DeepStream Integration

### 3.1 Pipeline Design

```
[CAN/Ethernet tap or OBD-II adapter]
     ↓ Serial or Ethernet capture daemon
[Raw frame stream → hex capture]
     ↓ Preprocessing plugin (hex→binary, padding to 112×112)
[Binary tensor stream (B, 20, 112, 112)]
     ↓ nvinfer element: CPYR TensorRT engine
[Reconstruction loss per batch]
     ↓ Anomaly scoring element (loss delta vs. baseline distribution)
[Anomaly score A — network layer]
     ↓ nvmsgbroker → Metropolis microservices
```

### 3.2 TensorRT Export

```bash
# Step 1: Export from PyTorch
python -c "
import torch
from models import ConvEncoder  # 1.2 KB encoder
model = ConvEncoder()
model.load_state_dict(torch.load('cpyr_conv_encoder.pt'))
model.eval()
dummy = torch.zeros(1, 20, 112, 112)
torch.onnx.export(model, dummy, 'cpyr_encoder.onnx',
    input_names=['frame_sequence'],
    output_names=['reconstructed_sequence'],
    dynamic_axes={'frame_sequence': {0: 'batch_size'}}
)
"

# Step 2: Convert to TensorRT
trtexec \
    --onnx=cpyr_encoder.onnx \
    --saveEngine=cpyr_encoder_int8.trt \
    --int8 \
    --calib=normal_traffic_calibration_data/ \
    --workspace=256

# Step 3: Deploy via nvinfer in DeepStream GStreamer pipeline
# In deepstream_config.txt:
# [primary-gie]
# model-engine-file=cpyr_encoder_int8.trt
```

### 3.3 Anomaly Threshold Calibration

After deployment at a new site, a 30-minute normal traffic capture is sufficient to calibrate the anomaly threshold:

```python
# Calibration script (runs once per deployment site)
normal_scores = collect_scores(normal_traffic_recording, cpyr_model)
threshold_95 = np.percentile(normal_scores, 95)   # Level 1 alert
threshold_99 = np.percentile(normal_scores, 99)   # Level 2 alert
threshold_999 = np.percentile(normal_scores, 99.9) # Level 3 alert
```

---

## 4. Traffic Vision → DeepStream Integration

### 4.1 Pipeline Design

```
[nvvideosource: camera feed]
     ↓ nvstreammux
[Frame batches: (B, H, W, C) BGR]
     ↓ Frame preprocessing plugin:
       • crop to road ROI
       • resize to 320×480
       • normalize to [0,1]
       • stack N consecutive frames into channel dim
[Tensor: (B, 3N, 320, 480)]
     ↓ nvinfer element: Traffic Vision TensorRT engine (dual model)
[Ground prediction + Predict prediction]
     ↓ MSE scoring plugin
[Anomaly score B — visual layer]
     ↓ Optional: spatial heatmap extraction (per-pixel MSE map)
     ↓ nvmsgbroker → Metropolis microservices
```

### 4.2 TensorRT Export

```bash
# Export Traffic Vision encoder
python -c "
import torch
from model import ConvAutoencoder
model = ConvAutoencoder(in_num_frames=30, out_num_frames=36)
model.load_state_dict(torch.load('model_f30_f36.pt'))
model.eval()
dummy = torch.zeros(1, 30, 320, 480)
torch.onnx.export(model, dummy, 'traffic_vision_ground.onnx',
    input_names=['frame_sequence'],
    output_names=['predicted_frames'],
    opset_version=12
)
"

# FP16 TensorRT (Tesla T4 / Jetson AGX Orin)
trtexec \
    --onnx=traffic_vision_ground.onnx \
    --saveEngine=traffic_vision_ground_fp16.trt \
    --fp16 \
    --workspace=4096
```

### 4.3 Spatial Heatmap for Metropolis Dashboard

```python
# Per-pixel MSE map (localized anomaly visualization)
predicted = model(input_frames[:, :in_num_frames, :, :])
per_pixel_mse = F.mse_loss(predicted, input_frames, reduction='none')
heatmap = per_pixel_mse.mean(dim=1, keepdim=True)  # average over channels

# Overlay on original frame for dashboard display
heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
overlay = cv2.applyColorMap((heatmap_normalized * 255).byte().numpy(), cv2.COLORMAP_JET)
```

The heatmap shows *which region of the road* triggered the anomaly — a capability not available in point-detection systems. This supports downstream localization without requiring an object detector.

---

## 5. V2X as Third MISV Channel

V2X communication (C-V2X or DSRC) provides the kinematic state of surrounding vehicles: position, speed, heading, acceleration, and cooperative awareness messages (CAM/BSM). The V2X channel contributes to MISV by detecting anomalies that are:
- Camera-invisible (occluded vehicle, nighttime, fog)
- Network-invisible (hazard is external to the monitored vehicle)

**Kinematic consistency check**: Given the V2X-reported positions and velocities of neighboring vehicles at time *t*, a short-horizon physics model predicts where each vehicle should be at time *t+1*. Large deviation from the physics model is the anomaly signal for the V2X channel.

**Connection to the SAE paper scenario**: The vehicle approaching from behind in the LKA collision scenario is a V2X participant. Its anomalous acceleration into the vacating gap is detectable via V2X before it is visible on camera. V2X fires first; Traffic Vision confirms visually; CPYR confirms at the control layer.

```
Timeline of detection for the SAE 2021-01-0196 scenario:
  t=0:    V2X — rear vehicle accelerates anomalously (Score C crosses Level 1)
  t+100ms: Traffic Vision — rear vehicle trajectory diverges from prediction (Score B → Level 2)
  t+150ms: CPYR — LKA state transition while RLD in hesitation phase (Score A → Level 3)
  t+200ms: MISV fusion — 3/3 channels above threshold → EMERGENCY ALERT
  t+400ms: (Without MISV) Driver first perceives situation
```

---

## 6. TAO Toolkit Continuous Learning

Both systems share the semi-supervised property that makes TAO's continuous learning loop annotation-free:

```
Phase 1 — Bootstrap (once per model family):
  Load pretrained CPYR encoder from NGC catalog
  Load pretrained Traffic Vision encoder from NGC catalog

Phase 2 — Site adaptation (once per deployment location):
  Collect 30 min normal traffic (camera) + 30 min normal CAN captures
  TAO fine-tune: only final layers re-trained on local normal distribution
  Export updated TensorRT engines

Phase 3 — Continuous drift collection (ongoing):
  DeepStream flags high-loss frames from both streams automatically
  High-loss frames without confirmed incident = candidate Zone 2 events
  Accumulate in dataset buffer

Phase 4 — Periodic retraining (weekly or triggered by drift threshold):
  TAO re-trains on accumulated normal + flagged-but-unconfirmed frames
  New TensorRT engines pushed to edge nodes via NVIDIA Fleet Command
  Anomaly thresholds recalibrated against new normal distribution
```

**No human annotation required at any phase.** The only human decision is whether a flagged event was a true hazard (used for SOTIF evidence, not for model retraining).

---

## 7. Edge Deployment on Jetson

### 7.1 Hardware Mapping

| System | Recommended Jetson | Rationale |
|---|---|---|
| CPYR Conv Encoder (1.2 KB) | Jetson Orin Nano (8 GB) | Fits in L2 cache; INT8 inference; sub-100ms per batch |
| CPYR Einstein-sum (51.2 KB) | Jetson Orin NX 16G | Slightly larger, still tiny; better margin under small batch |
| Traffic Vision (ground + predict) | Jetson AGX Orin 64G | Dual-model evaluation; FP16; 36 fps already achieved on T4 |
| MISV Fusion Plugin | Any of the above | Pure CPU logic; negligible compute |
| V2X Parser | Jetson Orin NX | Protocol parsing + kinematic solver |

### 7.2 Power Budget

```
CPYR Conv Encoder (INT8):       ~0.1 W additional (fits in baseline inference budget)
Traffic Vision FP16:            ~8 W (Jetson AGX Orin MAX-N mode)
V2X parser:                     ~0.5 W (software-only, no accelerator needed)
Total MISV overhead:            ~9 W on Jetson AGX Orin
```

This is well within the 40 W TDP of the Jetson AGX Orin, leaving headroom for the Metropolis microservices, display, and communication stacks.

### 7.3 Latency Budget (End-to-End)

```
Camera capture → frame extraction:      ~10 ms
Sequence assembly (16 frames):          ~5 ms
Traffic Vision TensorRT inference:      ~28 ms (target at Jetson AGX FP16)
CAN capture → binary tensor:            ~2 ms
CPYR TensorRT inference:                ~5 ms (INT8, 1.2 KB model)
MISV fusion:                            <1 ms
Alert routing → nvmsgbroker:            ~5 ms
─────────────────────────────────────────────
Total end-to-end latency:              ~55 ms
Alert delivered to driver HMI:         ~55 ms after last input frame
Traffic Vision lookahead:              ~250 ms before event
─────────────────────────────────────────────
Effective net warning time:            ~195 ms before dangerous event
```

---

## 8. Smart City Application Scenarios

### 8.1 Intersection Safety Monitor (Infrastructure Mode)

```
Deployment: Fixed roadside camera + V2X RSU at high-risk intersection

Traffic Vision: Monitors all vehicles in field of view
  → Detects: wrong-way entry, red-light running trajectories, pedestrian path conflicts

CPYR (optional): Receives V2X CAM messages from connected vehicles
  → Detects: anomalous control transitions (LKA, AEB, ESC) near intersection

Metropolis output: Pre-emptive traffic light extension, V2X warning broadcast,
                   eCall trigger for detected collisions
```

### 8.2 Highway Incident Detection (Infrastructure Mode)

```
Deployment: Roadside cameras every 500m + V2X RSU

Traffic Vision: Monitors vehicle flow
  → Detects: sudden deceleration patterns, stopped vehicles, debris trajectories

CPYR: Receives V2X data from equipped trucks/buses
  → Detects: ABS activation patterns, anomalous steering corrections

Metropolis output: Variable message sign update, following-vehicle warning,
                   emergency service dispatch
```

### 8.3 Fleet Safety Monitoring (Vehicle Mode)

```
Deployment: Dashcam + OBD-II CAN tap on commercial fleet vehicles

Traffic Vision: Forward-facing camera
  → Detects: unsafe following distance, lane departure trajectories, pedestrian proximity

CPYR: CAN bus monitoring
  → Detects: LKA/ADAS misuse patterns, driver distraction signals (erratic steering),
             anomalous brake-accelerator combinations

Metropolis output: Real-time driver coaching HMI, fleet manager dashboard,
                   insurance telematics risk scoring
```

### 8.4 Parking Structure Safety (Infrastructure Mode)

```
Deployment: Fixed cameras + ultrasonic sensors in multi-level parking

Traffic Vision (adapted): Low-speed scenario — trains on normal parking maneuvers
  → Detects: wrong-way entry on ramps, pedestrian-vehicle proximity anomalies

Metropolis output: Gate control, pedestrian alert system, incident logging
```

---

## 9. SOTIF Compliance Argument

### 9.1 Zone 2 Reduction Through MISV

```
Without MISV:
  P(undetected Zone 2 event) = P(Zone 2 | single detector)

With MISV (3 independent channels):
  P(undetected Zone 2 event) = P(Zone 2 | Traffic Vision)
                               × P(Zone 2 | CPYR)
                               × P(Zone 2 | V2X)

If each channel has 5% Zone 2 exposure rate:
  P(single channel) = 0.05
  P(MISV triple channel) = 0.05 × 0.05 × 0.05 = 0.000125
  Reduction factor: 400×
```

### 9.2 ISO 21448 Clause 9 Evidence Chain

| Clause 9 Requirement | CPYR Contribution | Traffic Vision Contribution |
|---|---|---|
| 9.1 Evaluate triggering conditions | High-loss CAN events without incident → new Zone 2 candidates | High-MSE video windows without incident → new Zone 2 candidates |
| 9.2 Estimate residual risk | Published F1 > 0.90 over 12 validation scenarios (SAE paper Table 2) | Demo video: correct alerts + correct de-escalations |
| 9.3 Accept/reduce residual risk | MISV product-probability argument formalizes reduction | Joint MISV architecture reduces residual below single-channel threshold |
| 9.4 Evaluate safety | Continuous per-frame score provides auditable evidence stream | Spatial heatmap localizes hazard source for incident review |

### 9.3 Triggering Condition Discovery Loop

```
Continuous operation:
  Every high-loss frame from either channel is logged with timestamp, GPS, weather

Periodic analysis (weekly):
  Cluster high-loss events by type: camera anomalies, network anomalies, both
  Events in "both channels" cluster = highest-confidence new triggering conditions
  Events in single-channel cluster = candidates for further investigation

Quarterly SOTIF update:
  New triggering conditions added to the SOTIF hazard analysis
  Coverage improvement logged as Zone 2 → Zone 1 migration
  Updated evidence package submitted for functional safety audit
```

---

## 10. Phased Collaboration Roadmap

### Phase 1 — Dual-System Proof of Concept (60 days)

**Objective**: Demonstrate both systems running concurrently on Metropolis hardware.

| Task | Owner | Duration |
|---|---|---|
| Export CPYR conv encoder to ONNX/TensorRT | AT Instruments | Week 1–2 |
| Export Traffic Vision (both checkpoints) to ONNX/TensorRT | AT Instruments | Week 1–2 |
| Deploy both as DeepStream `nvinfer` plugins | Joint | Week 3–4 |
| Calibrate anomaly thresholds on shared normal dataset | Joint | Week 4–5 |
| Implement MISV fusion plugin (2-of-3 weighted vote) | AT Instruments | Week 5–6 |
| Run dual-stream evaluation on Metropolis benchmark footage | Joint | Week 7–8 |

**Deliverables:**
- Two live DeepStream plugin packages (CPYR + Traffic Vision)
- Anomaly score ROC curves for each channel independently and fused
- Separation score table matching SAE paper format
- MISV system running on Jetson AGX Orin devkit

### Phase 2 — V2X Third Channel & Full Metropolis Integration (90 days)

**Objective**: Complete the MISV triad and connect to Metropolis microservices.

| Task | Owner | Duration |
|---|---|---|
| V2X message parser and kinematic consistency checker | AT Instruments | Month 1 |
| Three-stream MISV fusion and alert routing | Joint | Month 2 |
| Metropolis microservices integration (nvmsgbroker → dashboard) | NVIDIA | Month 2 |
| TAO fine-tuning pipeline (annotation-free, continuous) | Joint | Month 3 |
| NVIDIA Fleet Command integration for model updates | NVIDIA | Month 3 |
| Field test at one intersection (camera + V2X RSU) | Joint | Month 3 |

**Deliverables:**
- Three-channel MISV live at a real intersection
- False-positive rate comparison: single-channel vs. fused
- TAO continuous learning loop running without annotation
- SOTIF evidence package draft (ISO 21448 clause 9 format)

### Phase 3 — SOTIF Certification Support & Ecosystem Expansion (ongoing)

**Objective**: Establish MISV as a validated Metropolis partner solution with regulatory backing.

| Task | Timeline |
|---|---|
| Submit updated results for SAE follow-on paper | Month 4–6 |
| ISO 21448:2022 clause 9 evidence package (full) | Month 6–9 |
| NVIDIA Developer Blog application note | Month 6 |
| NGC model catalog: CPYR + Traffic Vision pretrained encoders | Month 6 |
| Extended deployment: 5 intersections, 2 highway segments | Month 9–12 |
| Multi-city normal dataset for improved generalization | Month 12+ |

---

## 11. Business Case Summary

### 11.1 Market Fit

NVIDIA Metropolis currently serves over 1,000 cities and 100+ partners in transportation and public safety. The MISV offering addresses a gap no existing partner fills: a SOTIF-compliant, annotation-free, dual-modality anomaly detector with a formal ISO 21448 safety argument.

**Target segments within Metropolis:**

| Segment | How MISV helps | Metropolis integration |
|---|---|---|
| Smart city traffic management | Reduces false-positive rate of existing camera-only detectors | Plug in alongside existing TAO models |
| Commercial fleet safety | No annotation bottleneck; adapts to any route within 30 min of normal captures | Vehicle-mode deployment on Jetson Orin NX |
| Autonomous vehicle validation | Provides ground-truth anomaly evidence for SOTIF Zone 2 reporting | DeepStream plugin for AV test fleet |
| Insurance telematics | Continuous risk scoring without accident labels | SaaS anomaly score stream via Metropolis microservices |
| Emergency response (eCall, ERA-GLONASS) | Sub-100ms alert generation with 250ms video lookahead | nvmsgbroker → emergency gateway |

### 11.2 Technical Differentiators

| Differentiator | Evidence |
|---|---|
| Only published SOTIF implementation | SAE WCX 2021, DOI 10.4271/2021-01-0196 — peer reviewed |
| Zero annotation required | Semi-supervised training proven on real vehicle data (both systems) |
| Already on NVIDIA hardware | Traffic Vision: developed on Tesla T4. CPYR: 1.2 KB fits L2 cache on any Jetson. |
| Proven dual-stream architecture | CPYR: LKA+VIS predictors. Traffic Vision: ground+predict models. Same principle, independent development. |
| MISV product-probability Zone 2 reduction | Formal SOTIF argument ready for ISO 21448 audit |
| Continuous anomaly score (not binary) | Graduated alert levels; proportional ADAS response; auditable evidence log |

### 11.3 Ask

We propose to enter the **NVIDIA Metropolis Partner Program** with the following engagement:

1. **Technical integration support** from NVIDIA to complete DeepStream plugin packaging and NGC model catalog onboarding
2. **Access to one Metropolis pilot city** for field validation of the MISV three-channel system
3. **Co-authorship on the follow-on SAE paper and NVIDIA Developer Blog post** establishing the MISV approach as an NVIDIA-validated architecture
4. **Joint go-to-market positioning** as the SOTIF-compliance layer for Metropolis traffic safety deployments

In return, AT Instruments / EVRaid provides:
- Both fully trained, production-ready model weights (CPYR + Traffic Vision)
- Complete DeepStream plugin source code
- TAO fine-tuning scripts and calibration tools
- Full ISO 21448 methodology documentation and evidence templates
- Ongoing development of the V2X third channel

---

*End of document.*

**SAE Paper:** Abdulazim, A., Elbahaey, M., and Mohamed, A., "Putting Safety of Intended Functionality SOTIF into Practice," SAE WCX Digital Summit, April 2021. DOI: 10.4271/2021-01-0196  
**Demo (Traffic Vision):** youtube.com/watch?v=sudJ5_wccdI  
**Demo (ASRG / CPYR):** youtube.com/watch?v=z3uAQIN0nYw  
**Demo (Combined MISV):** youtube.com/watch?v=LKH6Nsu54wc  
**Code (CPYR):** github.com/moustafa991982/Cpyr  
**Project site:** https://at-instr.com
