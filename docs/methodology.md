# CPYR — Methodology & Technical Notes

This document is the long-form technical companion to the [project README](../README.md). It expands on the published abstract of *Putting Safety of Intended Functionality SOTIF into Practice* (SAE 2021-01-0196) and the ASRG community talk, and documents the architectures, training setup, and results reproduced in this repository.

---

## 1. Problem statement

The growing connectivity of vehicles — Ethernet backbones, V2X, OTA, charging-station handshakes — has dramatically expanded the automotive attack surface. At the same time, ISO 21448 (SOTIF) requires manufacturers to argue safety not only against malfunctions (the ISO 26262 question) but against hazardous behaviors arising from functional insufficiencies, performance limitations, and foreseeable misuse — including those introduced by ML-based components.

Two practical implications:

1. The detection problem is **open-world**: novel attacks and previously unseen trigger conditions both matter. Closed-set classifiers don't help.
2. The deployment target is **resource-constrained**: ECU-class hardware, real-time budgets, and OTA-update size limits. A 1 GB transformer is not shipping.

CPYR therefore pursues **deep anomaly detection (DAD)** with three constraints: (a) train on normal traffic only, (b) generalise to unseen attacks, (c) stay small and fast enough to be deployable.

---

## 2. Why semi-supervised, why deep

Anomaly detection breaks into three regimes by data availability:

- **Supervised** — binary/multi-class classifier with labelled normal *and* anomalous data. Strong on known attack classes, blind to new ones, and dependent on labels that are expensive (and often impossible) to obtain at the rate the threat landscape moves.
- **Unsupervised** — outlier detection from intrinsic data structure. Doesn't need labels but assumes anomalies are rare; if the assumption fails, false-positive rates explode.
- **Semi-supervised** — train a model to *understand* normal behavior, flag deviations from that. Labelled normal data is cheap to collect (any fleet-vehicle log), and the approach generalises to unseen anomaly types by construction.

CPYR uses semi-supervised reconstruction with attention/contextual augmentation. Reconstruction loss is the anomaly score: when the model sees a frame distribution it was never trained on, it reconstructs poorly, and the loss spikes.

---

## 3. Pipeline overview

```
        ┌──────────────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐
input → │ pre-processing   │ →  │ encoder  │ →  │ decoder  │ →  │ recon-loss   │ → anomaly score
        │  hex→bin, pad     │    │  (multi- │    │ (multi-  │    │  per frame    │
        │                   │    │   variant)│    │  variant) │    │               │
        └──────────────────┘    └──────────┘    └──────────┘    └──────────────┘
```

### 3.1 Pre-processing

Two operations:

1. **Hex → binary.** Each frame's hex payload is unrolled bit-wise. The model now reconstructs over a 2-class output space instead of 16, which (a) makes the reconstruction objective drastically easier, and (b) exposes bit-level statistical irregularities that are the actual signature of fuzz attacks.
2. **Padding to fixed shape (112 × 112).** Variable-length frames are padded so CNN/LSTM stacks can be applied uniformly, and so the padded representation is large enough to support U-Net-class encoders.

### 3.2 Encoder variants

| Variant | Mechanism | Notes |
|---|---|---|
| **Conv 2-D, stride 1×1** | Compresses input while preserving frame shape; creates an explicit time relation across the stacked sequence | Single-channel output. Reconstruction quality of itself is not the goal — only the gap between normal and anomalous reconstructions. |
| **Einstein-sum (`einsum`)** | Multilinear contraction between the input sequence and a *trainable* tensor matched to a single frame's shape | Output is a compressed sequence representation per batch. Used standalone, or as a contextual encoder alongside U-Net. |
| **U-Net** | Contracting + expansive paths with skip connections (after the Buda et al. PyTorch Hub implementation) | Naturally an autoencoder, modular depth, supports transfer learning, and skip connections ("helping signals") preserve reconstruction quality at high compression ratios. |
| **U-Net + einsum**, with either Transformer decoder or Conv-Transpose 3-D decoder | Two-level analysis: U-Net for *point* features, einsum for *sequence* context | Designed under the intuition that anomaly detection on sequences needs both per-frame and across-frame reasoning. |

### 3.3 Decoder

The default decoder is **Convolution-Transpose 3-D**, chosen because it natively expands a single dimension (reconstructing the temporal dimension from the compressed code). Empirically, simple decoders matched complex ones — the bottleneck and the binary output space do most of the work.

For the full U-Net + einsum variant, both a Transformer Decoder (after *Attention Is All You Need*) and Conv-Transpose 3-D were tested.

---

## 4. Fuzz-attack dataset & training protocol

Dataset: an automotive Ethernet fuzz-attack capture, divided into four phases:

1. **Normal** — baseline traffic.
2. **Attack** — host receives frames at very high rate.
3. **Disable** — host can no longer sustain ingest, halts.
4. **End** — frame rate returns to normal.

Training settings:

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 8 |
| Sequence length | 20 frames |
| Padded frame | 112 × 112 |
| Epochs | 6 |
| Train | 368 batches (58,880 frames) |
| Eval | 92 batches (14,720 frames) |
| Hardware | Intel i5-8400 @ 2.80 GHz × 6, GeForce GTX 1070, 16 GB RAM |

Padding dimensions chosen to (a) accommodate the longest observed frame and (b) leave headroom for U-Net's repeated downsampling.

---

## 5. Fuzz-attack results

### 5.1 Headline separation table

| Model | Normal sep. | Attack/Disable sep. | End sep. | Eval (s) | Size on disk |
|---|---|---|---|---|---|
| U-Net + einsum + Transformer decoder | 1.00 | 0.95 | 1.00 | 17.2 | 223.2 MB |
| U-Net + einsum + Conv-Transpose 3-D | 1.00 | 0.95 | 1.00 | 16.4 | 33.3 MB |
| U-Net | 1.00 | 0.95 | 1.00 | 16.3 | 31.3 MB |
| **Convolutional encoder** | **1.00** | **0.95** | **1.00** | **1.66** | **1.2 KB** |
| **Einstein sum** | **1.00** | **0.95** | **1.00** | **2.01** | **51.2 KB** |

**Interpretation.** Conv encoder and einsum are dominant on the Pareto frontier — they match deeper models on separation while being roughly an order of magnitude faster and four-to-six orders of magnitude smaller on disk. The hex-to-binary trick is doing most of the work: the model only needs to detect frames with abnormal 0-to-1 ratios, and that is a problem a tiny network can solve.

### 5.2 Disable-phase pattern

A repeated pattern was observed in the *Disable* phase across **all** models: a sudden drop in reconstruction loss corresponding to overlap with the *End* phase. Because this drop occurs in the transition between active anomaly and recovery, it does not impair operational detection — alarms have already fired during the Attack phase.

### 5.3 Apparent overlap of Normal and End

In some plots, *Normal* and *End* appear near-inseparable. This is a plot-scale artefact: the dynamic range is dominated by the much larger separation between {Normal, End} on one side and {Attack, Disable} on the other.

### 5.4 Ablation: batch size sensitivity

Reduced batch size from 8 → 2 to test stability under shorter observation windows:

| Model | No training: Normal / A-D / End | After training: Normal / A-D / End |
|---|---|---|
| Convolutional encoder | 1.000 / 0.941 / 0.999 | 1.000 / 0.991 / 0.999 |
| Einstein sum         | 1.000 / 0.943 / 0.996 | 1.000 / 0.943 / 0.996 |

Both stay stable. Training improves the conv encoder's Attack/Disable separation noticeably (0.941 → 0.991) but barely moves einsum — suggesting both have hit a representational ceiling on this dataset, and that for harder problems either should be stacked into a deeper architecture.

Einsum gives a wider Normal-vs-Attack margin in the loss space, useful when downstream thresholding is sensitive.

---

## 6. The LKA contextual anomaly — SAE 2021-01-0196 case study

This is the *contextual* anomaly problem from the SAE paper, and it requires a different architecture.

### 6.1 The scenario

The Lane Keep Assist (LKA) function keeps the car centred in its lane while enabled. The driver is allowed to disable LKA at will — disabling LKA is **not** an anomaly. But disabling LKA *during a lane switch the driver did not initiate* is a SOTIF trigger event: a single frame whose meaning depends entirely on its temporal/operational context.

Two frame types matter:

- **VIS** — distance to left and right lane markers.
- **LKA** — current LKA state (on/off).

A traditional reconstruction-based detector cannot solve this: the LKA-off frame is, in isolation, perfectly normal-looking. The anomaly only exists relative to surrounding VIS dynamics.

### 6.2 Architecture: predictive context models

CPYR introduces two predictive models that consume context and predict the *next* frame, then compare prediction to reality. Reconstruction loss is replaced by *prediction error*.

Components:

- **Filter** — extracts VIS and LKA frames from heterogeneous batches.
- **Memory unit** — provides a one-step time delay (so the model sees `frame(t-1)` while predicting `frame(t)`).
- **History** — a convolution layer that stacks itself with the current frame and outputs a same-shape tensor representing the entire historical sequence.
- **Compressor** — two linear layers (expand to 1024, compress back to single-frame dimension) acting as a feature extractor.

#### VIS predictor

Inputs: `VIS(t-1)`, VIS history, `LKA(t)`.
Output: `VIS'(t)` — predicted vision frame at time *t*.

#### LKA predictor

Inputs: `VIS(t-1)`, VIS history, `VIS(t)`, `LKA(t-1)`.
Output: `LKA'(t)` — predicted LKA state at time *t*.

### 6.3 Evaluation losses

Two losses are used to score prediction quality and surface the anomaly:

- **Binary Cross Entropy (BCE)** — bit-wise prediction error.
- **Structural Similarity (SSIM)** — perceptual similarity between predicted and actual frames.

### 6.4 Results

Anomaly was injected at batch 24.

| Model | BCE: trigger / lag (batches) | SSIM: trigger / lag (batches) |
|---|---|---|
| **LKA predictor** | **24 / 0** | **24 / 0** |
| VIS predictor | 416–430 / 393 | 416–426 / 393 |

The **LKA predictor** detects instantaneously — the moment the LKA state contradicts the VIS context, prediction error spikes. The **VIS predictor** produces a sustained, longer-window trigger that's easier to recognise on a noisy dashboard but lags substantially. The two are complementary: combine them for layered early-warning + sustained-alarm coverage, at the cost of additional compute and memory.

This pattern — **single-frame contextual anomaly + context-aware predictor + dual loss for early/late triggers** — is the core methodological contribution of the SAE 2021-01-0196 paper.

---

## 7. Anomaly taxonomy (reference)

For positioning CPYR against the broader anomaly-detection literature:

- **Point anomaly** — irregularity that occurs randomly with no particular interpretation. Classical outlier territory.
- **Contextual (conditional) anomaly** — a data instance that's only anomalous in a specific context (after Song et al., 2007). The LKA-disabled-during-lane-switch case is a textbook contextual anomaly.
- **Collective (group) anomaly** — individual points look normal in isolation but are anomalous in aggregate. The fuzz-attack frame storm is a collective anomaly: any individual frame is structurally legal; the *rate* and *aggregate distribution* are what's anomalous.

CPYR addresses **collective** anomalies (fuzz attacks) with reconstruction-based DAD and **contextual** anomalies (SOTIF triggers) with predictive context models.

---

## 8. Architecture notes

### 8.1 U-Net

Standard contracting + expansive architecture. Contracting path: repeated 3×3 convs (unpadded) + ReLU + 2×2 max-pool stride 2; channel count doubles per downsample step. Expansive path: 2×2 up-conv (halves channels) + concatenation with cropped skip from contracting path + two 3×3 convs + ReLU. Final 1×1 conv maps to output classes. 23 conv layers total.

Implementation adapted from the [Buda et al. PyTorch Hub U-Net](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/).

### 8.2 Transformer decoder

Used only the decoder block from *Attention Is All You Need*, stacked N = 2 layers. Each layer: multi-head self-attention → multi-head attention over encoder output → position-wise feed-forward. Residual connections + layer norm around each sub-layer. Causal masking ensures position *i* only attends to positions < *i*.

Implementation adapted from [`jadore801120/attention-is-all-you-need-pytorch`](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

### 8.3 Einstein-sum encoder

The contraction is `output = einsum('bsd,sd->bd', input_seq, trainable_tensor)` (schematic). The trainable tensor has the shape of a single frame and is learned end-to-end; it acts as a learned attention/projection across the sequence axis, producing a per-batch compressed code.

---

## 9. Reproducing the experiments

See [`README.md`](../README.md) for the quick-start. Notebooks of interest:

- `clean_parser.ipynb` — dataset parsing and hex→binary conversion.
- `Integration script.ipynb` — fuzz-attack training and evaluation across all variants.
- `unet for sequence model.ipynb` — U-Net + sequence experiments.
- `detecting lka state.ipynb` — LKA contextual anomaly detection (the SAE paper experiment).

The original datasets are not redistributed; replace the data loader with your own captures of comparable structure.

---

## 10. Limitations & future work

- **Datasets are proprietary.** Reproducing the exact published numbers requires the original captures. The architecture and training code generalise, but datasets matter.
- **Attack diversity.** The fuzz-attack experiment covers one attack family. SOTIF analysis suggests broadening to spoofing, replay, and stealthy low-rate attacks.
- **On-target deployment.** The 1.2 KB convolutional encoder is small enough for ECU deployment in principle; quantisation, AUTOSAR Adaptive integration, and timing-budget verification on automotive-grade silicon (e.g., Renesas R-Car, NXP S32, Qualcomm Ride) are open work.
- **Explainability.** SOTIF assurance increasingly requires per-decision explanations. Black-box XAI methods over CPYR's predictors (LIME, SHAP, integrated gradients) would help argue the safety case under emerging standards such as ISO/PAS 8800.
- **Combined alarms.** The dual VIS/LKA predictor design points to ensemble fusion — a fruitful direction not pursued in the original work.

---

## 11. References

1. Abdulazim, A., Elbahaey, M., Mohamed, A. *Putting Safety of Intended Functionality SOTIF into Practice.* SAE Technical Paper 2021-01-0196, 2021. https://doi.org/10.4271/2021-01-0196
2. ISO 21448:2022 — *Road vehicles — Safety of the intended functionality.*
3. Ronneberger, O., Fischer, P., Brox, T. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
4. Vaswani, A. et al. *Attention Is All You Need.* NeurIPS 2017.
5. Song, X., Wu, M., Jermaine, C., Ranka, S. *Conditional anomaly detection.* IEEE TKDE, 2007.
6. Buda, M. PyTorch Hub U-Net for brain MRI segmentation. https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
