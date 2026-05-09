# CPYR — Deep Anomaly Detection for In-Vehicle Networks

> A semi-supervised deep learning framework for detecting cyber-attacks and SOTIF-relevant anomalies on automotive in-vehicle networks. Reference implementation for the SAE WCX 2021 paper *"Putting Safety of Intended Functionality SOTIF into Practice"* and the companion ASRG community talk.

[![SAE Paper](https://img.shields.io/badge/SAE-2021--01--0196-blue)](https://www.sae.org/papers/putting-safety-intended-functionality-sotif-practice-2021-01-0196/)
[![ASRG Talk](https://img.shields.io/badge/ASRG-Presentation-red)](https://www.youtube.com/watch?v=z3uAQIN0nYw)
[![License](https://img.shields.io/badge/license-See%20LICENSE-lightgrey)](LICENSE)
[![Project Site](https://img.shields.io/badge/site-at--instr.com-success)](https://at-instr.com/)

---

## Why this project exists

Modern vehicles run dozens of interconnected ECUs over Ethernet- and CAN-based in-vehicle networks. Two distinct safety problems live in that traffic:

1. **Cybersecurity** — malicious actors injecting frames (fuzz attacks, denial-of-service, spoofing) to disable or manipulate functions.
2. **SOTIF** (ISO 21448) — hazardous behaviors that emerge from *correctly functioning* components reacting to context the system was never specified to handle, including foreseeable misuse and the performance limitations of ML-based features.

Classical signature-based IDS approaches struggle with both. They miss novel attacks, they can't reason about context, and they're often too heavy for ECU-scale deployment. CPYR proposes a different approach: **semi-supervised deep anomaly detection (DAD)**, trained only on normal traffic, small enough to run on automotive-grade hardware, and equipped with contextual awareness for SOTIF-relevant scenarios.

The work was first presented at the **SAE WCX Digital Summit 2021** as a practical demonstration of SOTIF applied to a Lane Keep Assist (LKA) cyber-physical anomaly, and later to the **automotive security community via ASRG**.

---

## Cited work

If you use or reference this code, please cite the original paper:

> Abdulazim, A., Elbahaey, M., and Mohamed, A., **"Putting Safety of Intended Functionality SOTIF into Practice,"** SAE WCX Digital Summit, April 13, 2021. SAE Technical Paper 2021-01-0196. https://doi.org/10.4271/2021-01-0196

A `CITATION.cff` is provided for automated tooling.

**Companion talk:** *AI Use Cases in Automotive Cybersecurity & SOTIF*, ASRG community presentation — https://www.youtube.com/watch?v=z3uAQIN0nYw

---

## Two contributions, one framework

CPYR addresses two complementary anomaly detection problems with a shared codebase:

### 1. Fuzz-attack detection on automotive Ethernet (cybersecurity)

A semi-supervised reconstruction-based detector trained on normal in-vehicle Ethernet traffic. At inference, the reconstruction loss separates four phases of a fuzz attack: **Normal → Attack → Disable → End-of-attack**.

Key design choices:

- **Hex-to-binary preprocessing.** Frames are unrolled from hexadecimal to a binary representation. This collapses the per-position output space from 16 classes to 2, making reconstruction tractable and exposing bit-level statistical anomalies that hex representations hide.
- **Stride-1 2-D convolution encoder.** Compresses the input while preserving frame shape and creating an explicit *temporal* relationship across stacked frames.
- **Einstein-summation alternative.** A trainable tensor combined via `einsum` produces a compressed sequence representation, used either standalone or as an auxiliary contextual encoder alongside U-Net.
- **Convolution-Transpose 3-D decoder.** Reconstructs the input from the compressed code; a deliberately simple decoder, since reconstruction *quality* matters less than the *delta* between normal and anomalous reconstruction.

### 2. LKA contextual anomaly detection (SOTIF)

The harder problem — and the one tied to the SAE 2021-01-0196 paper. Disabling Lane Keep Assist is **not** an anomaly per se (the driver is allowed to override). Disabling LKA *during a lane switch the driver did not initiate* is a **contextual anomaly** that may indicate an unauthorized command — exactly the SOTIF "trigger event" pattern: a single frame that's only anomalous given its temporal and operational context.

Two predictive models cooperate:

- **VIS predictor** — predicts the next vision frame (left/right lane-marker distances) from `VIS(t-1)`, VIS history, and current LKA state.
- **LKA predictor** — predicts the next LKA state from `VIS(t-1)`, VIS history, current VIS, and `LKA(t-1)`.

Both are evaluated with **Binary Cross Entropy (BCE)** and **Structural Similarity (SSIM)** loss. The LKA predictor produces an *instantaneous* trigger (zero-batch delay at the actual anomaly batch), while the VIS predictor produces a longer, more recognisable trigger window — combine the two for a layered early/late alarm.

---

## Headline results

### Fuzz-attack detection (separation scores; higher is better)

| Model | Normal | Attack/Disable | End | Eval time | Size on disk |
|---|---|---|---|---|---|
| U-Net + Einstein sum + Transformer Decoder | 1.00 | 0.95 | 1.00 | 17.2 s | 223.2 MB |
| U-Net + Einstein sum + Conv-Transpose 3-D | 1.00 | 0.95 | 1.00 | 16.4 s | 33.3 MB |
| U-Net | 1.00 | 0.95 | 1.00 | 16.3 s | 31.3 MB |
| **Convolutional Encoder** | **1.00** | **0.95** | **1.00** | **1.66 s** | **1.2 KB** |
| **Einstein sum** | **1.00** | **0.95** | **1.00** | **2.01 s** | **51.2 KB** |

The two simplest models match the deepest architectures on separation quality while being **~10× faster** and **orders of magnitude smaller on disk** — a direct consequence of the hex-to-binary preprocessing collapsing the problem to bit-ratio classification.

### Ablation under reduced batch size (batch = 2)

| Model | Without training: Normal / Attack-Disable / End | With training: Normal / Attack-Disable / End |
|---|---|---|
| Convolutional Encoder | 1.000 / 0.941 / 0.999 | 1.000 / 0.991 / 0.999 |
| Einstein sum | 1.000 / 0.943 / 0.996 | 1.000 / 0.943 / 0.996 |

Both remain stable as batch size shrinks; Einstein-sum yields a wider Normal-vs-Attack margin, useful when downstream thresholding matters.

### LKA contextual anomaly (anomaly injected at batch 24)

| Model | BCE: trigger batch / lag | SSIM: trigger batch / lag |
|---|---|---|
| **LKA predictor** | **24 / 0 batches** | **24 / 0 batches** |
| VIS predictor | 416–430 / 393 batches | 416–426 / 393 batches |

The LKA predictor fires the moment context contradicts state — instantaneous detection of the SOTIF trigger condition.

Full hyperparameters, training details, and methodology are in [`docs/methodology.md`](docs/methodology.md).

---

## Repository layout

```
Cpyr/
├── basemodel.py              # Base model class (training/inference scaffolding)
├── models.py                 # Model definitions: Conv encoder, einsum, U-Net variants
├── unet.py                   # U-Net implementation (after Buda et al. PyTorch Hub)
├── unet_ein.py               # U-Net + Einstein-sum hybrid encoder
├── modified_transformer/     # Transformer decoder (after "Attention Is All You Need")
├── ein.py                    # Einstein-summation encoder block
├── fuz_conv.py               # Fuzzy convolution layer
├── cbs.py                    # Conv-BN-Sigmoid composite block
├── customized_sigmoid.py     # Custom sigmoid activation
├── customized_softmax.py     # Custom softmax activation
├── up_down_sampler.py        # U-Net down/up-sampling primitives
├── resnet_hybrid.py          # ResNet hybrid backbone
├── data_handler.py           # Dataset loading, hex→binary, padding
├── data/                     # Dataset folder (binaries excluded — see .gitignore)
├── fit.py                    # Training loop
├── optimizer.py              # Adam wrapper
├── scheduler.py              # LR scheduling
├── callbacks.py              # Training callbacks (checkpoint, early stop)
├── meter.py                  # Loss/metric meters
├── utils.py                  # Misc utilities
├── records/                  # Training logs / experiment records
├── unet/                     # U-Net experiments
├── clean_parser.ipynb        # Dataset parsing & cleaning
├── Parser.ipynb              # Earlier parser notebook
├── unet for sequence model.ipynb     # U-Net sequence-model experiments
├── detecting lka state.ipynb         # LKA contextual anomaly detection (SAE paper)
├── Integration script.ipynb          # End-to-end integration
├── docs/
│   └── methodology.md        # Full technical write-up
├── CITATION.cff              # Citation metadata
├── requirements.txt          # Python dependencies
├── LICENSE
└── README.md
```

---

## Quick start

### Requirements

- Python 3.8+
- PyTorch ≥ 1.8 (CUDA recommended; experiments were run on a GeForce GTX 1070)
- See [`requirements.txt`](requirements.txt) for the full list

```bash
git clone https://github.com/moustafa991982/Cpyr.git
cd Cpyr
pip install -r requirements.txt
```

### Reproducing the fuzz-attack experiment

The original experiments use the following settings (from the technical documentation):

- Optimizer: **Adam**, learning rate `1e-3`
- Batch size: **8**, sequence length: **20 frames**
- Frame padding: **112 × 112**
- **6 epochs** on **368 training batches** (58,880 frames), evaluated on **92 batches** (14,720 frames)

Run the integration notebook:

```bash
jupyter notebook "Integration script.ipynb"
```

### Reproducing the LKA contextual experiment

```bash
jupyter notebook "detecting lka state.ipynb"
```

The notebook trains both VIS and LKA predictors and reproduces the trigger-lag table.

> **Dataset note.** The fuzz-attack and LKA datasets used in the original work contain proprietary vehicle traces and are not redistributed in this repository. The architecture and training code are fully reproducible against any in-vehicle Ethernet capture with comparable structure.

---

## How this fits into a broader portfolio

CPYR is the ML-and-safety leg of a three-part portfolio on **trustworthy automotive systems**:

| Domain | Repository / Reference |
|---|---|
| **AI safety & SOTIF** (this repo) | `Cpyr` — anomaly detection for in-vehicle networks, SAE 2021-01-0196 |
| **Confidential computing for SDV** | [`HSMConfidentialContainer`](https://github.com/moustafa991982/HSMConfidentialContainer) — Azure Confidential Containers + custom HSM daemon for SDV secure-boot key provisioning |
| **Automotive PKI & secure platforms** | LinkedIn article series on ISO 15118 Plug-and-Charge PKI; QNX `secpol` security policy research |

Together they span the layers you need to ship a defensible AI-enabled vehicle: secure silicon roots of trust → confidential workload provisioning → ML-driven runtime safety/security monitoring.

---

## Author

**Moustafa El Bahaey** — Chief Engineer / Chief Systems Engineer, EVRaid. ~18 years in automotive cybersecurity and embedded systems. ISO/SAE 21434, PMP, NVIDIA GTC 2021 speaker, IoT Tech Expo North America 2026 panelist.

- Project site: https://at-instr.com/
- LinkedIn: https://www.linkedin.com/in/moustafa-e-b-7726aa14/
- GitHub: https://github.com/moustafa991982

Co-authors of the SAE paper: A. Abdulazim, A. Mohamed.

---

## License

See [`LICENSE`](LICENSE).
