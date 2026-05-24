# CPYR — Project Briefing for NVIDIA Metropolis
### Traffic Accident Detection & Dangerous Situation Awareness Under ISO 21448 SOTIF
**Meeting: NVIDIA Metropolis | May 29, 2026**
**Presenter: Moustafa El Bahaey — Chief Engineer, EVRaid / at-instr.com**

---

## 1. Executive Summary

CPYR is a peer-reviewed, published deep-learning framework for detecting **dangerous situations before they become accidents** — using semi-supervised anomaly detection trained exclusively on normal operational data. Originally demonstrated on automotive in-vehicle networks (SAE WCX 2021, paper 2021-01-0196), the framework's core contribution is a SOTIF-native contextual anomaly engine that can be directly extended to infrastructure-level traffic monitoring on the NVIDIA Metropolis platform.

The single most relevant capability for NVIDIA Metropolis: **CPYR detects situations that are only dangerous in context** — the precise class of hazard that ISO 21448 SOTIF was written to address, and the class most frequently missed by classical rule-based or supervised detection systems.

---

## 2. The Safety Gap CPYR Fills

### 2.1 Why Conventional Detectors Miss SOTIF-Class Accidents

Traffic accident detection today is dominated by two approaches:

| Approach | How it works | What it misses |
|---|---|---|
| Rule-based (speed/gap thresholds) | Fixed thresholds on individual measurements | Context-dependent hazards: a legal speed is dangerous at a blind curve; a safe gap narrows in rain |
| Supervised DL (classification) | Labeled "accident" vs "no accident" frames | Rare, novel, or pre-accident build-up states that have no training label |

ISO 21448 SOTIF (2022) specifically defines the failure mode both approaches share: a system that is **functioning correctly** yet produces **hazardous outcomes** because its design assumptions do not cover the operational context it encounters. Every unmarked construction zone, every atypical pedestrian trajectory, every adverse weather condition that degrades sensor range is a SOTIF trigger.

### 2.2 The SOTIF Hazard Taxonomy (ISO 21448:2022)

```
Zone 1 — Known Unsafe:    known triggering conditions → hazardous behaviour (being eliminated)
Zone 2 — Unknown Unsafe:  unknown triggering conditions → hazardous behaviour (the hard problem)
Zone 3 — Known Safe:      known triggering conditions → safe behaviour
Zone 4 — Unknown Safe:    unknown triggering conditions → safe behaviour
```

The goal of SOTIF validation is to shrink Zone 2 toward zero. CPYR's semi-supervised contextual anomaly detection directly targets Zone 2: it learns the boundary of "normal context + normal behaviour" and raises an alarm when any combination drifts outside that boundary, regardless of whether the triggering condition was anticipated at design time.

---

## 3. CPYR: What the Framework Does

### 3.1 Published Validation (SAE 2021-01-0196)

> Abdulazim, A., Elbahaey, M., and Mohamed, A., **"Putting Safety of Intended Functionality SOTIF into Practice,"** SAE WCX Digital Summit, April 2021. DOI: 10.4271/2021-01-0196

The paper demonstrates two complementary detection problems using a shared codebase:

**Problem A — Collective (Cybersecurity) Anomaly**: Detecting fuzz attacks on automotive Ethernet by training only on normal network traffic. Reconstruction loss separates all four attack phases (Normal / Attack / Disable / End) with separation scores of 0.95–1.00 across five encoder architectures.

**Problem B — Contextual (SOTIF) Anomaly**: Detecting the disabling of Lane Keep Assist *during a lane switch the driver did not initiate* — a single-frame event that is **not an anomaly out of context** but is a SOTIF trigger event within its operational context. This is the direct analogue of a vehicle cutting into a lane when a gap is legal but the surrounding context makes the maneuver dangerous.

### 3.2 Core Technical Approach

**Semi-supervised training**: Models train exclusively on normal operational data. No labeled accident samples are required. The anomaly score is the reconstruction or prediction error delta between what the model expects given the current context and what it observes.

**Dual-predictor contextual engine**:

```
VIS Predictor:  [VIS(t-1), VIS history, LKA(t)]     → predicted VIS(t)
LKA Predictor:  [VIS(t-1), VIS history, VIS(t), LKA(t-1)] → predicted LKA(t)
```

The LKA predictor fires at **zero-batch delay** the instant context contradicts state. The VIS predictor provides a sustained 393-batch confirmation window. Together they implement a layered early-warning + confirmation alarm — directly matching how a human operator or downstream safety system needs to act.

**Dual-loss evaluation (BCE + SSIM)**: Binary Cross Entropy captures per-bit prediction errors; Structural Similarity Index captures perceptual/structural divergence. Using both simultaneously prevents either class of anomaly from going undetected when one metric degrades.

**Encoder flexibility**:

| Encoder | Size | Eval time | Deployment target |
|---|---|---|---|
| Convolutional (stride 1×1) | **1.2 KB** | 1.66 s | Jetson Orin Nano / edge ECU |
| Einstein-sum | 51.2 KB | 2.01 s | Jetson AGX / roadside unit |
| U-Net | 31.3 MB | 16.3 s | Data-center preprocessing |
| U-Net + einsum + Transformer | 223.2 MB | 17.2 s | Cloud analytics / model training |

The 1.2 KB convolutional encoder matches the largest architectures on separation quality (1.00 / 0.95 / 1.00) at 10× the speed — a direct consequence of the preprocessing step collapsing the problem to binary classification, which in turn maps cleanly to binary-format sensor or video feature streams.

---

## 4. Extension to Traffic Accident Detection

### 4.1 Generalizing the SOTIF Contextual Engine

The LKA experiment is a proof-of-concept for a general architectural pattern:

```
Any domain where:
  • Normal behavior can be characterized from data (semi-supervised)
  • A "dangerous situation" is a behavior that is only anomalous given operational context
  • Early warning (before the event) is more valuable than post-event classification

→ CPYR's dual-predictor contextual engine applies directly.
```

In the traffic infrastructure domain, the predictor pairs become:

| In-Vehicle (published) | Traffic Infrastructure (proposed) |
|---|---|
| VIS (lane marker distances) | Scene state: vehicle trajectories, gaps, speeds, occupancy |
| LKA (driver assist command) | Event signal: lane change, braking, merge, pedestrian cross |
| Anomaly: LKA disable during unintended lane switch | Anomaly: hard braking / lane change given scene context that predicts collision |

### 4.2 Dangerous Situation Classes Addressable by CPYR

**Pre-collision trajectory divergence**: Vehicles whose speed/heading combination is normal in isolation but anomalous given surrounding traffic density and gap closure rate. CPYR's contextual engine scores this as a prediction error without ever needing a labeled "near-miss" example.

**SOTIF-class sensor degradation**: A detection system functioning correctly (no hardware fault) but operating in fog, glare, or occlusion that pushes it into Zone 2. CPYR can be trained to predict expected detection outputs given scene metadata; deviation signals degraded perception rather than no detection.

**Atypical road user behavior**: Pedestrians, cyclists, or micro-mobility users whose individual frame is not anomalous but whose trajectory in context contradicts the normal scene model — exactly the SOTIF "unknown triggering condition" class.

**Wrong-way entry / road incursion**: A vehicle moving at a legal speed but in a direction that is contextually impossible given lane geometry. Reconstruction loss on the normal scene model spikes immediately.

**Adverse condition escalation**: Progressive sensor-environment mismatch during weather onset. Unlike threshold systems that snap from "safe" to "unsafe," CPYR's reconstruction loss rises continuously, enabling graduated alert levels and proactive intervention timing.

### 4.3 Why Semi-Supervised Is Critical Here

Labeled traffic accident datasets are:
- **Sparse**: Accidents are rare events; collecting enough labeled examples is prohibitively expensive.
- **Distribution-shifted**: Accidents from 2019 data do not cover today's mixed-autonomy traffic.
- **Privacy-constrained**: Video of accidents involves individuals and is often legally restricted.

Semi-supervised training requires only **normal operational footage** — abundant, continuously collectible, and legally unproblematic. The anomaly detector adapts to any new normal through re-training on the local scene's baseline, without any human labeling effort.

---

## 5. Fit with the NVIDIA Metropolis Platform

### 5.1 Architecture Mapping

```
NVIDIA Metropolis Stack          CPYR Component
─────────────────────────────────────────────────────────────────
Camera / sensor ingest           → Raw frame stream (replacing Ethernet frames)
DeepStream SDK pipeline          → Preprocessing + batching (hex→binary ≡ frame→feature)
TAO Toolkit (fine-tuning)        → Transfer learning: adapt pre-trained CPYR encoder
                                   to local scene using normal footage only
TensorRT inference               → CPYR 1.2 KB encoder: sub-2 s per batch, RT-compatible
NVIDIA Jetson Orin (edge)        → Edge inference node for roadside unit deployment
NVIDIA DGX / cloud               → U-Net + Transformer variant for deep retraining
NGC Model Registry               → CPYR encoder checkpoints as pretrained base models
Metropolis Microservices         → Anomaly score stream → alert routing → dashboard
```

### 5.2 DeepStream Integration Path

CPYR's input pipeline (raw data → fixed-size binary tensor → encoder → reconstruction/prediction loss → anomaly score) is a stateless inference graph that maps directly onto a DeepStream GStreamer pipeline:

```
[nvvideo source] → [nvstreammux] → [nvinfer: CPYR encoder] → [anomaly score plugin]
                                                               → [nvmsgbroker → alert]
```

The encoder's TensorRT compatibility is straightforward: PyTorch → ONNX export → `trtexec` → TensorRT engine. The 1.2 KB model generates a TensorRT plan that fits entirely in L2 cache on any Jetson Orin variant, enabling true zero-copy inference on edge hardware.

### 5.3 TAO Toolkit Workflow

The semi-supervised nature of CPYR makes it uniquely suited to TAO's continuous learning loop:

1. **Bootstrap**: Load pre-trained CPYR encoder from NGC (trained on aggregate normal traffic)
2. **Local fine-tune**: Stream normal footage from the deployment site; TAO re-trains the final layers on the local normal distribution (no labeling required)
3. **Drift collection**: DeepStream flags high-loss frames; these feed the next retraining cycle
4. **Threshold calibration**: Anomaly score percentiles from local normal data set deployment-specific thresholds automatically

This loop runs without human annotation at any stage, matching Metropolis's documented continuous-learning architecture for smart city deployments.

### 5.4 Edge Deployment Characteristics

| Requirement | CPYR Capability |
|---|---|
| Model footprint | 1.2 KB (conv encoder) — fits in microcontroller cache |
| Inference latency | < 2 s per batch of 20 frames at GTX 1070; faster on Jetson Orin with TensorRT |
| Training data | Normal operational data only; no accident labels |
| Hardware tested | Intel i5-8400 + GeForce GTX 1070 (consumer-class; Jetson Orin exceeds this) |
| Framework | PyTorch → ONNX → TensorRT (standard Metropolis path) |
| Retraining cadence | Online or batch; semi-supervised, no labeling bottleneck |

---

## 6. SOTIF Compliance Contribution

ISO 21448:2022 requires evidence that the residual risk from Zone 2 (unknown unsafe triggering conditions) has been reduced to an acceptable level. CPYR contributes to that evidence base in three ways:

**Detection evidence**: The anomaly score time-series provides a continuous, auditable record of how far each observed scene deviated from the normal model. This is a direct input to the SOTIF validation argument under clause 9 (Evaluation of the residual risk).

**Triggering condition discovery**: High-loss events that do not result in accidents are candidate "unknown triggering conditions." Clustering these events surfaces new Zone 2 entries for the SOTIF hazard log — closing the feedback loop that ISO 21448 requires.

**Performance limitation monitoring**: When sensor degradation (fog, occlusion) causes the perception stack to enter Zone 2, the reconstruction error of the scene model rises before any downstream detection failure occurs. This enables proactive intervention — the standard's preferred outcome over post-failure detection.

---

## 7. Differentiation vs. Existing Metropolis Solutions

| Dimension | Typical Metropolis Partner App | CPYR Approach |
|---|---|---|
| Training data requirement | Labeled accident / near-miss videos | Normal operational footage only |
| New hazard class handling | Requires retraining with new labels | Detects automatically (zero-shot for unseen normals) |
| Sensor modality | Video-specific | Modality-agnostic (any fixed-length feature stream) |
| Regulatory framing | Post-hoc detection / classification | SOTIF-compliant risk reduction argument |
| Model size for edge | Typically 10–200 MB | 1.2 KB (conv) to 51 KB (einsum) |
| Explainability | Black-box classification score | Reconstruction delta is interpretable: which bits/features deviated and by how much |
| Publication backing | Varies | SAE 2021-01-0196; peer-reviewed SOTIF methodology |

---

## 8. Proposed Collaboration Scope

### Phase 1 — Proof of Concept (60 days)
- Port CPYR convolutional encoder to ONNX/TensorRT
- Run semi-supervised training on a publicly available normal traffic dataset (e.g., HighD, inD, or a Metropolis partner dataset)
- Deploy on Jetson Orin devkit via DeepStream; measure anomaly score distributions vs. known incident clips
- Deliverable: anomaly score ROC curve against a held-out evaluation set; separation score table matching the SAE paper format

### Phase 2 — Metropolis Integration (90 days)
- Implement TAO fine-tuning pipeline for site-specific normal model
- Build DeepStream plugin wrapping the CPYR anomaly score
- Integrate anomaly score stream into Metropolis microservices alert routing
- Deliverable: end-to-end demo on live camera feed; alert latency < 3 s from event onset

### Phase 3 — SOTIF Validation Package (ongoing)
- Generate triggering-condition discovery reports from anomaly clusters
- Build SOTIF evidence package (ISO 21448 clause 9 format)
- Publish updated results as a follow-on to SAE 2021-01-0196
- Deliverable: co-authored technical paper / application note for NVIDIA Developer Blog

---

## 9. Author Background

**Moustafa El Bahaey** — Chief Engineer / Chief Systems Engineer, EVRaid  
~18 years in automotive cybersecurity and embedded systems  
ISO/SAE 21434 | PMP | NVIDIA GTC 2021 Speaker | IoT Tech Expo North America 2026 Panelist

- SAE paper: [DOI 10.4271/2021-01-0196](https://doi.org/10.4271/2021-01-0196)
- ASRG talk: [AI Use Cases in Automotive Cybersecurity & SOTIF](https://www.youtube.com/watch?v=z3uAQIN0nYw)
- Project site: https://at-instr.com/
- GitHub: https://github.com/moustafa991982/Cpyr

Co-authors of the SAE paper: A. Abdulazim, A. Mohamed

---

## 10. Key Messages (One Slide Each)

1. **The problem is SOTIF Zone 2.** Accidents caused by systems functioning correctly in unexpected context. Neither rules nor supervised classifiers reliably detect this class.

2. **CPYR is a published SOTIF detector.** The only published deep-learning framework framed explicitly as a SOTIF implementation, validated in peer review (SAE WCX 2021).

3. **Semi-supervised = no accident labels needed.** Train on normal footage only. The model learns what "safe" looks like; anything outside that boundary is a candidate hazard.

4. **The model is Jetson-sized.** 1.2 KB. Sub-2 second inference. No hardware excuse not to deploy it at every intersection.

5. **It fits Metropolis today.** PyTorch → ONNX → TensorRT. DeepStream plugin. TAO fine-tuning. The integration path is a known quantity, not a research project.

6. **It closes the ISO 21448 evidence loop.** Every anomaly score is an auditable data point for the SOTIF validation argument. Every high-loss non-accident event is a candidate Zone 2 triggering condition to feed back into the hazard log.

---

*Prepared for the NVIDIA Metropolis meeting, May 29, 2026.*
*SAE Technical Paper 2021-01-0196 — DOI: 10.4271/2021-01-0196*
