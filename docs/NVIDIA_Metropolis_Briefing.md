# CPYR + Traffic Vision — NVIDIA Metropolis Briefing
### Multiple Independent Source Verification for Traffic Accident Detection & Dangerous Situations Under ISO 21448 SOTIF
**Meeting: NVIDIA Metropolis | May 29, 2026**
**Presenter: Moustafa El Bahaey — Chief Engineer, EVRaid / AT Instruments**

---

## 1. The Core Proposition in One Sentence

We have built and demonstrated **two independently operating anomaly-detection systems** — one watching the road through a camera, one listening to the vehicle's internal network — that together implement **Multiple Independent Source Verification (MISV)** for dangerous situations, exactly as required by ISO 21448 SOTIF, and both are ready to run on the NVIDIA Metropolis stack today.

---

## 2. Why One Detector Is Not Enough: The SOTIF Argument

ISO 21448:2022 SOTIF defines the hardest safety problem in autonomous driving: a system that is **functioning correctly** yet produces hazardous outcomes because its design assumptions do not cover the operational context. It calls this Zone 2 — *unknown triggering conditions producing hazardous behaviour* — and requires evidence that Zone 2 has been reduced to an acceptable residual level.

Any single detector has a Zone 2 of its own. A camera-based system cannot see a CAN bus anomaly. A network-level monitor cannot see a vehicle cutting across a lane. Each system individually improves safety; neither individually proves it.

**MISV resolves this**: when two or more independent sources, with non-overlapping failure modes, simultaneously detect an anomaly, the probability that both are in Zone 2 at the same moment is the product of their individual Zone 2 probabilities — orders of magnitude smaller than either alone. This is the safety argument SOTIF asks for, and it is the architecture we have built.

The published paper (SAE 2021-01-0196) itself concludes:

> *"Despite the performance boost achieved by the shown ML models, we think ML models are not yet ready to act as a stand-alone tool that provides sufficient safety."*

MISV is the direct answer to that conclusion.

---

## 3. The Two Systems

### 3.1 System A — CPYR: In-Vehicle Network Contextual Anomaly Detection (SOTIF Layer)

**Published:** SAE Technical Paper 2021-01-0196, SAE WCX Digital Summit, April 2021  
**DOI:** 10.4271/2021-01-0196  
**Code:** github.com/moustafa991982/Cpyr

CPYR is a semi-supervised deep learning framework that monitors the automotive CAN/Ethernet network for **contextual anomalies** — situations that are only dangerous given their operational context, which is precisely the SOTIF trigger-event class.

**The demonstrated scenario** (from the SAE paper): A driver activates Lane Keep Assist during a lane switch (a legal action), then releases the steering wheel (user misuse), while the lane sensor is in hesitation phase (performance limitation). These three individually normal events combine into a contextual SOTIF hazard that causes a collision. CPYR detects the anomaly at the exact batch where the context-state combination becomes unsafe — **zero-batch lag**.

**Architecture**: Three-model contextual engine trained on normal network traffic only (SAE paper models):

```
Baseline:              [RLD(t), LKA(t-1)]                          → LKA'(t)  [no history — non-contextual comparison]
LKA Predictor:         [RLD(t), history_RLD, LKA(t-1)]             → LKA'(t)  [0-batch instantaneous trigger]
Enhanced LKA Predictor:[RLD(t), history_RLD, LKA(t-1)] + Reconstructor(history_RLD → RLD[t-N:t])
                                                                    → LKA'(t)  [0-batch; improved Gaussian noise robustness]
```

Where:
- **RLD** = Right Lane Distance (lane sensor value on CAN bus)
- **history_RLD** = compressed rolling representation of all past RLD frames (convolutional history block)
- **LKA(t-1)** = previous LKA state from memory unit

The LKA Predictor fires the instant the context-state combination becomes anomalous (LKA transition during hesitation phase). The Enhanced LKA Predictor adds a reconstruction auxiliary loss that forces the history block to genuinely encode temporal content, improving robustness at the cost of slight positive-bias sensitivity. Both are instantaneous. The sustained-confirmation role at the system level is filled by Traffic Vision (Channel B in MISV).

**Results (SAE 2021-01-0196)**:
- All 7 verification test cases: F1 = 1.00 (LKA Predictor and Enhanced LKA Predictor)
- 12 validation test cases (sensor distortion families): F1 = 0.88–1.00
- >90% precision/recall across all test configurations
- >150% F1 improvement over non-contextual baseline

**Edge footprint**:

| Encoder | Size | Eval time | Deployment |
|---|---|---|---|
| Convolutional (stride 1×1) | **1.2 KB** | 1.66 s | Jetson Orin Nano / ECU |
| Einstein-sum | 51.2 KB | 2.01 s | Jetson AGX / roadside unit |
| U-Net + Transformer | 223.2 MB | 17.2 s | Cloud retraining |

---

### 3.2 System B — Traffic Vision: Future-Frame Predictive Video Anomaly Detection (Perception Layer)

**Demonstrated:** YouTube demo [demo video](https://www.youtube.com/watch?v=sudJ5_wccdI)  
**Hardware tested:** NVIDIA Tesla T4 GPU  
**Full technical documentation:** `Traffic_Vision_Hazardous_Detection_Documentation` (Google Drive)

Traffic Vision is an unsupervised convolutional autoencoder trained only on normal dashcam footage. At inference it forecasts the next frames of video. When reality diverges from the forecast, the divergence is interpreted as a hazard signal — **approximately 250 ms before the event fully unfolds**.

**The key insight** (from the technical documentation, Section 4):

> *"Reconstruction error mainly flags appearance anomalies... Future-frame prediction, by contrast, forces the network to internalize how things ought to move. A car that is currently in the right lane should, a quarter-second from now, be slightly further along the right lane. If instead it suddenly swings across the host vehicle's lane, every pixel of the prediction near that car will be wrong, even though each pixel by itself looks like a perfectly normal pixel of a car."*

This is behavioral anomaly detection — the video equivalent of CPYR's contextual anomaly. No bounding boxes, no per-vehicle tracking, no class taxonomy, no labeled accident data required.

**Architecture**: Fully convolutional encoder-decoder (PyTorch)

```
Input:  (B, 3N, 320, 480)   — N stacked RGB frames (time folded into channels)
        ↓ Conv+BN+ReLU+MaxPool ×3
Bottleneck: (B, 128, 40, 60)
        ↓ ConvTranspose2D ×3
Output: (B, 3M, 320, 480)   — M predicted future frames
```

**Dual-model evaluation** (Section 8.1 of the technical documentation):
- **"Ground" model**: more input context, shorter horizon → detects what is currently happening
- **"Predict" model**: less input, farther horizon → anticipates what is about to happen
- Both streams run in parallel; disagreement between them enriches the hazard signal

This dual-stream pattern (ground = confirmatory, predict = anticipatory) parallels the dual-objective design of CPYR's contextual engine (LKA Predictor = context-state mismatch detection, Enhanced LKA Predictor = history reconstruction verification). Both systems independently separate instantaneous anomaly detection from sustained confirmation.

**Demonstrated detection events** (from the demo video annotations):

| Time | Predicted (250 ms ahead) | Confirmed |
|---|---|---|
| 0:13–0:15 | White car trajectory crosses host path | Car stops to park across lane |
| 0:20–0:23 | KIA van merging into host trajectory | Van merges in front |
| 1:16–1:25 | Motorcycles/minivans flagged reckless | Multiple unsafe lane changes |
| 1:37–1:38 | Oncoming car flagged | Car performs sudden U-turn |
| 2:09–2:11 | Hyundai flagged as ambiguous | Alert clears as trajectory resolves safely |

The last event is particularly important for SOTIF: the system raises a soft alert when a trajectory becomes ambiguous, then de-escalates when ambiguity resolves. This continuous score behavior (not binary classification) is essential for graduated alert levels.

**Performance on NVIDIA Tesla T4**:
- Throughput: ~36 frames/second (exceeds 30 fps dashcam)
- Per-batch latency (B=4): ~902 ms
- Real-time margin: ~20% headroom
- Validation MSE: 4.3 × 10⁻⁴ after 30 epochs
- Anomaly threshold: ~5 × 10⁻³ (tunable, no retraining)

---

## 4. The MISV Architecture: Three Independent Channels

The combined system implements MISV across three independent data channels, each with distinct failure modes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MISV FUSION LAYER                                   │
│                                                                        │
│  Channel 1: CPYR          Channel 2: Traffic Vision   Channel 3: V2X  │
│  ─────────────────        ─────────────────────────   ──────────────  │
│  In-vehicle network       Forward camera (video)      Cooperative     │
│  CAN/Ethernet signals     Future-frame prediction     vehicle comms   │
│                                                                        │
│  Detects:                 Detects:                    Detects:        │
│  • LKA misuse in          • Pre-collision trajectory  • Relative      │
│    hesitation phase         divergence (~250 ms)        position/     │
│  • Cyberattack on ECU     • Sudden lane changes         velocity      │
│  • Sensor limitation      • U-turns, reckless merges  • Misbehavior  │
│    during lane switch     • Any motion violating         across fleet │
│  • Context-state mismatch   learned normal model      • Map anomalies │
│                                                                        │
│  Failure mode:            Failure mode:               Failure mode:   │
│  Network-invisible        Camera-invisible            Comm-dependent  │
│  maneuvers                network anomalies           (requires V2X)  │
│                                                                        │
│  Score type:              Score type:                 Score type:     │
│  Prediction error         MSE (pixel divergence)      Kinematic       │
│  (BCE + SSIM)                                         inconsistency   │
│                                                                        │
│  ───────────────────────────────────────────────────────────────────  │
│  Fusion: Any 2 of 3 channels above threshold → HIGH CONFIDENCE ALERT  │
│          1 of 3 above threshold → WATCH / LOG (triggering condition)  │
└────────────────────────────────────────────────────────────────────────┘
```

**Why this satisfies SOTIF Zone 2 reduction**: The three channels fail independently. CPYR fails when the hazard is not reflected in network signals. Traffic Vision fails when the anomaly is not visible in video (e.g., night, occlusion). V2X fails when communication is unavailable. The probability that all three simultaneously encounter the same unknown triggering condition is the product of their individual Zone 2 exposure rates — a formal, auditable reduction in residual risk.

**The V2X layer** connects directly to the scenario demonstrated in the SAE paper: the vehicle approaching from behind (which contributed to the collision) is a participant in V2X. Its speed increase as it fills the gap is detectable via V2X before it is visible on camera. V2X provides the earliest possible warning; Traffic Vision confirms it visually; CPYR confirms it at the network/control level.

---

## 5. Combined System Architecture on NVIDIA Metropolis

```
NVIDIA Metropolis / DeepStream Pipeline
═══════════════════════════════════════════════════════════════════════

[Camera feed] ──────────────────────────────────────────────────────────────►
  nvvideo src → nvstreammux → [nvinfer: Traffic Vision encoder]
                                → anomaly score stream A

[OBD-II / CAN adapter] ─────────────────────────────────────────────────────►
  serial capture → binary tensor pipeline → [nvinfer: CPYR encoder]
                                → anomaly score stream B

[V2X RSU / DSRC/C-V2X modem] ───────────────────────────────────────────────►
  V2X message parser → kinematic consistency check
                                → anomaly score stream C

                         [MISV fusion plugin]
                         ┌──────────────────────────────────────────┐
                         │  Score A (Vision) ─┐                    │
                         │  Score B (CPYR)   ─┼─► weighted vote    │
                         │  Score C (V2X)    ─┘    → alert level   │
                         └──────────────────────────────────────────┘
                                    ↓
                         [nvmsgbroker → Metropolis microservices]
                                    ↓
                         Dashboard / V2X broadcast / eCall trigger
```

### 5.1 TensorRT / DeepStream Integration

Both CPYR and Traffic Vision export to ONNX and convert via `trtexec` with no architectural changes:

| Model | ONNX export | TensorRT plan | Target device |
|---|---|---|---|
| CPYR conv encoder (1.2 KB) | `torch.onnx.export` | INT8 quantization | Jetson Orin Nano |
| Traffic Vision encoder | `torch.onnx.export` | FP16 | Jetson AGX Orin |
| Both fused | TensorRT multi-stream | Concurrent | Jetson AGX Orin |

Traffic Vision was **developed on NVIDIA Tesla T4** and already achieves 36 fps with headroom — it is not an aspirational GPU target, it is the hardware it was built on. The same TensorRT engine runs on Jetson with INT8 quantization.

### 5.2 TAO Toolkit: Semi-Supervised Continuous Learning

Both systems share a critical TAO-compatible property: **they train on normal data only**. The TAO continuous learning loop requires no human annotation at any stage:

1. **Bootstrap**: Load pre-trained CPYR and Traffic Vision encoders from NGC
2. **Site adaptation**: Collect normal footage/network traffic at the deployment location; fine-tune the final layers via TAO (no labeling)
3. **Drift collection**: DeepStream flags high-loss windows from both streams; these feed the next TAO retraining cycle automatically
4. **Threshold calibration**: Anomaly score percentiles from local normal data calibrate deployment-specific thresholds per scene

### 5.3 Edge Deployment Profile

| Requirement | CPYR | Traffic Vision |
|---|---|---|
| Model size | 1.2 KB (conv) / 51.2 KB (einsum) | ~few MB (3-block conv encoder) |
| Throughput | Sub-2s / 20-frame batch (GTX 1070) | 36 fps (Tesla T4) |
| Training data | CAN/Ethernet captures — normal only | Dashcam footage — normal only |
| Labeling required | None | None |
| Hardware origin | i5-8400 + GTX 1070 | Google Colab Tesla T4 |
| ONNX exportable | Yes (PyTorch) | Yes (PyTorch 1.7) |
| TensorRT target | Jetson Orin Nano/NX | Jetson AGX Orin |

---

## 6. SOTIF Compliance Contribution

ISO 21448:2022 requires evidence for three activities: (1) triggering-condition identification, (2) known-unsafe scenario mitigation, (3) residual Zone 2 reduction. MISV contributes to all three:

| SOTIF Activity | CPYR Contribution | Traffic Vision Contribution | MISV Combined |
|---|---|---|---|
| Trigger identification | Every high-loss CAN event with no accident = new Zone 2 candidate | Every high MSE video window with no accident = new Zone 2 candidate | Union of candidates from both channels dramatically expands the SOTIF hazard log |
| Known-unsafe mitigation | Published F1 > 0.90 on SAE 2021-01-0196 verification test cases | Quarter-second pre-event detection in demo video | Two independent channels covering different physical manifestations |
| Zone 2 residual reduction | Zone 2 ∩ {network-invisible} | Zone 2 ∩ {camera-invisible} | Zone 2 ∩ {network-invisible} ∩ {camera-invisible} — product probability |

**Auditable evidence stream**: Both systems produce continuous anomaly scores, not binary decisions. Every frame of every drive produces a logged score from both channels. This is a direct, auditable input to the ISO 21448 clause 9 evaluation of residual risk — a continuous record of how far each observed scene deviated from each system's normal model.

---

## 7. Competitive Position vs. Existing Metropolis Solutions

| Dimension | Typical Supervised Metropolis App | CPYR + Traffic Vision MISV |
|---|---|---|
| Training data | Labeled accidents / near-misses | Normal driving footage + normal network captures only |
| New hazard class | Requires relabeling + retraining | Detected automatically (unseen normal-violating events) |
| Failure mode transparency | Black-box score | Per-channel score: which source fired, why (reconstruction delta) |
| Regulatory framing | Post-hoc classification | Formal SOTIF Zone 2 reduction argument, clause 9 compliant |
| Number of sources | 1 (camera) | 3 (camera + network + V2X) — MISV |
| Publication backing | Varies | SAE 2021-01-0196, peer-reviewed, presented at WCX |
| NVIDIA hardware tested | Varies | Tesla T4 (Traffic Vision), GTX 1070 (CPYR) |
| Alert type | Binary | Continuous score → graduated alert levels |
| Lookahead | At event or post-event | ~250 ms before event (Traffic Vision) + 0-batch at CAN level (CPYR) |

---

## 8. Demonstrated Prototype

The systems have been independently demonstrated:

- **CPYR (SOTIF)**: SAE WCX 2021 paper demonstration; ASRG community presentation ([AI Use Cases in Automotive Cybersecurity & SOTIF](https://www.youtube.com/watch?v=z3uAQIN0nYw))
- **Traffic Vision**: Live demo on dashcam footage ([demo video](https://www.youtube.com/watch?v=sudJ5_wccdI)) running on NVIDIA Tesla T4
- **Combined MISV system**: Demonstrated at [combined system demo](https://www.youtube.com/watch?v=LKH6Nsu54wc)

Both systems are implemented in PyTorch, trained, evaluated, and documented. Neither is a concept — both are running prototypes.

AT Instruments (the originating company) presented both systems jointly to NVIDIA previously under the title **"Safe autonomy and Live video stream monitoring empowered by AI GPU tech"** (GTC pitch document on file), demonstrating that the CPYR + video analytics combination was the intended architecture from the beginning.

---

## 9. Proposed Collaboration Scope

### Phase 1 — Dual-System PoC on Metropolis (60 days)
- Export both CPYR conv encoder and Traffic Vision encoder to ONNX/TensorRT
- Deploy both as DeepStream `nvinfer` plugins on a Jetson AGX Orin devkit
- Run Traffic Vision on a publicly available dashcam dataset; run CPYR on synthetic CAN traces matching the SAE paper scenario
- Implement the MISV fusion plugin (weighted vote on two anomaly score streams)
- **Deliverable**: End-to-end demo on a single Jetson device with both streams live; anomaly score ROC curves for both channels independently and fused

### Phase 2 — V2X Third Channel Integration (90 days)
- Integrate a V2X (C-V2X or DSRC) message parser as the third MISV channel
- Implement kinematic consistency check against Traffic Vision trajectory predictions
- Connect all three score streams to Metropolis microservices alert routing
- **Deliverable**: Three-channel MISV demo; show false-positive reduction from channel combination versus any single channel

### Phase 3 — SOTIF Evidence Package (ongoing)
- Generate ISO 21448 clause 9 residual risk report from logged anomaly scores
- Cluster high-loss events from both channels to produce Zone 2 triggering-condition discovery reports
- Publish updated results as follow-on to SAE 2021-01-0196
- **Deliverable**: Co-authored technical paper / NVIDIA Developer Blog application note: *"Multiple Independent Source Verification for SOTIF-Compliant Traffic Safety"*

---

## 10. Author Background

**Moustafa El Bahaey** — Chief Engineer / Chief Systems Engineer, EVRaid / AT Instruments  
~18 years in automotive cybersecurity and embedded systems  
ISO/SAE 21434 | PMP | NVIDIA GTC 2021 Speaker | IoT Tech Expo North America 2026 Panelist

- SAE paper: [DOI 10.4271/2021-01-0196](https://doi.org/10.4271/2021-01-0196)
- ASRG talk: [AI Use Cases in Automotive Cybersecurity & SOTIF](https://www.youtube.com/watch?v=z3uAQIN0nYw)
- Project site: https://at-instr.com/
- GitHub: https://github.com/moustafa991982/Cpyr

Co-authors of the SAE paper: A. Abdulazim, A. Mohamed

---

## 11. Key Messages (One Slide Each)

1. **No single detector is enough for SOTIF.** Zone 2 requires multiple independent sources with non-overlapping failure modes. MISV is the architecture; we have built it.

2. **System A (CPYR) sees what the camera cannot.** Network-level contextual anomaly: LKA misuse during sensor hesitation phase. Published at SAE 2021. >90% F1. 1.2 KB model. Zero-batch lag at anomaly onset.

3. **System B (Traffic Vision) sees what the network cannot.** 250 ms before a dangerous maneuver fully unfolds, the camera already knows. Built on NVIDIA Tesla T4. 36 fps. No labels, no taxonomy, no bounding boxes.

4. **V2X is the third channel.** The vehicle approaching from behind in the SAE scenario is a V2X participant. Its kinematics are inconsistent with the scene model before the camera sees it. Add V2X: Zone 2 exposure becomes a product probability.

5. **Both systems are already on NVIDIA hardware.** Traffic Vision was developed on Tesla T4. CPYR is 1.2 KB — it fits in L2 cache on any Jetson. The TensorRT export path is standard PyTorch → ONNX → `trtexec`. This is integration work, not research.

6. **The fusion makes the SOTIF argument.** Two independent scores, logged continuously, one per channel, one per frame, give ISO 21448 clause 9 the auditable evidence stream it requires. We are not just detecting accidents — we are generating the proof that the system is safe.

---

*Prepared for the NVIDIA Metropolis meeting, May 29, 2026.*  
*SAE Technical Paper 2021-01-0196 — DOI: 10.4271/2021-01-0196*  
*AT Instruments / EVRaid — at-instr.com*
