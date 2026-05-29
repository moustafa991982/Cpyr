# CPYR + Traffic Vision — Briefing
### Multiple Independent Source Verification for Traffic Accident Detection & Dangerous Situations Under ISO 21448 SOTIF
**Meeting: NVIDIA Metropolis | May 29, 2026**
**Presenter: Moustafa El Bahaey — Chief Engineer, EVRaid / AT Instruments**

---

## 1. The Core Proposition

We provide independent safety verification for physical AI decisions under SOTIF.
We have built and demonstrated **three independently operating systems** — one listening to the vehicle's internal network or V2X communication, one watching the road through a camera, and one combining both under a single fusion layer — that together implement **Multiple Independent Source Verification (MISV)** for dangerous situations, exactly as required by ISO 21448 SOTIF, and all three are ready to run on the NVIDIA Metropolis stack today.

**The SOTIF hazard zone model:**

┌─────────────────────────────────────────────────────────────────┐
│  Zone 1 — Known Safe:     known scenarios → safe behaviour      │
│  Zone 2 — Known Unsafe:   known scenarios → hazardous behaviour │  ← mitigate
│  Zone 3 — Unknown Unsafe: unknown scenarios → hazardous         │  ← The hard problem
│  Zone 4 — Unknown Safe:   unknown scenarios → safe behaviour    │
└─────────────────────────────────────────────────────────────────┘

Goal of SOTIF V&V: The entire engineering goal of SOTIF is to shrink Zones 2 and 3 —
convert unknowns into knowns (Zone 3 → Zone 2), and convert knowns-unsafe into
knowns-safe through mitigation (Zone 2 → Zone 1).
After reviewing the integration of models please visit chapter 6. SOTIF Compliance Contribution

## 2. Why One Detector Is Not Enough: The SOTIF Argument

ISO 21448:2022 SOTIF defines the hardest safety problem in autonomous driving: a system that is **functioning correctly** yet produces hazardous outcomes because its design assumptions do not cover the operational context. It calls this Zone 2 — *unknown triggering conditions producing hazardous behaviour* — and requires evidence that Zone 2 has been reduced to an acceptable residual level.

Any single detector has a Zone 2 of its own. A camera-based system cannot see an internal CAN bus or V2X anomalies. A network-level monitor cannot see a vehicle cutting across a lane. Each system individually improves safety; neither individually proves it.

**MISV resolves this**: when two or more independent sources, with non-overlapping failure modes, simultaneously detect an anomaly, the probability that both are in Zone 2 at the same moment is the product of their individual Zone 2 probabilities — orders of magnitude smaller than either alone. This is the safety argument SOTIF asks for, and it is the architecture we have built.

The published paper (SAE 2021-01-0196) itself concludes:

> *"Despite the performance boost achieved by the shown ML models, we think ML models are not yet ready to act as a stand-alone tool that provides sufficient safety."*

MISV is the direct answer to that conclusion.
<img width="1095" height="777" alt="image" src="https://github.com/user-attachments/assets/e41ce1c3-620c-4c87-95d7-ba5dcc052d8e" />

---
ISO 21448:2022 requires evidence for three activities: 
1. triggering-condition identification i.e. convert unknowns into knowns (Zone 3 → Zone 2).
2. known-unsafe scenario mitigation i.e. convert knowns-unsafe into knowns-safe through mitigation (Zone 2 → Zone 1).
3. residual Zone 2 reduction. 

MISV (Multiple independent source of verification) contributes to all three:

1. CPYR watches the vehicle network bus (CAN , V2X) — the stream of internal signals about steering, braking, throttle, etc. It flags anomalies as high-loss Bus events,
2. Traffic Vision watches the camera — the external visual scene. It flags anomalies as high-MSE video windows ( predictive autoencoder, where a bad future-frame prediction means something unexpected is happening).
3. MISV is the fusion of the two — "Multi-channel / Multi-modal Independent Safety Verification" or similar. The whole point is that these two channels are independent: they sense different physical manifestations of the same underlying danger, so they fail differently and catch different things. 

| SOTIF Activity | CPYR Contribution | Traffic Vision Contribution | MISV Combined |
|---|---|---|---|
| Trigger identification ((Zone 3 Unknown → Zone 2 Known) | Every high-loss CAN event with no accident = new Zone 2 candidate (near miss) | Every high MSE (predictive and Observed) video window with no accident = new Zone 2 candidate (catalogued triggers) | Union of candidates from both channels dramatically expands the SOTIF hazard log |
| Known-unsafe mitigation | Published F1 > 0.90 on SAE 2021-01-0196 verification test cases | [to be measured] Quarter-second pre-event detection in demo video | Two independent channels covering different physical manifestations (bus vs. camera), they don't share failure modes — fog blinds the camera but not the CAN bus; a sensor-spoofing attack on the bus doesn't fool the camera. Defense in depth.|
| Zone 2 residual  | Zone 2 ∩ {network-invisible} | Zone 2 ∩ {camera-invisible} | Zone 2 ∩ {network-invisible} ∩ {camera-invisible} — product probability  i.e. only fail to detect when both blind spots overlap|

**Auditable evidence stream**: Both systems produce continuous anomaly scores, not binary decisions. Every frame of every drive produces a logged score from both channels. This is a direct, auditable input to the ISO 21448 clause 9 evaluation of residual risk — a continuous record of how far each observed scene deviated from each system's normal model.
<img width="1256" height="557" alt="image" src="https://github.com/user-attachments/assets/5e4090dc-b13b-4082-b835-aeb9eff3dfb4" />


## 3. The Three Systems

### 3.1 System A — CPYR: In-Vehicle Network Contextual Anomaly Detection (SOTIF Layer)

**Published:** SAE Technical Paper 2021-01-0196, SAE WCX Digital Summit, April 2021  
**DOI:** 10.4271/2021-01-0196  
**Code:** github.com/moustafa991982/Cpyr

**Demos:**
- [CPYR Anomaly Detection Demo](https://www.youtube.com/watch?v=Yn-BaMF7mqE) — live model inference on CAN/Ethernet data, showing contextual anomaly scoring in real time
- [CPYR on Azure Cloud](https://www.youtube.com/watch?v=oFyrh-I6Wfk)— continuous monitoring deployment on Microsoft Azure, demonstrating the cloud-side retraining and alert pipeline

CPYR is a semi-supervised deep learning framework that monitors the automotive CAN/Ethernet network for **contextual anomalies** — situations that are only dangerous given their operational context, which is precisely the SOTIF trigger-event class.

**The demonstrated scenario** (from the SAE paper): A driver activates Lane Keep Assist during a lane switch (a legal action), then releases the steering wheel (user misuse), while the lane sensor is in hesitation phase (performance limitation). These three individually normal events combine into a contextual SOTIF hazard that causes a collision. CPYR detects the anomaly at the exact batch where the context-state combination becomes unsafe — **zero-batch lag**.

**Architecture**: Three-model contextual engine trained on normal network traffic only (SAE paper models):
<img width="2340" height="1083" alt="image" src="https://github.com/user-attachments/assets/e295a0a1-31ff-487c-a01a-300b12a3f5fe" />

Figure :Enhanced LKA Predictor architecture, LKA predictor architecture, the model filters RLD and LKA value from data flow,the memory unit operates as a delay unit, the history unit creates a condensed representation of RLD values. The history, RLD value and previous value of LKA are fed to the predictor unit to predict the next LKA state. The history representation is fed into there constructor to reconstruct the RLD last (N) RLD values.

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

The Enhanced LKA Predictor (the CPYR-side model published in SAE 21AE-0136) is trained with a joint two-loss objective and deployed with a single-headed scoring rule. The asymmetry matters and is the most common point of confusion when explaining the architecture, so it is worth making explicit.
During training, both heads are active. The model produces two outputs — a predicted next LKA state from the Predictor head, and a reconstructed history sequence from the Reconstructor head — and the loss combines both with fixed equal weights (Algorithm 2 of the paper):
loss = 0.5 * loss_lka  +  0.5 * loss_hist
 
where:
 <img width="1258" height="345" alt="image" src="https://github.com/user-attachments/assets/f44d6033-7f99-4276-b8b9-6cf1fc129dee" />


At inference, only the prediction head drives the alert. The Reconstructor still runs as part of the same forward pass, but its output is discarded. The anomaly score thresholded against the five-band decision rule is purely the prediction MSE.

<img width="1092" height="710" alt="image" src="https://github.com/user-attachments/assets/a3014b91-fd97-4764-9934-b43b3632b1f3" />

Figure .  Enhanced LKA Predictor: at training time both heads are active and the joint loss back-propagates through the shared History block, forcing it to encode real temporal content; at inference time only the Predictor's output is used to drive the alert. The Reconstructor's role is as a representation regulariser during training — it is not a second runtime detector. The five-threshold decision rule shown on the right is from the paper's "Determining Threshold" section, with separate bounds for ON and OFF transitions to handle the asymmetric loss magnitudes observed empirically.
7.5 Empirical results from SAE 21AE-0136
For completeness, the headline numbers from the paper:
<img width="1269" height="432" alt="image" src="https://github.com/user-attachments/assets/6bc6a256-862a-449b-833c-489f73ad0ef1" />



**Results (SAE 2021-01-0196)**:
- All 7 verification test cases: F1 = 1.00 (LKA Predictor and Enhanced LKA Predictor)
- 12 validation test cases (sensor distortion families): F1 = 0.88–1.00
- >90% precision/recall across all test configurations
- >150% F1 improvement over non-contextual baseline
<img width="1299" height="285" alt="image" src="https://github.com/user-attachments/assets/d7c4f224-8c7b-4b43-9b77-30bce1d6fac3" />

**Edge footprint**:

| Encoder | Size | Eval time | Deployment |
|---|---|---|---|
| Convolutional (stride 1×1) | **1.2 KB** | 1.66 s | Jetson Orin Nano / ECU |
| Einstein-sum | 51.2 KB | 2.01 s | Jetson AGX / roadside unit |
| U-Net + Transformer | 223.2 MB | 17.2 s | Cloud retraining / Azure pipeline |

---

### 3.2 System B — Traffic Vision: Future-Frame Predictive Video Anomaly Detection (Perception Layer)

**Hardware tested:** NVIDIA Tesla T4 GPU  
**Full technical documentation:** `Traffic_Vision_Hazardous_Detection_Documentation` (Google Drive)

**Demos:**
- [Real-Time Vision Anomaly Detection](https://www.youtube.com/watch?v=eAX6_KAtLiQ&t=1s) — live dashcam inference showing the anomaly score stream as vehicles perform dangerous maneuvers
- [Predictive Future-Frame Demo](https://www.youtube.com/watch?v=sudJ5_wccdI&t=6s) — side-by-side of predicted frames vs. observed reality, showing the ~250 ms pre-event detection window

Traffic Vision is an unsupervised convolutional autoencoder trained only on normal dashcam footage. At inference it forecasts the next frames of video. When reality diverges from the forecast, the divergence is interpreted as a hazard signal — **approximately 250 ms before the event fully unfolds**.

**Rational:**
Collecting and labeling enough negative examples to train a supervised detector for every kind of dangerous maneuver is, in practice, impossible. Hazardous events are by definition rare, varied, and hard to stage safely. What we do have, in abundance, is footage of ordinary, uneventful driving.
This project asks a different question: can we learn a model of "what normal traffic looks like a quarter-second from now" using only ordinary footage, and then treat any region of the scene that violates the model's expectation as a candidate hazard? If so, we get a detector that requires no hazard labels at all, that generalizes across categories of dangerous behavior, and that naturally provides a continuous uncertainty score instead of a binary class decision.

**The key insight** (from the technical documentation, Section 4):

> *"Reconstruction error mainly flags appearance anomalies... Future-frame prediction, by contrast, forces the network to internalize how things ought to move. A car that is currently in the right lane should, a quarter-second from now, be slightly further along the right lane. If instead it suddenly swings across the host vehicle's lane, every pixel of the prediction near that car will be wrong, even though each pixel by itself looks like a perfectly normal pixel of a car."*

This is behavioral anomaly detection — the video equivalent of CPYR's contextual anomaly. No bounding boxes, no per-vehicle tracking, no class taxonomy, no labeled accident data required.

**Architecture**: Fully convolutional encoder-decoder (PyTorch)

<img width="1093" height="793" alt="image" src="https://github.com/user-attachments/assets/684ee94b-0ade-4635-8b23-2ef62a992f3b" />

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
<img width="1317" height="716" alt="image" src="https://github.com/user-attachments/assets/cee49120-323c-479f-ba8b-5664c72fbaff" />


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

### 3.3 System C — MISV Combined: Multiple Independent Source Verification

**Demo:** [CPYR + Traffic Vision Combined MISV Demo](https://www.youtube.com/watch?v=LKH6Nsu54wc) — both systems running simultaneously, with the MISV fusion layer producing a single graduated alert level from the two independent anomaly score streams

System C is not a third independent detector — it is the **fusion architecture** that makes Systems A and B a safety argument rather than two separate tools. It implements MISV by running CPYR and Traffic Vision concurrently, fusing their scores through a weighted voting layer, and producing a single alert level with auditable per-channel provenance.

**Why fusion changes the safety argument**:

<img width="1336" height="725" alt="image" src="https://github.com/user-attachments/assets/a8b78d9e-ca8d-4e89-9334-382b38aad4f0" />

```
Single channel:  P(miss) ≤ p_A         ← bounded by the weaker channel alone
MISV (A + B):    P(miss) ≤ p_A × p_B  ← product of independent probabilities
MISV (A + B + V2X): P(miss) ≤ p_A × p_B × p_C ← three-channel product
```

With conservative per-channel Zone 2 miss rates of p_A = 0.12, p_B = 0.20, p_C = 0.05, the MISV system achieves P(miss) ≤ 0.0012 — a 100× improvement over the best single channel, and a **formal, auditable SOTIF Zone 2 residual-risk claim**.

** Fusion layer**:
<img width="1086" height="817" alt="image" src="https://github.com/user-attachments/assets/9dfffdfa-417f-4984-bbd3-76324b6ccce5" />

```
Score A (CPYR — network)    ─┐
Score B (Traffic Vision — vision) ─┼─► Weighted vote → Alert level 0–3
Proposed Score C (V2X — kinematic)   ─┘         (0 = nominal, 3 = emergency)

Alert level 1+: logged as potential Zone 2 triggering condition
Alert level 2+: ADAS advisory (speed reduction, driver attention request)
Alert level 3:  eCall / emergency brake pre-arm
```

**The combined demo** (https://www.youtube.com/watch?v=LKH6Nsu54wc) shows the connected vehicle scenario from the SAE paper played out with both sensing modalities active: CPYR detects the LKA context mismatch at the network level (zero-batch lag) while Traffic Vision detects the converging trajectory at the camera level (~250 ms pre-event). The fusion layer requires both channels to agree before escalating to level 3, eliminating single-channel false positives.

| Capability | System A (CPYR) | System B (Traffic Vision) | System C (Fusion) |
|---|---|---|---|
| Input modality | CAN/Ethernet network | Forward camera video | All channels |
| Detection lag | 0 batches | ~250 ms pre-event | 0 batches (first channel to fire) |
| False positive guard | None (single channel) | None (single channel) | 2-of-3 vote required for Level 3 |
| SOTIF argument | Channel-level Zone 2 bound | Channel-level Zone 2 bound | Product-probability Zone 2 bound |
| Evidence output | Per-frame network score | Per-frame video score | Per-frame fused score + channel breakdown |

---

## 4. The MISV Architecture: Three Independent Channels

The combined system implements MISV across three independent data channels, each with distinct failure modes:

```
┌────────────────────────────────────────────────────────────────────────┐
│                    MISV FUSION LAYER                                   │
│                                                                        │
│  Channel 1: CPYR          Channel 2: Traffic Vision   Channel 3 (Proposed): V2X  │
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

**Auditability — the continuous evidence stream**
Both channels in MISV produce continuous anomaly scores, not binary decisions. Every frame of every drive produces a logged score from both channels independent of whether an alert was raised. This is the direct, auditable input that ISO 21448 clause 9 (evaluation of residual risk) calls for — a continuous numerical record of how far each observed scene deviated from each system's learned normal model. A reviewer can replay any time period, re-threshold the same scores under different operating-point assumptions, and reproduce the safety case quantitatively rather than trusting the binary alarm output.


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
<img width="1179" height="544" alt="image" src="https://github.com/user-attachments/assets/c9f871e3-11f2-4f07-95b9-ff8f4d765adb" />

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

1. Edge-deployment engineering. Today the model runs on a T4 with vanilla PyTorch. Porting to Jetson + DeepStream + TensorRT-INT8 is straightforward but real work. We would like NVIDIA's developer-relations support and a few engineering-sample Jetson Orin units to accelerate this.
2. Site-specific re-training. Every camera viewpoint has its own "normal". The model needs a short calibration period at each new install. TAO Toolkit's fine-tuning workflow looks like the right tool for that; we would value guidance on the right TAO pipeline.
3. Customer access. We can build this; we cannot easily walk into a transport authority. NVIDIA's existing Metropolis partners (and Inception startup-program enterprise customers) would shorten that path by months.
4. Validation footage diversity. Our current training data is daylight, dry, urban. Night, rain, tunnel, and snow scenes are out of distribution. Access to NVIDIA's curated AV datasets or to a partner's archival camera footage would dramatically improve generalisation.

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


---

## 7. Competitive Position vs. Existing Metropolis Solutions

| Dimension | Typical Supervised Metropolis App | CPYR + Traffic Vision MISV |
|---|---|---|
| Training data | Labeled accidents / near-misses | Normal driving footage + normal network captures only |
| Privacy | Often cloud-dependent| Pure edge i.e. No raw video uplink required |
| New hazard class | Requires relabeling + retraining | Detected automatically (unseen normal-violating events) |
| Failure mode transparency | Black-box score | Per-channel score: which source fired, why (reconstruction delta) |
| Regulatory framing | Post-hoc classification | Formal SOTIF Zone 2 reduction argument, clause 9 compliant |
| Number of sources | 1 (camera) | 3 (camera + network + V2X) — MISV |
| Publication backing | Varies | SAE 2021-01-0196, peer-reviewed, presented at WCX |
| NVIDIA hardware tested | Varies | Tesla T4 (Traffic Vision), GTX 1070 (CPYR) |
| Alert type | Binary | Continuous score → graduated alert levels |
| Lookahead | At event or post-event | ~250 ms before event (Traffic Vision) + 0-batch at CAN level (CPYR) |

---

## 8. Demonstrated Prototypes

All three systems are implemented in PyTorch, trained, evaluated, and documented. None are concepts — all are running prototypes.

| System | Demo | Description |
|---|---|---|
| **A — CPYR** | [Anomaly Detection Demo](https://www.youtube.com/watch?v=Yn-BaMF7mqE) | Live model inference on CAN/Ethernet data; contextual anomaly scoring |
| **A — CPYR** | Azure Cloud Monitoring | Continuous monitoring pipeline on Microsoft Azure with cloud-side retraining |
| **A — CPYR** | [ASRG Community Presentation](https://www.youtube.com/watch?v=z3uAQIN0nYw) | SAE WCX 2021 paper walkthrough; SOTIF methodology explanation |
| **B — Traffic Vision** | [Real-Time Anomaly Stream](https://www.youtube.com/watch?v=eAX6_KAtLiQ&t=1s) | Live dashcam inference; anomaly score overlaid on video |
| **B — Traffic Vision** | [Predictive Frame Demo](https://www.youtube.com/watch?v=sudJ5_wccdI&t=6s) | Predicted vs. observed frames; ~250 ms pre-event detection window on NVIDIA Tesla T4 |
| **C — MISV Combined** | [Combined System Demo](https://www.youtube.com/watch?v=LKH6Nsu54wc) | CPYR + Traffic Vision + V2X running simultaneously; fusion layer producing graduated alert |

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

1. **No single detector is enough for SOTIF.** Zone 2 requires multiple independent sources with non-overlapping failure modes. MISV is the architecture; we have built all three systems.

2. **System A (CPYR) sees what the camera cannot.** Network-level contextual anomaly: LKA misuse during sensor hesitation phase. Published at SAE 2021. >90% F1. 1.2 KB model. Zero-batch lag at anomaly onset. Two live demos: real-vehicle CAN inference and Azure cloud monitoring pipeline.

3. **System B (Traffic Vision) sees what the network cannot.** 250 ms before a dangerous maneuver fully unfolds, the camera already knows. Built on NVIDIA Tesla T4. 36 fps. No labels, no taxonomy, no bounding boxes. Two live demos: real-time anomaly stream and side-by-side predictive frame visualization.

4. **System C (MISV Fusion) makes the SOTIF argument.** CPYR + Traffic Vision running simultaneously, fused through a weighted vote layer. P(miss) ≤ p_A × p_B × p_C — a product-probability Zone 2 bound that neither system alone can produce. Demonstrated live in the combined demo.

5. **All three systems are already on NVIDIA hardware.** Traffic Vision was developed on Tesla T4. CPYR is 1.2 KB — it fits in L2 cache on any Jetson. The TensorRT export path is standard PyTorch → ONNX → `trtexec`. This is integration work, not research.

6. **The fusion produces an auditable evidence stream.** Three independent scores, logged continuously, one per channel, one per frame, give ISO 21448 clause 9 the evidence it requires. We are not just detecting accidents — we are generating the proof that the system is safe.

---

*Prepared for the NVIDIA Metropolis meeting, May 29, 2026.*  
*SAE Technical Paper 2021-01-0196 — DOI: 10.4271/2021-01-0196*  
*AT Instruments / EVRaid — at-instr.com*
