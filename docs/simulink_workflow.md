# Telemedicine Screening Workflow & Simulink Modelling

**Project**: AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Model Name**: `ai_dr_screening.slx`  
**Target Screening Program**: ≥ 100,000 patients/year

---

## 1. System Architecture

The executable Simulink model (`matlab/simulink/ai_dr_screening.slx`) simulates the complete physical and telecommunication workflow of rural retinal screening:

```text
[Patient Arrival] ──> [Camera Acquisition Queue] ──> [Network Bandwidth Uplink]
                                                              │
                                                              ▼
[Ophthalmologist Pool] <── [Referral Switch] <── [Automated AI] <── [IQA Gatekeeper]
          │                     │                                         │
          ▼                     ▼                                         ▼
   [Human Review]       [Cleared / Routine]                       [Recapture Loop]
```

---

## 2. Mathematical Modeling & Queuing Theory

Each subsystem is modeled as an $M/M/c$ multi-server queue governed by the **Erlang-C delay formula**:

1. **Patient Arrival Stream**:
   $$\lambda_{\text{patients}} = \frac{100,000}{300 \times 8} = 41.67 \text{ patients/hour} = 0.01157 \text{ patients/sec}$$
   $$\lambda_{\text{images}} = 2 \times \lambda_{\text{patients}} = 0.02315 \text{ images/sec}$$

2. **Fundus Camera Acquisition**:
   - Service rate: $\mu_{\text{cam}} = \frac{1}{45.0\text{ s}} = 0.0222 \text{ images/sec/camera}$
   - Capacity: $c = 4$ cameras
   - Utilization: $\rho_{\text{cam}} = \frac{\lambda_{\text{images}}}{c \cdot \mu_{\text{cam}}} \approx 26.0\%$

3. **Rural Network Transmission**:
   - Compressed JPEG size: 2.5 MB (20 Megabits)
   - Transmission time: $t_{\text{up}} = \frac{20 \text{ Mb}}{4.0 \text{ Mbps}} + 0.15\text{s} = 5.15\text{ seconds}$
   - Concurrent uplink channels: $\lceil 4.0 / 2.0 \rceil = 2$ channels

4. **Image Quality Assessment (IQA) & Recapture**:
   - MobileNetV3-Small latency: $11.4\text{ ms}$
   - Usable rate: $82.0\%$ | Recapture rate: $18.0\%$
   - Additional acquisition & network load: $+18.0\%$

5. **Automated AI Pipeline**:
   - Total measured latency: $456.7\text{ ms}$ (EfficientNet: 24.2ms, U-Net: 390.5ms, Grad-CAM: 120.8ms)
   - Single-worker throughput: $3,941.3\text{ patients/hour}$
   - Worker utilization ($c = 2$ threads): $< 2.0\%$

6. **Ophthalmologist Review Capacity**:
   - Referable DR cases requiring review: $18.0\%$ of patient cohort
   - $\lambda_{\text{rev}} = 0.18 \times 41.67 = 7.50\text{ cases/hour}$
   - Review time with AI report: $30\text{ seconds/case}$ ($\mu_{\text{rev}} = 120\text{ cases/hour/doctor}$)
   - Reviewer utilization ($c = 4$ doctors): $1.56\%$

---

## 3. Operational Scenario Comparison

| Scenario | Annual Volume | Cameras | Bandwidth | Doctors | Avg Wait (min) | Max Queue | Primary Bottleneck |
|---|---|---|---|---|---|---|---|
| **Scenario A: Small Rural PHC** | 5,000 | 1 | 1.0 Mbps | 1 | 0.10 | 0 | Camera Acquisition |
| **Scenario B: Medium PHC** | 25,000 | 2 | 4.0 Mbps | 2 | 0.03 | 0 | Camera Acquisition |
| **Scenario C: District Program** | 100,000 | 4 | 8.0 Mbps | 4 | 0.01 | 0 | Camera Acquisition |
| **Scenario D: Peak Mobile Camp** | 100,000 | 4 | 4.0 Mbps | 4 | 0.03 | 0 | Camera Acquisition |
| **Scenario E: Improved Infrastructure**| 100,000 | 5 | 15.0 Mbps | 5 | <0.01 | 0 | Camera Acquisition |

---

## 4. Resource Optimization Results

Grid search across 1,225 operational configurations determined the minimum viable deployment for $\ge 100,000\text{ patients/year}$:
- **Optimal Cameras**: 2 units (Utilization: 52.1%)
- **Optimal AI Workers**: 1 instance (Utilization: 1.2%)
- **Optimal Ophthalmologists**: 2 doctors (Utilization: 3.1%)
- **Optimal Bandwidth**: 2.0 Mbps (Utilization: 23.5%)
- **Expected Patient Wait Time**: 0.66 minutes
- **Annual Screening Volume Demonstrated**: **100,000 patients / year**
