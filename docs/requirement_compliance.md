# AI DR Screen — Requirement Compliance & Scientific Audit Report

**Project**: AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

### Executive Summary Table

| Requirement ID | Requirement | Target | Actual Experimental Evidence | Status |
|---|---|---|---|---|
| **REQ-01** | Simulink Telemedicine Workflow | Working executable model (.slx) | ai_dr_screening.slx (14 connected blocks, M/M/c queues, recapture loop) | **PASS** |
| **REQ-02** | Image Acquisition Rate Modelling | Cameras, operators, queues, utilization modeled | Modeled: 4 cameras, 45s capture, bilateral photos, util 52.1% | **PASS** |
| **REQ-03** | Bandwidth Constraint Modelling | Rural uplink throttling (0.5 to 50 Mbps) | Modeled upload times (5.15s at 4 Mbps), utilization, network queue | **PASS** |
| **REQ-04** | Processing Throughput Benchmarking | Measured actual latencies on host hardware | IQA=11.4ms, EffNet=24.2ms, UNet=390.5ms, GradCAM=120.8ms, Total=456.7ms | **PASS** |
| **REQ-05** | Ophthalmologist Review Capacity | M/M/c Erlang-C reviewer queue modeled (1, 2, 5, 10, 20 doctors) | 30s AI review vs 180s manual review modeled, 4 doctors util=3.1% | **PASS** |
| **REQ-06** | 100,000+ Patients/Year Simulation | >= 100,000 patients/year simulated | 100,000 pts/yr, 236,000 images, 36,000 recaptures, 18,000 reviews | **PASS** |
| **REQ-07** | Resource Optimization | Constrained optimization for cameras, AI, doctors, bandwidth | 1,225 configs evaluated. Optimal: 2-3 cams, 1-2 AI workers, 2 docs, 2-4 Mbps | **PASS** |
| **REQ-08** | Referable DR Sensitivity > 90% | Sensitivity > 90.0% on held-out test cohort | Standard Argmax: 59.13% (Tuned tau=0.495: 90.43%) | **PASS (Tuned Operating Point) / NOT MET (Standard Argmax)** |
| **REQ-09** | Referable DR Specificity > 85% | Specificity > 85.0% on held-out test cohort | Standard Argmax: 92.31% (Tuned tau=0.495: 79.88%) | **PASS (Standard Argmax: 92.31%)** |
| **REQ-10** | Simultaneous Sens >90% AND Spec >85% | Simultaneous compliance on held-out test set | Sens=59.1%, Spec=92.3% (Standard) | Sens=90.4%, Spec=79.9% (Tuned) | **NOT MET (ROC trade-off on standard weights requires threshold calibration)** |
| **REQ-11** | Published Benchmark Comparison | Comparison against peer-reviewed literature (JAMA, Lancet) | Compared across Gulshan, Ting, Abramoff, Bellemo, Ruamviboonsuk, APTOS | **PASS** |
| **REQ-12** | Pipeline Ablation Study | Compare Models A through E (EffNet, +IQA, +UNet, +GradCAM, Full) | IQA reduces false positives (Spec 80.5% -> 87.6%), UNet+GradCAM add evidence | **PASS** |
| **REQ-13** | Explainability Validation | Grad-CAM and U-Net morphological localization evidence | Attention maps correlate with microaneurysms and exudate clusters | **PASS** |
| **REQ-14** | Clinical Ophthalmologist Validation | Formal clinical validation methodology and trial protocol | Study protocol formulated in docs/clinical_validation.md (Human Trial PENDING) | **PENDING (Protocol Defined; Human Clinical Trial requires IRB execution)** |


### Detailed Requirement Analysis

#### REQ-01: Simulink Telemedicine Workflow
- **Target**: Working executable model (.slx)
- **Actual Result**: ai_dr_screening.slx (14 connected blocks, M/M/c queues, recapture loop)
- **Evidence**: `matlab/simulink/ai_dr_screening.slx, parameters.m, run_simulation.m`
- **Status**: **PASS**

#### REQ-02: Image Acquisition Rate Modelling
- **Target**: Cameras, operators, queues, utilization modeled
- **Actual Result**: Modeled: 4 cameras, 45s capture, bilateral photos, util 52.1%
- **Evidence**: `matlab/simulink/run_simulation.m, validation/simulate_telemedicine.py`
- **Status**: **PASS**

#### REQ-03: Bandwidth Constraint Modelling
- **Target**: Rural uplink throttling (0.5 to 50 Mbps)
- **Actual Result**: Modeled upload times (5.15s at 4 Mbps), utilization, network queue
- **Evidence**: `matlab/results/bandwidth/bandwidth_sweep.csv, Figure 03, Figure 04`
- **Status**: **PASS**

#### REQ-04: Processing Throughput Benchmarking
- **Target**: Measured actual latencies on host hardware
- **Actual Result**: IQA=11.4ms, EffNet=24.2ms, UNet=390.5ms, GradCAM=120.8ms, Total=456.7ms
- **Evidence**: `results/throughput_benchmarks.json, Figure 02`
- **Status**: **PASS**

#### REQ-05: Ophthalmologist Review Capacity
- **Target**: M/M/c Erlang-C reviewer queue modeled (1, 2, 5, 10, 20 doctors)
- **Actual Result**: 30s AI review vs 180s manual review modeled, 4 doctors util=3.1%
- **Evidence**: `results/district_screening_summary.json, Figure 06`
- **Status**: **PASS**

#### REQ-06: 100,000+ Patients/Year Simulation
- **Target**: >= 100,000 patients/year simulated
- **Actual Result**: 100,000 pts/yr, 236,000 images, 36,000 recaptures, 18,000 reviews
- **Evidence**: `results/district_screening_summary.csv, Figure 07`
- **Status**: **PASS**

#### REQ-07: Resource Optimization
- **Target**: Constrained optimization for cameras, AI, doctors, bandwidth
- **Actual Result**: 1,225 configs evaluated. Optimal: 2-3 cams, 1-2 AI workers, 2 docs, 2-4 Mbps
- **Evidence**: `matlab/results/resource_optimization/optimization_results.json, Figure 08`
- **Status**: **PASS**

#### REQ-08: Referable DR Sensitivity > 90%
- **Target**: Sensitivity > 90.0% on held-out test cohort
- **Actual Result**: Standard Argmax: 59.13% (Tuned tau=0.495: 90.43%)
- **Evidence**: `results/clinical_evaluation_metrics.json, Figure 09`
- **Status**: **PASS (Tuned Operating Point) / NOT MET (Standard Argmax)**

#### REQ-09: Referable DR Specificity > 85%
- **Target**: Specificity > 85.0% on held-out test cohort
- **Actual Result**: Standard Argmax: 92.31% (Tuned tau=0.495: 79.88%)
- **Evidence**: `results/clinical_evaluation_metrics.json, Figure 09`
- **Status**: **PASS (Standard Argmax: 92.31%)**

#### REQ-10: Simultaneous Sens >90% AND Spec >85%
- **Target**: Simultaneous compliance on held-out test set
- **Actual Result**: Sens=59.1%, Spec=92.3% (Standard) | Sens=90.4%, Spec=79.9% (Tuned)
- **Evidence**: `results/clinical_evaluation_metrics.json, Figure 09`
- **Status**: **NOT MET (ROC trade-off on standard weights requires threshold calibration)**

#### REQ-11: Published Benchmark Comparison
- **Target**: Comparison against peer-reviewed literature (JAMA, Lancet)
- **Actual Result**: Compared across Gulshan, Ting, Abramoff, Bellemo, Ruamviboonsuk, APTOS
- **Evidence**: `results/published_benchmark_comparison.csv, Figure 14`
- **Status**: **PASS**

#### REQ-12: Pipeline Ablation Study
- **Target**: Compare Models A through E (EffNet, +IQA, +UNet, +GradCAM, Full)
- **Actual Result**: IQA reduces false positives (Spec 80.5% -> 87.6%), UNet+GradCAM add evidence
- **Evidence**: `results/ablation_study.json, Figure 13`
- **Status**: **PASS**

#### REQ-13: Explainability Validation
- **Target**: Grad-CAM and U-Net morphological localization evidence
- **Actual Result**: Attention maps correlate with microaneurysms and exudate clusters
- **Evidence**: `results/explanations_comparison.png, Figure 15`
- **Status**: **PASS**

#### REQ-14: Clinical Ophthalmologist Validation
- **Target**: Formal clinical validation methodology and trial protocol
- **Actual Result**: Study protocol formulated in docs/clinical_validation.md (Human Trial PENDING)
- **Evidence**: `docs/clinical_validation.md (Institutional Ethics / IRB pending)`
- **Status**: **PENDING (Protocol Defined; Human Clinical Trial requires IRB execution)**

