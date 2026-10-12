# Reproducibility & Environment Specification Guide

**Project**: AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India  
**Document Version**: 2.6

---

## 1. Environment & Hardware Stack

- **Operating System**: Microsoft Windows 11 Pro (64-bit)
- **Python Version**: Python 3.13.14 (Virtual Environment `.venv`)
- **PyTorch Version**: PyTorch 2.14.0+cpu / torchvision 0.19.0
- **MATLAB / Simulink**: MATLAB R2023b / R2024b compatible (`.slx` format version 10.0)
- **Key Python Libraries**:
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `scikit-learn >= 1.3.0`
  - `matplotlib >= 3.7.0`
  - `pillow >= 10.0.0`
  - `flask >= 3.0.0`

---

## 2. Model Checkpoints & Pretrained Weights

All neural network weights are persisted locally in `models/`:
- **EfficientNet-B0 DR Classifier**: `models/efficientnet_b0_dr.pth` (16.4 MB)
- **ResNet-50 Comparison Classifier**: `models/resnet50_dr.pth` (94.4 MB)
- **IQA MobileNetV3-Small**: `models/retinal_quality_mobilenet.pth` (6.2 MB)
- **U-Net Retinal Lesion Segmentation**: `models/unet_model.py` (Architecture weights initialized)

---

## 3. Dataset Configuration & Held-Out Test Cohort

- **Location**: `DR_Screening_Pipeline/DR_Screening_Pipeline/data/test/`
- **Sample Count**: 284 clinical fundus photographs
- **Labels File**: `_classes.csv`
- **Class Breakdown**:
  - `No_DR`: Grade 0
  - `Mild`: Grade 1
  - `Moderate`: Grade 2 (Referable)
  - `Severe`: Grade 3 (Referable)
  - `Proliferate_DR`: Grade 4 (Referable)

---

## 4. Master Reproduction Commands

To re-run the entire experimental and simulation pipeline from scratch:

```bash
# 1. Measure actual hardware latencies & throughput
.\.venv\Scripts\python.exe validation/benchmark_throughput.py

# 2. Evaluate held-out test cohort clinical metrics
.\.venv\Scripts\python.exe validation/clinical_metrics.py

# 3. Simulate 100,000+ patients/year telemedicine pipeline
.\.venv\Scripts\python.exe validation/simulate_telemedicine.py

# 4. Generate literature benchmark comparison
.\.venv\Scripts\python.exe validation/benchmark_comparison.py

# 5. Run 5-stage pipeline ablation study
.\.venv\Scripts\python.exe validation/ablation_study.py

# 6. Generate all 15 publication figures
.\.venv\Scripts\python.exe validation/generate_figures.py

# 7. Execute automated requirement compliance audit
.\.venv\Scripts\python.exe validation/requirement_audit.py
```

In MATLAB:
```matlab
cd matlab/simulink
initialize_simulation();
scenarios = run_scenarios();
opt = optimize_resources('AnnualPatients', 100000);
export_results(scenarios.C);
```
