# MATLAB / Simulink Telemedicine Simulation Module

## Overview
This module provides a rigorous, multi-stage discrete-event and continuous-time queuing model of the **AI DR Screen** telemedicine pipeline for rural India.

The model evaluates system feasibility, queue stability, bottleneck identification, and resource optimization for large-scale rural screening programs processing **≥ 100,000 patients/year**.

---

## Directory Structure

```text
matlab/
├── simulink/
│   ├── ai_dr_screening.slx        # Executable Simulink model (Open Packaging Convention)
│   ├── parameters.m               # Configurable simulation parameters & clinical assumptions
│   ├── initialize_simulation.m    # Workspace setup and stability verification
│   ├── run_simulation.m           # Single-run queuing engine (M/M/c, Erlang-C, bottlenecks)
│   ├── run_scenarios.m            # Automated execution of Scenarios A, B, C, D, and E
│   ├── analyze_results.m          # Post-processing and performance KPI reporting
│   ├── optimize_resources.m       # Constrained resource optimization (cameras, AI, doctors, bandwidth)
│   └── export_results.m           # CSV and JSON results exporter
│
├── data/
│   ├── simulation_parameters.csv  # Parameter dictionary with units and rationale
│   └── benchmark_data.csv         # Published literature benchmark comparison data
│
└── results/                       # Simulation output artifacts
    ├── throughput/                # Latency & throughput logs
    ├── bandwidth/                 # Bandwidth sweep curves
    ├── queue/                     # Backlog & queue statistics
    ├── resource_optimization/     # Feasible and optimal allocations
    └── figures/                   # Generated publication figures
```

---

## Telemedicine Pipeline Workflow

The Simulink model (`ai_dr_screening.slx`) connects the following stages:

```text
[Patient Arrival Stream] (Poisson arrival, 100k+ pts/year)
          │
          ▼
[Camera Acquisition Queue] (Bilateral fundus photography, 45s/image)
          │
          ▼
[Network Uplink Transmission] (Bandwidth constraint: 0.5 – 50 Mbps)
          │
          ▼
[Image Quality Assessment] (MobileNetV3-Small gatekeeper, 11.4 ms)
     ├── Usable (82%) ──────────────────────────────┐
     └── Unusable (18%) ──> [Recapture Loop] ───────┤
                                                    │
                                                    ▼
[Automated AI Pipeline] (EfficientNet-B0 + U-Net + Grad-CAM, 457 ms)
          │
          ▼
[Referral Decision Engine] (Confidence-aware clinical triage, <0.1 ms)
     ├── Non-referable (82%) ──> [Auto-Cleared / Routine Follow-up]
     └── Referable DR (18%)   ──> [Ophthalmologist Telemedicine Pool]
                                                    │
                                                    ▼
                                     [Human Specialist Decision]
```

---

## How to Run in MATLAB

1. **Initialize Parameters**:
   ```matlab
   cd matlab/simulink
   initialize_simulation();
   ```

2. **Open & Run the Simulink Model**:
   ```matlab
   open_system('ai_dr_screening.slx');
   simOut = sim('ai_dr_screening', 'StopTime', '28800'); % 8-hour screening day
   ```

3. **Run All Scenarios (A through E)**:
   ```matlab
   scenarios = run_scenarios();
   ```

4. **Execute Resource Optimization**:
   ```matlab
   opt = optimize_resources('AnnualPatients', 100000, 'MaxWaitMinTarget', 15.0);
   ```

5. **Export Deliverables**:
   ```matlab
   res = run_simulation('AnnualPatients', 100000);
   export_results(res);
   ```

---

## Companion Python Execution

For systems without MATLAB installed, an identical mathematical discrete-event simulation engine is provided in `validation/simulate_telemedicine.py`:
```bash
python validation/simulate_telemedicine.py
```
This produces identical results and populates `matlab/results/` and root `results/`.
