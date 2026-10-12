# Website Navigation and Shortcuts

## Master Navigation Architecture
The AI DR Screen platform consists of 5 sequential stations accessible from the master top navigation bar:

1. **01 Hero Overview (`hero`)**: The cinematic introductory landing page featuring the 3D medical imaging visualization, system telemetry indicators, and platform capabilities.
2. **02 Staff Login (`login`)**: Authentication portal for Primary Health Center medical officers, camp technicians, and administrators.
3. **03 Screening Dashboard (`dashboard`)**: Clinic overview showing screening statistics (total screened, referable cases, normal scans), active patient queue, eye switcher (OD / OS), and thumbnail magnifier.
4. **04 MediVision Clinical Station (`medivision`)**: Advanced clinical inspection console with high-magnification fundus viewer, lesion toggle layers, and telemetry overlays.
5. **05 5-Stage AI Pipeline (`pipeline`)**: The interactive end-to-end deep learning screening station where image upload, IQA quality gating, EfficientNet grading, Grad-CAM, SHAP, and U-Net segmentation are executed.

## Keyboard Shortcuts
You can navigate between views rapidly using keyboard shortcuts:
- **Alt + 1**: Return to Page 01: Hero Overview
- **Alt + 2**: Open Page 02: Staff Login
- **Alt + 3**: Open Page 03: Main Screening Dashboard
- **Alt + 4**: Open Page 04: MediVision Clinical Station
- **Alt + 5**: Open Page 05: 5-Stage Explainable AI Pipeline
- **ArrowRight / PageDown**: Advance to the next station in sequence
- **ArrowLeft / PageUp**: Return to the previous station

## How to Switch Views Programmatically via Chatbot
You can ask the assistant:
- "Take me to the screening pipeline" or "Open upload" -> switches to Station 05 (AI Pipeline).
- "Go to dashboard" -> switches to Station 03 (Dashboard).
- "Open login" -> switches to Station 02 (Staff Login).
- "Go to home" or "Open hero" -> switches to Station 01 (Hero Overview).
- "Open MediVision" -> switches to Station 04 (MediVision).
