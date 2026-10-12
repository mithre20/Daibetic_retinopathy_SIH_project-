# General Information About AI DR Screen

## What is AI DR Screen?
AI DR Screen (code-named XAIRIS) is an explainable artificial intelligence clinical decision-support and screening platform designed specifically for diabetic retinopathy (DR) triage and early detection in rural and semi-urban Primary Health Centers (PHCs) across India.

## What does AI DR Screen do?
AI DR Screen assists frontline healthcare workers, ophthalmic assistants, and medical officers in evaluating retinal fundus photographs. It checks optical image quality, detects diabetic retinopathy presence and severity, generates transparent visual and statistical explanations (Grad-CAM, SHAP, U-Net lesion segmentation), extracts biometric health markers from medical reports (glucose, HbA1c, blood pressure), and determines deterministic clinical referral recommendations according to standardized guidelines.

## Who can use AI DR Screen?
1. Community health workers (ASHAs) and mobile camp operators performing field retinal screening.
2. Ophthalmic technicians and optometrists at Primary Health Centers (PHCs) and Community Health Centers (CHCs).
3. Medical Officers and general physicians needing transparent AI triage support.
4. Patients seeking to understand their screening report, referral priority, and next clinical steps under the supervision of health staff.

## How does AI DR Screen work?
The workflow follows a 5-stage explainable triage pipeline:
1. **Patient & Scan Registration**: Basic patient identification, eye designation (OD: Right Eye, OS: Left Eye), and optical fundus photograph ingestion.
2. **Image Quality Assessment (IQA)**: Automated quality gatekeeper (MobileNetV3) evaluating focus, clarity, illumination, and retinal anatomical visibility. If quality is poor or unusable, screening is immediately halted to prevent diagnostic error.
3. **Deep Learning DR Severity Classification**: EfficientNet-B0 (and ResNet-50) neural networks classify retinal scans into 5 international clinical severity grades (Grades 0 to 4).
4. **Multi-Model Explainable AI (XAI)**:
   - **Grad-CAM**: Generates coarse gradient-weighted class activation heatmaps showing where the classifier focused.
   - **SHAP (Expected Gradients)**: Quantifies positive (disease-driving) and negative (healthy-confirming) pixel attributions.
   - **U-Net Segmentation**: Produces pixel-level spatial masks identifying 4 hallmark lesions: Microaneurysms, Hemorrhages, Hard Exudates, and Soft Exudates (Cotton Wool Spots).
5. **Deterministic Clinical Referral Engine**: Deterministic rules map the DR grade, model confidence, and extracted lab reports to standard risk levels (Low, Moderate, High) and referral actions (Routine, Recommended, Priority).

## Is AI DR Screen a diagnostic system?
No. AI DR Screen is an automated clinical decision-support and screening triage tool. It does **NOT** provide a definitive medical diagnosis, nor does it replace comprehensive dilated ophthalmoscopy by a certified ophthalmologist. All screening recommendations must be confirmed by an eye specialist.

## Is it free?
Yes, the software is deployed as a public health decision-support utility for rural outreach camps and government primary health centers under national health digital initiatives.

## How do I start screening?
1. Navigate to the **AI Pipeline** tab (Station 05) or **Dashboard** (Station 03).
2. Upload a digital retinal fundus photograph (PNG, JPG, JPEG) or drag and drop it into the upload dropzone.
3. (Optional) Upload an accompanying laboratory report or enter glucose and blood pressure values.
4. Click **Run Screening Pipeline** to initiate automated quality assessment, severity grading, explainability generation, and referral triage.

## Can I use it on a mobile phone or tablet?
Yes. The platform is engineered with responsive, touch-friendly interfaces suitable for modern mobile browsers, clinical tablets used in mobile screening vans, and desktop workstations.

## Can I use it on a laptop?
Yes. It works seamlessly across laptops and PCs running Windows, macOS, Linux, or ChromeOS.

## Which browsers are supported?
Modern standards-compliant web browsers are supported, including Google Chrome, Mozilla Firefox, Microsoft Edge, Safari, and Chromium-based mobile browsers. WebGL must be enabled for 3D visual modules.

## How long does screening take?
On modern CPU or GPU-enabled workstations, the complete end-to-end pipeline (IQA, classification, Grad-CAM, SHAP, and U-Net segmentation) takes approximately 1 to 3 seconds per retinal scan.

## Do I need an account?
Staff operators log in using their Primary Health Center credentials or demo accounts on the **Login** tab (Station 02). Patients do not require personal login credentials because screenings are conducted by designated healthcare workers.

## What does the chatbot do?
The AI DR Screen Assistant is an intelligent, context-aware companion that helps users navigate the application, understand retinal upload requirements, interpret IQA results, explain DR grades, understand AI heatmaps and segmentations, clarify lab report metrics, and explain referral decisions using deterministic clinical rules.

## What can the chatbot help me with?
- Guiding you through the 5-stage screening pipeline.
- Explaining why a retinal image was accepted or rejected by the IQA gate.
- Explaining DR grades (0 to 4) and what confidence percentages mean.
- Demystifying Grad-CAM, SHAP attributions, and U-Net lesion masks.
- Explaining lab parameters like FBS, RBS, PPBS, and HbA1c.
- Navigating to specific pages or tabs in the platform.
- Answering technical questions, data privacy questions, and providing safe healthcare guidance.
