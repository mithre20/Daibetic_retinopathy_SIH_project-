# Diabetic Retinopathy Severity Grading

## How does AI detect Diabetic Retinopathy?
AI detects diabetic retinopathy by analyzing high-resolution digital retinal photographs using deep convolutional neural networks. The models detect subtle microvascular pathology including dot-and-blot hemorrhages, microaneurysms, lipid exudates, cotton-wool spots (infarcts), and neovascularization (abnormal fragile new blood vessels).

## What is EfficientNet?
EfficientNet-B0 is an advanced, highly efficient convolutional neural network architecture that balances network depth, width, and input image resolution through compound scaling. It achieves state-of-the-art diagnostic accuracy while remaining lightweight enough to perform real-time screening on standard edge computers without needing multi-thousand-dollar GPU clusters.

## Why is EfficientNet used?
1. Exceptional feature extraction capability for fine microvascular structures.
2. Low computational overhead, allowing rapid screening in rural field clinics.
3. Proven transfer learning performance on standardized benchmark ophthalmic datasets (e.g. EyePACS, APTOS 2019, Messidor).

## The Five International Clinical Diabetic Retinopathy (ICDR) Grades
The platform classifies scans into 5 standardized clinical severity grades:

### Grade 0: No DR (No Diabetic Retinopathy)
- **Clinical Meaning**: The retina shows no visible microvascular diabetic abnormalities. Retinal vessels, fovea, and optic disc appear healthy.
- **Risk Level**: Low Risk.
- **Recommendation**: Routine follow-up (annual retinal examination as per diabetes standard of care).

### Grade 1: Mild DR (Mild Non-Proliferative Diabetic Retinopathy - NPDR)
- **Clinical Meaning**: Only microaneurysms (tiny balloon-like outpouchings of capillary walls) are present in the retina.
- **Risk Level**: Moderate Risk.
- **Recommendation**: Eye examination recommended by an optometrist or ophthalmologist within 6 to 12 months.

### Grade 2: Moderate DR (Moderate Non-Proliferative Diabetic Retinopathy - NPDR)
- **Clinical Meaning**: More microaneurysms, dot hemorrhages, venous beading, or hard exudates are visible, but less extensive than Severe NPDR.
- **Risk Level**: Moderate Risk.
- **Recommendation**: Comprehensive ophthalmic examination within 3 to 6 months to monitor disease trajectory and evaluate macular edema status.

### Grade 3: Severe DR (Severe Non-Proliferative Diabetic Retinopathy - NPDR)
- **Clinical Meaning**: Follows the clinical 4:2:1 rule: severe intraretinal hemorrhages in all 4 quadrants, definite venous beading in 2+ quadrants, or prominent intraretinal microvascular abnormalities (IRMA) in 1+ quadrant, without frank neovascularization. High danger of vision-threatening progression.
- **Risk Level**: High Risk.
- **Recommendation**: Priority ophthalmologist referral within 2 to 4 weeks for clinical staging and intervention planning.

### Grade 4: Proliferative DR (PDR)
- **Clinical Meaning**: Advanced stage characterized by active neovascularization (fragile new abnormal blood vessels growing on the retina or optic disc), vitreous/preretinal hemorrhage, or fibrovascular traction that can cause retinal detachment and irreversible blindness.
- **Risk Level**: High Risk.
- **Recommendation**: Priority/Urgent ophthalmologist referral for immediate ophthalmic intervention (e.g. panretinal photocoagulation [PRP laser], anti-VEGF injections, or vitreoretinal surgery).

## Is the AI result a medical diagnosis?
**No.** The AI screening result is an automated triage classification to assist clinical decision-making. It is not a definitive diagnosis. Definitive diagnosis requires direct dilated stereoscopic examination by a licensed ophthalmologist.
