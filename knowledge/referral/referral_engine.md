# Deterministic Clinical Referral Engine and Screening Results

## What is the Referral Engine?
The referral engine (`decision/referral_engine.py` and `decision/referral_rules.py`) is an auditable, deterministic, rule-based clinical triage system. It translates AI disease predictions, confidence metrics, and image quality status into standardized risk categories and actionable clinical recommendations.

## Deterministic Referral Rules
According to the screening triage rules configured in AI DR Screen:

### Grade 0 (No Diabetic Retinopathy)
- **Risk Level**: Low Risk
- **Referral Priority**: Routine
- **Recommendation**: Routine follow-up
- **Clinical Action**: Annual routine retinal screening recommended as per diabetes standard of care.

### Grade 1 (Mild Non-Proliferative DR)
- **Risk Level**: Moderate Risk
- **Referral Priority**: Recommended
- **Recommendation**: Eye examination recommended
- **Clinical Action**: Comprehensive eye examination by an optometrist or ophthalmologist within 6 to 12 months.

### Grade 2 (Moderate Non-Proliferative DR)
- **Risk Level**: Moderate Risk
- **Referral Priority**: Recommended
- **Recommendation**: Eye examination recommended
- **Clinical Action**: Comprehensive ophthalmic examination within 3 to 6 months to evaluate progression and macular status.

### Grade 3 (Severe Non-Proliferative DR)
- **Risk Level**: High Risk
- **Referral Priority**: Priority
- **Recommendation**: Priority ophthalmologist referral
- **Clinical Action**: Prompt referral to a retinal specialist or ophthalmologist within 2 to 4 weeks for clinical staging and intervention.

### Grade 4 (Proliferative Diabetic Retinopathy)
- **Risk Level**: High Risk
- **Referral Priority**: Priority
- **Recommendation**: Priority ophthalmologist referral
- **Clinical Action**: Urgent referral to an ophthalmologist or vitreoretinal specialist for immediate intervention (e.g. panretinal photocoagulation laser or anti-VEGF therapy).

## What Actions Can I Take After Screening?
Within the existing application, you can:
1. **View Visual Explanations**: Inspect Grad-CAM heatmaps, positive/negative SHAP feature plots, and U-Net lesion segmentation masks on the Results panel.
2. **Download Clinical Report**: Click the "Download Report" button on the Results panel to export the structured screening findings and explainability graphics.
3. **Start a New Screening**: Click "New Screening" or "Change Image" to clear the session and evaluate another scan.
4. **Switch Between Views**: Use the master navigation bar tabs (Hero, Login, Dashboard, MediVision, AI Pipeline) or press Alt+1 through Alt+5.

## What Should I Do After Receiving a High Risk Result?
If your screening result indicates High Risk (Grade 3 or Grade 4):
- Do not panic.
- Schedule an in-person dilated eye examination with an ophthalmologist or retinal specialist promptly.
- Take your printed screening report and recent blood work to your appointment.
- Prompt medical care can prevent permanent vision loss.
