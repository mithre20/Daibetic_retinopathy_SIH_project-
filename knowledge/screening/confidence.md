# AI Confidence Scores and Low-Confidence Handling

## What is Model Confidence?
Model confidence is a probability score ranging between 0.0 (0%) and 1.0 (100%) computed via the softmax layer of the neural network. It quantifies how certain the model is that the retinal image matches the assigned Diabetic Retinopathy severity grade compared to the alternative grades.

## What is the Minimum Confidence Threshold?
The application uses an engineering threshold of **50% (0.50)** minimum confidence for DR predictions.
- **Normal / High Confidence (>= 50%)**: The model has clear statistical separation favoring the predicted grade.
- **Low Confidence (< 50%)**: The probabilities are distributed across multiple grades, indicating diagnostic ambiguity.

## Can the AI Make Mistakes?
Yes. Deep learning neural networks are statistical pattern recognizers. An AI model can produce false positives (flagging non-pathological variations as lesions) or false negatives (overlooking faint microaneurysms in peripheral areas). This is why explainable AI visualizations (Grad-CAM, SHAP, U-Net) and deterministic safety referral rules are incorporated to enable human verification.

## How are Low-Confidence Results Handled?
The application applies strict deterministic clinical safety rules whenever an AI prediction falls below the minimum confidence threshold:

### CRITICAL SAFETY RULE: High-Severity Grades Must NEVER Be Downgraded
If an image is classified as **Grade 3 (Severe DR)** or **Grade 4 (Proliferative DR)** with low confidence (e.g. 42% confidence):
1. The system **NEVER** downgrades the risk level to Low or Moderate.
2. The triage priority **MUST ALWAYS REMAIN High Risk / Priority Referral**.
3. The clinical action remains an urgent in-person ophthalmology examination.
4. The system flags the prediction as `LOW_CONFIDENCE` and alerts the healthcare worker:
   "The AI prediction confidence is limited. Because potential high-severity diabetic retinopathy was detected, professional in-person clinical review is especially critical."

### What happens for Low-Severity Low-Confidence Predictions?
For Grade 0, 1, or 2 with low confidence, the referral engine flags the uncertainty and recommends professional clinical evaluation rather than assuming the eye is completely normal.

## Why should I consult an eye specialist?
An automated screening scan provides early triage warning, but only an eye specialist can examine the eye in 3D using a slit-lamp biomicroscope, assess intraocular pressure, verify peripheral retinal health, and prescribe clinical treatment.
