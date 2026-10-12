# SHAP (SHapley Additive exPlanations)

## What is SHAP?
SHAP (SHapley Additive exPlanations) is a game-theoretic framework for explaining machine learning outputs. In AI DR Screen, SHAP computes pixel-wise Shapley values using `shap.GradientExplainer` (Expected Gradients) to determine how individual image features push the prediction toward or away from a specific DR grade.

## Why is SHAP used alongside Grad-CAM?
While Grad-CAM only shows where the model looked generally, SHAP separates visual evidence into **positive contributions** (features increasing the likelihood of disease) and **negative contributions** (features reducing the likelihood of disease or confirming healthy tissue).

## What are Positive Contributions?
Positive contributions (highlighted in red or warm tones) are pixel regions whose presence increased the model's confidence in the predicted DR grade. For instance, dark hemorrhagic spots or bright lipid deposits pushing the prediction toward Moderate or Severe DR.

## What are Negative Contributions?
Negative contributions (highlighted in blue or cool tones) are pixel regions whose presence decreased the likelihood of that grade, or provided evidence against higher disease severity (such as clear, healthy macula or intact vessel margins).

## Does SHAP diagnose diabetic retinopathy?
**NO.** SHAP explains the mathematical feature contribution to the deep learning model's output. It does not replace clinical pathology analysis and should not be cited as direct medical proof unless verified by clinical ophthalmic evaluation.

## What is Sparsity in SHAP?
Sparsity measures what fraction of the retinal scan actively drove the AI decision. High sparsity indicates the model made its prediction based on isolated focal lesions, whereas low sparsity indicates widespread background changes across the retina.
