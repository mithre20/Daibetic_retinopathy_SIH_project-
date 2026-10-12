# AI DR Screen Chatbot — Knowledge Base Structure & Updates

## 1. Directory Structure
The knowledge base resides in `knowledge/` and is divided into modular subject folders:

```
knowledge/
├── general/
│   ├── about_aidr.md          # Overview, mission, target users, device support
│   └── faq.md                 # Frequently asked questions
├── image/
│   ├── retinal_upload.md      # File formats, upload process, selfie vs fundus
│   ├── image_quality.md       # MobileNetV3 IQA gatekeeper, rejection reasons
│   └── capture_guidance.md    # Field photography tips (steady camera, glare)
├── screening/
│   ├── dr_grades.md           # The 5 clinical DR grades (0 to 4), EfficientNet
│   └── confidence.md          # Softmax confidence, low-confidence rules
├── explainability/
│   ├── gradcam.md             # Gradient-weighted Class Activation Mapping
│   ├── shap.md                # Positive and negative pixel attributions
│   ├── unet.md                # Microaneurysm, hemorrhage, exudate masks
│   └── comparison.md          # Conceptual comparison of XAI methods
├── medical_report/
│   ├── reports_ocr.md         # PDF/image OCR analysis, manual override
│   └── glucose_bp.md          # Glycemic indices (FBS, PPBS, HbA1c) and BP
├── referral/
│   └── referral_engine.md     # Deterministic triage rules, high-severity safety
├── navigation/
│   └── website_navigation.md  # 5 stations overview, keyboard shortcuts
├── healthcare/
│   └── healthcare_centers.md  # Verified referral registry and anti-hallucination
├── privacy/
│   └── privacy_security.md    # Data confidentiality, ABDM alignment
├── safety/
│   └── medical_safety.md      # Clinical boundaries and emergency symptoms
└── technical/
    └── troubleshooting.md     # Common runtime issues, server reconnection
```

## 2. Adding or Updating Knowledge Documents
1. **File Format**: Standard GitHub Flavored Markdown (`.md`).
2. **Structure**: Use level-2 (`##`) and level-3 (`###`) headers for specific questions and topics. The chunker splits documents along these headers to keep question-answer blocks atomic.
3. **Re-indexing**: The vector store automatically re-indexes knowledge files when the server boots or when `get_vector_store().build_index()` is invoked.
