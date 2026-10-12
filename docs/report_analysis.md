# Patient Health Report Analysis Module

**AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India**

---

## 1. Purpose & Clinical Context

Diabetic Retinopathy (DR) is a microvascular complication of chronic hyperglycemia and hypertension. In rural community vision camps across India, screening operators frequently encounter patients presenting not only with retinal fundus photographs, but also physical laboratory printouts, handwritten or printed medical reports, or smartphones with scanned lab documents.

The **Patient Health Report Analysis** module ingests lab reports (PDF, JPG, JPEG, PNG) or manual vitals entries, extracts blood glucose parameters and blood pressure, validates physiological plausibility, and presents this clinical context alongside deterministic DR screening results.

### Key Architectural Principles
1. **Clinical Context, NOT Retinal Prediction**: Blood pressure, blood glucose, and HbA1c are **never** predicted from fundus images. Retinal imaging grades retinal microvascular lesions; laboratory reports capture systemic metabolic markers.
2. **Zero Hallucination / Zero Fabrication**: If a vital sign or glucose metric is absent from the report, the engine reports `"not found"` (`None`). It **never** guesses or invents numbers.
3. **No Arbitrary Multi-Modal Formulas**: The module **never** calculates unvalidated synthetic risk scores (e.g. `DR + Glucose + BP = 87%`). Retinal DR grade and systemic vitals are presented transparently as complementary dimensions in a unified clinical summary.
4. **Mandatory Operator Verification**: Extracted parameters are presented to the healthcare operator with original units, normalized values, and reference ranges, allowing interactive editing and confirmation prior to final referral generation.
5. **Zero Disruption to Existing Models**: Operates independently without altering EfficientNet-B0, ResNet-50, U-Net, Grad-CAM, SHAP, IQA, or the Referral Decision Engine.

---

## 2. Architecture & Data Flow

```text
       ┌─────────────────────────────────────────────────────────────┐
       │             PATIENT HEALTH REPORT INPUT                     │
       │   - Digital PDF (PyPDF)                                     │
       │   - Scanned PDF / Image (EasyOCR / CRAFT + CRNN)            │
       │   - Direct Manual Clinician Entry                           │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             TEXT EXTRACTION & NORMALIZATION                 │
       │   - PyPDF text extraction for vector PDFs                   │
       │   - EasyOCR fallback for scans with encoding safeguards    │
       │   - Windows charmap & Unicode-safe output handling          │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             CONTEXTUAL REGEX PARSER                         │
       │   - Fasting, Postprandial, Random Blood Sugar, HbA1c        │
       │   - Composite BP (e.g., 130/85 mmHg) & Separate Sys/Dia     │
       │   - Laboratory Reference Ranges & Original Unit Capture     │
       │   - OCR error tolerance (e.g., 1 vs l, 0 vs O/o)            │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             UNIT CONVERSION & VALIDATION                    │
       │   - Dual unit conversion: mg/dL <--> mmol/L (x18.0182)      │
       │   - Physiological plausibility bounds checking              │
       │   - Non-alarmist screening interpretations                  │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             OPERATOR VERIFICATION STEP                      │
       │   - Highlight extracted vitals with source tags             │
       │   - Clinician edit & confirm toggle in UI                   │
       └──────────────────────────────┬──────────────────────────────┘
                                      │
                                      ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             UNIFIED SCREENING SUMMARY                       │
       │   - Validated DR Grade (EfficientNet-B0)                    │
       │   - Referral Urgency (Deterministic Rule-Based Engine)      │
       │   - Systemic Metabolic Context (Glucose, HbA1c, BP)         │
       │   - Explainability Maps (Grad-CAM, SHAP, U-Net)             │
       └─────────────────────────────────────────────────────────────┘
```

---

## 3. Extraction & Parsing Implementation

### Text Extraction (`report_analysis/extractor.py`)
- **PDF Extraction**: Uses `pypdf.PdfReader` to extract native text streams. If native text is absent (scanned PDF), pages are converted to images via `pypdf`/`PIL` and processed through OCR.
- **Image OCR**: Employs `easyocr.Reader` (`craft` text detector + deep recognizer) with explicit Windows console compatibility safeguards (`os.environ["PYTHONIOENCODING"] = "utf-8"` and suppressed progress print statements to prevent `cp1252` encoding exceptions).
- **Supported Formats**: `.pdf`, `.png`, `.jpg`, `.jpeg`. Unsupported file extensions are cleanly rejected with an informative error message.

### Contextual Regex Engine (`report_analysis/parser.py`)
The parser searches line-by-line and across sliding windows for standard clinical lab patterns:

| Metric | Target Keywords / Patterns | Example Detected Strings |
| :--- | :--- | :--- |
| **Fasting Glucose** | `fasting`, `FBS`, `FBG`, `glucose - fasting` | `Fasting Blood Glucose: 132 mg/dL` |
| **Postprandial** | `postprandial`, `post prandial`, `PPBS`, `PPBG` | `PPBS: 198 mg/dL (Ref: 70-140)` |
| **Random Glucose** | `random`, `RBS`, `blood glucose`, `sugar` | `Random Blood Sugar: 165 mg/dL` |
| **HbA1c** | `HbA1c`, `glycated hemoglobin`, `glycohemoglobin` | `HbA1c: 7.8 %` |
| **Blood Pressure** | `BP`, `blood pressure`, `systolic/diastolic`, `\d{2,3}\s*/\s*\d{2,3}` | `BP: 138/88 mmHg` |

---

## 4. Unit Conversion & Plausibility Validation

### Unit Conversion (`report_analysis/converter.py`)
- Formula: $\text{mg/dL} = \text{mmol/L} \times 18.0182$
- Bi-directional conversion ensures both metric systems are readily available for international standards while preserving original reported units.

### Physiological Bounds (`report_analysis/validator.py`)
To prevent dangerous false inputs or OCR artifacts from being accepted blindly:
- **Blood Glucose**: $20 \le x \le 1000\text{ mg/dL}$
- **HbA1c**: $3.0 \le x \le 25.0\%$
- **Systolic BP**: $60 \le x \le 260\text{ mmHg}$
- **Diastolic BP**: $40 \le x \le 160\text{ mmHg}$
- **BP Consistency**: Systolic must be strictly greater than Diastolic ($\text{Systolic} > \text{Diastolic}$).

Values outside these biological limits are flagged as `"Physiologically Implausible"` and prompt immediate operator review.

---

## 5. API Endpoints

### 1. `POST /api/analyze-report`
Ingests a medical report file (multipart/form-data) or raw text.
- **Request**: Multipart file (`file`) or JSON with `text`.
- **Response**:
```json
{
  "success": true,
  "filename": "lab_report.pdf",
  "data": {
    "glucose": {
      "fasting": {
        "parameter": "Fasting Blood Glucose",
        "value": 132.0,
        "unit": "mg/dL",
        "normalized_value": 132.0,
        "normalized_unit": "mg/dL",
        "value_mmol_l": 7.33,
        "reference_range": "70 - 99 mg/dL",
        "interpretation": "Elevated / Diabetes range",
        "is_plausible": true,
        "source": "Uploaded Report"
      }
    },
    "blood_pressure": {
      "systolic": 138,
      "diastolic": 88,
      "unit": "mmHg",
      "interpretation": "Stage 1 Hypertension",
      "is_plausible": true,
      "source": "Uploaded Report"
    }
  }
}
```

### 2. `POST /api/health-summary`
Combines DR screening results with verified health report data.
- **Request**: JSON body with `dr_result`, `report_data`, and optional `manual_entries`.
- **Response**: Unified screening summary payload ready for UI rendering and patient report generation.

---

## 6. Regulatory & Clinical Disclaimers

> [!IMPORTANT]
> **Not a Standalone Diagnostic Tool**: This software is designed exclusively for screening assistance, risk stratification, and patient prioritization in rural community healthcare settings under the supervision of qualified clinicians.
>
> All AI-generated outputs, OCR readings, and automated referral recommendations must be clinically verified by a registered ophthalmologist or medical professional prior to initiating treatment.
