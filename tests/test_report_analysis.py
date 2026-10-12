"""
Unit and Integration Tests for Patient Health Report Analysis Module
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Test Suite covering:
- Test 1: Valid PDF parsing & direct text extraction
- Test 2: Scanned PDF / image report OCR extraction
- Test 3: Fasting glucose extraction (synonyms & abbreviations)
- Test 4: HbA1c extraction (percentage & reference ranges)
- Test 5: Blood pressure extraction in composite '120/80' format
- Test 6: Blood pressure extraction in separated 'Systolic: X / Diastolic: Y' format
- Test 7: mg/dL glucose unit handling & normalization
- Test 8: mmol/L glucose unit handling & conversion (mmol/L -> mg/dL)
- Test 9: Rejection of unsupported file extensions (.txt, .docx, .exe)
- Test 10: Graceful handling of missing glucose
- Test 11: Graceful handling of missing blood pressure
- Test 12: Robust handling of unparseable/corrupted report data
- Test 13: Manual vitals entry and user verification override
- Test 14: Backward-compatibility verification for EfficientNet-B0
- Test 15: Backward-compatibility verification for ResNet50
- Test 16: Backward-compatibility verification for U-Net
"""

import os
import io
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from report_analysis import (
    ReportExtractor,
    ReportParser,
    HealthReportAnalyzer,
    convert_glucose,
    validate_glucose,
    validate_hba1c,
    validate_blood_pressure,
    analyze_report
)


def create_sample_pdf_bytes(text_lines: list) -> bytes:
    """Helper to synthesize a digital PDF document with specified text lines."""
    stream_content = "BT\n/F1 12 Tf\n50 700 Td\n"
    for idx, line in enumerate(text_lines):
        escaped = line.replace("(", "\\(").replace(")", "\\)")
        if idx == 0:
            stream_content += f"({escaped}) Tj\n"
        else:
            stream_content += f"0 -24 Td\n({escaped}) Tj\n"
    stream_content += "ET\n"
    stream_bytes = stream_content.encode("latin1")

    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        b"5 0 obj << /Length " + str(len(stream_bytes)).encode("latin1") + b" >>\nstream\n"
        + stream_bytes +
        b"\nendstream\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000244 00000 n \n"
        b"0000000318 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n500\n%%EOF"
    )
    return pdf


def create_sample_image_bytes(text_lines: list) -> bytes:
    """Helper to synthesize a clear image report with rendered text lines."""
    img = Image.new("RGB", (650, 60 + len(text_lines) * 45), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 25
    for line in text_lines:
        draw.text((25, y), line, fill=(0, 0, 0))
        y += 40
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestReportAnalysis(unittest.TestCase):
    """
    Comprehensive test suite for the patient health report analysis pipeline.
    """

    def setUp(self):
        self.analyzer = HealthReportAnalyzer()
        self.parser = ReportParser()

    # -------------------------------------------------------------------------
    # Test 1: Upload valid PDF
    # -------------------------------------------------------------------------
    def test_1_upload_valid_pdf(self):
        pdf_bytes = create_sample_pdf_bytes([
            "Patient Clinical Laboratory Report",
            "Fasting Blood Glucose: 148 mg/dL (70 - 99)",
            "HbA1c: 7.6 %",
            "Blood Pressure: 145/92 mmHg"
        ])
        res = self.analyzer.analyze_report(pdf_bytes, filename="lab_report.pdf")
        self.assertTrue(res["success"])
        self.assertEqual(res["file_type"], "PDF")
        self.assertIn("fasting", res["glucose"])
        self.assertEqual(res["glucose"]["fasting"]["value"], 148.0)
        self.assertIsNotNone(res["blood_pressure"])
        self.assertEqual(res["blood_pressure"]["systolic"], 145)
        self.assertEqual(res["blood_pressure"]["diastolic"], 92)

    # -------------------------------------------------------------------------
    # Test 2: Upload scanned PDF / image
    # -------------------------------------------------------------------------
    def test_2_upload_image_report(self):
        img_bytes = create_sample_image_bytes([
            "Fasting Blood Glucose 140 mg/dL",
            "Blood Pressure 130/85 mmHg"
        ])
        res = self.analyzer.analyze_report(img_bytes, filename="scanned_report.png")
        self.assertTrue(res["success"])
        self.assertEqual(res["file_type"], "IMAGE")
        self.assertTrue(res["has_glucose"] or res["has_blood_pressure"])
        if res["blood_pressure"]:
            self.assertEqual(res["blood_pressure"]["systolic"], 130)
            self.assertEqual(res["blood_pressure"]["diastolic"], 85)

    # -------------------------------------------------------------------------
    # Test 3: Extract fasting glucose
    # -------------------------------------------------------------------------
    def test_3_extract_fasting_glucose(self):
        variations = [
            "Fasting Blood Glucose 148 mg/dL",
            "FBS: 154 mg/dL",
            "Glucose - Fasting: 132 mg/dL",
            "FBG 160 mg/dL"
        ]
        for var in variations:
            parsed = self.parser.parse_text(var)
            self.assertIn("fasting", parsed["glucose"], f"Failed to match fasting keyword: {var}")
            self.assertGreater(parsed["glucose"]["fasting"]["value"], 100)

    # -------------------------------------------------------------------------
    # Test 4: Extract HbA1c
    # -------------------------------------------------------------------------
    def test_4_extract_hba1c(self):
        variations = [
            "HbA1c: 7.6%",
            "HbA1c 8.2 % (4.0 - 5.6)",
            "Glycated Hemoglobin: 6.8%",
            "Glycosylated Hemoglobin 7.1"
        ]
        for var in variations:
            parsed = self.parser.parse_text(var)
            self.assertIn("hba1c", parsed["glucose"], f"Failed to match HbA1c: {var}")
            self.assertGreaterEqual(parsed["glucose"]["hba1c"]["value"], 6.0)

    # -------------------------------------------------------------------------
    # Test 5: Extract BP in 120/80 format
    # -------------------------------------------------------------------------
    def test_5_extract_bp_composite(self):
        sample_texts = [
            "BP: 120/80 mmHg",
            "Blood Pressure 145/92",
            "NIBP: 138/88 mmHg"
        ]
        for st in sample_texts:
            parsed = self.parser.parse_text(st)
            self.assertIsNotNone(parsed["blood_pressure"], f"Failed to parse composite BP: {st}")
            self.assertGreater(parsed["blood_pressure"]["systolic"], parsed["blood_pressure"]["diastolic"])

    # -------------------------------------------------------------------------
    # Test 6: Extract separate Systolic and Diastolic
    # -------------------------------------------------------------------------
    def test_6_extract_separate_bp(self):
        text = "Vitals Check:\nSystolic BP: 130 mmHg\nDiastolic BP: 85 mmHg\nPulse: 72 bpm"
        parsed = self.parser.parse_text(text)
        self.assertIsNotNone(parsed["blood_pressure"])
        self.assertEqual(parsed["blood_pressure"]["systolic"], 130)
        self.assertEqual(parsed["blood_pressure"]["diastolic"], 85)

    # -------------------------------------------------------------------------
    # Test 7: Handle mg/dL unit
    # -------------------------------------------------------------------------
    def test_7_handle_mg_dl_unit(self):
        conv = convert_glucose(148.0, "mg/dL")
        self.assertEqual(conv["normalized_value"], 148.0)
        self.assertEqual(conv["normalized_unit"], "mg/dL")
        self.assertEqual(conv["original_unit"], "mg/dL")
        self.assertAlmostEqual(conv["value_mmol_l"], 148.0 / 18.0182, delta=0.1)

    # -------------------------------------------------------------------------
    # Test 8: Handle mmol/L unit
    # -------------------------------------------------------------------------
    def test_8_handle_mmol_l_unit(self):
        conv = convert_glucose(7.6, "mmol/L")
        self.assertEqual(conv["original_unit"], "mmol/L")
        # 7.6 mmol/L * 18.0182 = ~136.9 mg/dL
        self.assertAlmostEqual(conv["value_mg_dl"], 136.9, delta=0.5)
        self.assertEqual(conv["normalized_unit"], "mg/dL")

    # -------------------------------------------------------------------------
    # Test 9: Reject unsupported file types
    # -------------------------------------------------------------------------
    def test_9_reject_unsupported_file_types(self):
        bad_files = ["report.docx", "results.txt", "scan.exe", "data.csv"]
        for bf in bad_files:
            res = self.analyzer.analyze_report(b"dummy data", filename=bf)
            self.assertFalse(res["success"])
            self.assertIn("Unsupported file type", res["error"])

    # -------------------------------------------------------------------------
    # Test 10: Handle missing glucose gracefully
    # -------------------------------------------------------------------------
    def test_10_handle_missing_glucose(self):
        text = "Patient Report\nBlood Pressure: 120/80 mmHg\nCholesterol: 180 mg/dL"
        parsed = self.parser.parse_text(text)
        self.assertFalse(parsed["has_glucose"])
        self.assertEqual(len(parsed["glucose"]), 0)
        self.assertTrue(parsed["has_blood_pressure"])

    # -------------------------------------------------------------------------
    # Test 11: Handle missing BP gracefully
    # -------------------------------------------------------------------------
    def test_11_handle_missing_bp(self):
        text = "Patient Report\nFasting Blood Glucose: 110 mg/dL\nHbA1c: 5.8 %"
        parsed = self.parser.parse_text(text)
        self.assertTrue(parsed["has_glucose"])
        self.assertFalse(parsed["has_blood_pressure"])
        self.assertIsNone(parsed["blood_pressure"])

    # -------------------------------------------------------------------------
    # Test 12: Handle OCR errors / degraded files
    # -------------------------------------------------------------------------
    def test_12_handle_ocr_degraded_text(self):
        corrupt_bytes = b"Not a valid PDF or Image header"
        res = self.analyzer.analyze_report(corrupt_bytes, filename="broken.jpg")
        self.assertFalse(res["success"])
        self.assertIn("error", res)

    # -------------------------------------------------------------------------
    # Test 13: Manual vitals entry & user verification override
    # -------------------------------------------------------------------------
    def test_13_manual_vitals_and_override(self):
        # Simulated extracted report data
        rep_data = {
            "glucose": {
                "fasting": {
                    "parameter": "Fasting Blood Glucose",
                    "value": 148.0,
                    "unit": "mg/dL",
                    "source": "Uploaded Report"
                }
            },
            "blood_pressure": {
                "systolic": 145,
                "diastolic": 92,
                "unit": "mmHg",
                "source": "Uploaded Report"
            }
        }
        # Clinician manual override for BP
        manual_bp = {"systolic": 120, "diastolic": 80, "unit": "mmHg"}
        merged = self.analyzer.merge_with_manual_entries(
            report_data=rep_data,
            manual_bp=manual_bp
        )
        self.assertEqual(merged["manual_blood_pressure"]["systolic"], 120)
        self.assertEqual(merged["report_blood_pressure"]["systolic"], 145)
        # Verify source tagging
        self.assertEqual(merged["manual_blood_pressure"]["source"], "Manual Entry")
        self.assertEqual(merged["report_blood_pressure"]["source"], "Uploaded Report")

    # -------------------------------------------------------------------------
    # Test 14: Verify EfficientNet-B0 remains functional
    # -------------------------------------------------------------------------
    def test_14_efficientnet_remains_functional(self):
        from backend.model.inference import DRInferenceEngine
        engine = DRInferenceEngine()
        self.assertIsNotNone(engine.model)
        # Test on clean black dummy fundus scan
        dummy_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        pred = engine.predict(dummy_img)
        self.assertIn("prediction", pred)
        self.assertIn(pred["prediction"], [0, 1, 2, 3, 4])
        self.assertIn("probabilities", pred)

    # -------------------------------------------------------------------------
    # Test 15: Verify ResNet50 remains functional
    # -------------------------------------------------------------------------
    def test_15_resnet50_remains_functional(self):
        from models.resnet50_model import ResNet50DRClassifier
        model = ResNet50DRClassifier(num_classes=5)
        weights_path = os.path.join(ROOT_DIR, "models", "resnet50_dr.pth")
        if os.path.exists(weights_path):
            import torch
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        import torch
        inp = torch.randn(1, 3, 224, 224)
        out = model(inp)
        self.assertEqual(out.shape, (1, 5))

    # -------------------------------------------------------------------------
    # Test 16: Verify U-Net remains functional
    # -------------------------------------------------------------------------
    def test_16_unet_remains_functional(self):
        from models.unet_model import UNet
        unet = UNet(in_channels=3, num_classes=5)
        self.assertIsNotNone(unet)
        import torch
        inp = torch.randn(1, 3, 256, 256)
        out = unet(inp)
        self.assertEqual(out.shape, (1, 5, 256, 256))


if __name__ == "__main__":
    unittest.main(verbosity=2)
