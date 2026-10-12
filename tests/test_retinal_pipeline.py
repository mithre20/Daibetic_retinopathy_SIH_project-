"""
Unit Tests for Unified Retinal Analysis Pipeline
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retinal_analysis.pipeline import UnifiedRetinalPipeline, analyze_retinal_structures
from retinal_analysis.report import generate_retinal_evidence_report


class TestUnifiedRetinalPipeline(unittest.TestCase):

    def setUp(self):
        self.pipeline = UnifiedRetinalPipeline(target_size=(256, 256))
        # Create a synthetic fundus scan
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.test_img[30:270, 30:270] = [130, 50, 20]
        import cv2
        # Disc
        cv2.circle(self.test_img, (240, 150), 25, (255, 220, 140), -1)
        # Fovea
        cv2.circle(self.test_img, (110, 155), 15, (40, 15, 5), -1)
        # Vessels
        cv2.line(self.test_img, (60, 150), (240, 150), (60, 15, 5), 3)
        # Exudate
        cv2.circle(self.test_img, (150, 120), 4, (255, 240, 50), -1)

    def test_1_pipeline_structure_schema(self):
        """Test 1: Verify pipeline returns all structural and lesion components."""
        res = self.pipeline.analyze(self.test_img)
        self.assertIn("retinal_structures", res)
        self.assertIn("lesions", res)
        self.assertIn("neovascularization", res)
        self.assertIn("execution_times_ms", res)
        
        # Structural keys
        structs = res["retinal_structures"]
        self.assertIn("optic_disc", structs)
        self.assertIn("fovea", structs)
        self.assertIn("vessels", structs)
        
        # Lesion keys
        lesions = res["lesions"]
        self.assertIn("microaneurysms", lesions)
        self.assertIn("hemorrhages", lesions)
        self.assertIn("hard_exudates", lesions)
        self.assertIn("soft_exudates", lesions)
        
        # Neovascularization status
        self.assertIn("PENDING DATASET", res["neovascularization"]["status"])

    def test_2_composite_evidence_generation(self):
        """Test 2: Verify all image outputs are saved to output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self.pipeline.analyze(self.test_img, output_dir=tmp_dir, include_base64=True)
            self.assertTrue(os.path.isfile(res["composite_overlay_path"]))
            self.assertTrue(os.path.isfile(res["json_path"]))
            self.assertIn("composite_overlay_base64", res)
            self.assertTrue(res["composite_overlay_base64"].startswith("data:image/png;base64,"))

    def test_3_report_generator_formatting(self):
        """Test 3: Verify human-readable evidence report output."""
        res = self.pipeline.analyze(self.test_img)
        dr_dummy = {"severity": "Moderate DR", "confidence": 0.88, "referable": True}
        report_text = generate_retinal_evidence_report(dr_dummy, res)
        
        self.assertIn("AI DR SCREEN — RETINAL ANALYSIS", report_text)
        self.assertIn("Grade:      Moderate DR", report_text)
        self.assertIn("Optic Disc:", report_text)
        self.assertIn("Fovea:", report_text)
        self.assertIn("Vessel Segmentation:", report_text)
        self.assertIn("Microaneurysms:", report_text)
        self.assertIn("Hemorrhages:", report_text)
        self.assertIn("Neovascularization: Not evaluated", report_text)


if __name__ == "__main__":
    unittest.main()
