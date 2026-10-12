"""
Unit Tests for Hemorrhage Detection and Classification Module
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

from retinal_analysis.lesions.hemorrhage import HemorrhageAnalyzer, analyze_hemorrhages


class TestHemorrhageAnalysis(unittest.TestCase):

    def setUp(self):
        self.analyzer = HemorrhageAnalyzer(target_size=(256, 256))
        # Create fundus with distinct blot and flame hemorrhages
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.test_img[30:270, 30:270] = [150, 70, 25]
        import cv2
        # Blot hemorrhage (irregular dark red patch)
        cv2.ellipse(self.test_img, (120, 120), (14, 8), 30, 0, 360, (70, 10, 5), -1)
        cv2.circle(self.test_img, (180, 190), 8, (65, 10, 5), -1)

    def test_1_analyzer_initialization(self):
        """Test 1: Verify HemorrhageAnalyzer initialization."""
        self.assertEqual(self.analyzer.target_size, (256, 256))

    def test_2_analysis_output_schema(self):
        """Test 2: Verify classification label, counts, quadrants, and masks."""
        res = self.analyzer.analyze(self.test_img)
        self.assertIn("classification", res)
        self.assertIn("detected", res)
        self.assertIn("hemorrhage_count", res)
        self.assertIn("area_percentage", res)
        self.assertIn("quadrant_distribution", res)
        self.assertIn("binary_mask", res)
        self.assertIn(res["classification"], ["Hemorrhages Present", "Hemorrhages Absent"])
        self.assertIsInstance(res["quadrant_distribution"], dict)

    def test_3_invalid_file_handling(self):
        """Test 3: Verify missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            analyze_hemorrhages("non_existent_retina.jpg")

    def test_4_inference_outputs(self):
        """Test 4: Verify production inference generates mask and overlay."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = analyze_hemorrhages(self.test_img, output_dir=tmp_dir)
            self.assertTrue(os.path.isfile(res["saved_paths"]["predicted_mask"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["overlay"]))


if __name__ == "__main__":
    unittest.main()
