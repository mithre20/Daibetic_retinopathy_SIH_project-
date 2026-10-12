"""
Unit Tests for Microaneurysm Detection Module
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

from retinal_analysis.lesions.microaneurysm import MicroaneurysmDetector, detect_microaneurysms


class TestMicroaneurysmDetection(unittest.TestCase):

    def setUp(self):
        self.detector = MicroaneurysmDetector(target_size=(256, 256))
        # Create test fundus with small punctate dark red dots
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.test_img[30:270, 30:270] = [150, 70, 25]
        import cv2
        # Punctate microaneurysms (radius 2-3 px)
        cv2.circle(self.test_img, (100, 100), 2, (50, 10, 5), -1)
        cv2.circle(self.test_img, (160, 180), 3, (50, 10, 5), -1)

    def test_1_detector_initialization(self):
        """Test 1: Verify MicroaneurysmDetector initialization."""
        self.assertEqual(self.detector.target_size, (256, 256))

    def test_2_detection_output_schema(self):
        """Test 2: Verify candidate counts, subpixel status, and confidence."""
        res = self.detector.detect(self.test_img)
        self.assertIn("detected", res)
        self.assertIn("candidates_count", res)
        self.assertIn("candidate_coordinates", res)
        self.assertIn("binary_mask", res)
        self.assertIn("confidence", res)
        self.assertIn("subpixel_localization", res)
        # Sub-pixel accuracy must be honestly reported as NOT VALIDATED
        self.assertEqual(res["subpixel_localization"], "NOT VALIDATED")
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)

    def test_3_invalid_file_handling(self):
        """Test 3: Verify missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            detect_microaneurysms("non_existent_scan.png")

    def test_4_inference_outputs(self):
        """Test 4: Verify production inference generates mask and overlay."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = detect_microaneurysms(self.test_img, output_dir=tmp_dir)
            self.assertTrue(os.path.isfile(res["saved_paths"]["predicted_mask"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["overlay"]))


if __name__ == "__main__":
    unittest.main()
