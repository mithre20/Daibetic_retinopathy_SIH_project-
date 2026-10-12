"""
Unit Tests for Exudate Segmentation Module (Hard & Soft Exudates)
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

from retinal_analysis.lesions.exudate import ExudateSegmenter, segment_exudates


class TestExudateSegmentation(unittest.TestCase):

    def setUp(self):
        self.segmenter = ExudateSegmenter(target_size=(256, 256))
        # Create test fundus with both bright yellow hard exudates and fluffy soft exudates
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.test_img[30:270, 30:270] = [130, 50, 20]
        import cv2
        # Hard exudates (bright yellow dots)
        cv2.circle(self.test_img, (120, 100), 4, (255, 240, 50), -1)
        cv2.circle(self.test_img, (130, 105), 5, (255, 235, 60), -1)
        # Soft exudate (fuzzy larger pale patch)
        cv2.circle(self.test_img, (200, 180), 12, (200, 180, 170), -1)

    def test_1_segmenter_initialization(self):
        """Test 1: Verify ExudateSegmenter initialization."""
        self.assertEqual(self.segmenter.target_size, (256, 256))

    def test_2_differential_segmentation_schema(self):
        """Test 2: Verify hard vs soft exudate outputs and masks."""
        res = self.segmenter.segment(self.test_img)
        self.assertIn("hard_exudates_detected", res)
        self.assertIn("soft_exudates_detected", res)
        self.assertIn("hard_exudates_mask", res)
        self.assertIn("soft_exudates_mask", res)
        self.assertIn("hard_exudates_area_pct", res)
        self.assertIn("soft_exudates_area_pct", res)
        self.assertIn("confidence", res)

    def test_3_invalid_file_handling(self):
        """Test 3: Verify missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            segment_exudates("non_existent_fundus.png")

    def test_4_inference_outputs(self):
        """Test 4: Verify production inference generates both hard and soft masks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = segment_exudates(self.test_img, output_dir=tmp_dir)
            self.assertTrue(os.path.isfile(res["saved_paths"]["hard_exudates_mask"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["soft_exudates_mask"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["overlay"]))


if __name__ == "__main__":
    unittest.main()
