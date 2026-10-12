"""
Unit Tests for Retinal Vessel Segmentation Module
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

from retinal_analysis.vessels.model import RetinalVesselSegmenter
from retinal_analysis.vessels.inference import segment_retinal_vessels
from retinal_analysis.vessels.evaluate import evaluate_vessel_segmentation


class TestVesselSegmentation(unittest.TestCase):

    def setUp(self):
        self.segmenter = RetinalVesselSegmenter(target_size=(256, 256))
        # Create test fundus with dark linear vessel lines
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.test_img[30:270, 30:270] = [140, 60, 20]
        import cv2
        # Draw dark red/brown branching vessel lines
        cv2.line(self.test_img, (50, 150), (250, 150), (60, 15, 5), 4)
        cv2.line(self.test_img, (150, 50), (150, 250), (60, 15, 5), 3)

    def test_1_segmenter_initialization(self):
        """Test 1: Verify RetinalVesselSegmenter initialization."""
        self.assertEqual(self.segmenter.target_size, (256, 256))

    def test_2_segmentation_output_schema(self):
        """Test 2: Verify binary mask, probability map, and vessel density."""
        res = self.segmenter.segment(self.test_img)
        self.assertIn("binary_mask", res)
        self.assertIn("vessel_probability", res)
        self.assertIn("vessel_density_pct", res)
        self.assertEqual(res["binary_mask"].shape, (256, 256))
        self.assertTrue(0.0 <= res["vessel_density_pct"] <= 100.0)

    def test_3_invalid_file_handling(self):
        """Test 3: Verify missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            segment_retinal_vessels("non_existent_image.png")

    def test_4_inference_outputs(self):
        """Test 4: Verify inference saves predicted mask and overlay."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = segment_retinal_vessels(self.test_img, output_dir=tmp_dir)
            self.assertTrue(os.path.isfile(res["saved_paths"]["predicted_mask"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["overlay"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["metrics_json"]))

    def test_5_evaluation_metrics_computation(self):
        """Test 5: Verify evaluation metrics calculation against GT mask."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, "test_vessel.png")
            gt_path = os.path.join(tmp_dir, "gt_vessel.png")
            
            Image.fromarray(self.test_img).save(img_path)
            # Create a simple binary GT mask matching the drawn line
            gt = np.zeros((300, 300), dtype=np.uint8)
            gt[148:153, 50:250] = 255
            Image.fromarray(gt).save(gt_path)
            
            eval_res = evaluate_vessel_segmentation([img_path], ground_truth_mask_paths=[gt_path])
            self.assertIn("mean_dice", eval_res)
            self.assertIn("mean_sensitivity", eval_res)
            self.assertIn("mean_specificity", eval_res)
            self.assertEqual(eval_res["status"], "EVALUATED_AGAINST_GROUND_TRUTH")


if __name__ == "__main__":
    unittest.main()
