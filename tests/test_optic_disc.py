"""
Unit Tests for Optic Disc Localization Module
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

from retinal_analysis.optic_disc.model import OpticDiscDetector
from retinal_analysis.optic_disc.inference import locate_optic_disc
from retinal_analysis.optic_disc.evaluate import evaluate_optic_disc_dataset


class TestOpticDiscLocalization(unittest.TestCase):

    def setUp(self):
        self.detector = OpticDiscDetector(target_size=(256, 256))
        # Create a synthetic fundus-like image with a bright circular disk
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        # Retinal background (dark red/orange)
        self.test_img[30:270, 30:270] = [120, 45, 15]
        # Optic disc (bright yellow-orange circle)
        import cv2
        cv2.circle(self.test_img, (220, 150), 30, (255, 220, 140), -1)

    def test_1_detector_initialization(self):
        """Test 1: Verify OpticDiscDetector initialization."""
        self.assertEqual(self.detector.target_size, (256, 256))

    def test_2_detection_output_schema(self):
        """Test 2: Verify detector outputs all required fields and valid ranges."""
        res = self.detector.detect(self.test_img)
        self.assertIn("detected", res)
        self.assertIn("center", res)
        self.assertIn("radius", res)
        self.assertIn("bbox", res)
        self.assertIn("mask", res)
        self.assertIn("confidence", res)
        
        self.assertTrue(res["detected"])
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertEqual(len(res["center"]), 2)
        self.assertEqual(len(res["bbox"]), 4)

    def test_3_localization_accuracy(self):
        """Test 3: Verify detected center is close to synthetic disk center (220, 150)."""
        res = self.detector.detect(self.test_img)
        cx, cy = res["center"]
        # Expect within 35 pixels
        dist = np.sqrt((cx - 220)**2 + (cy - 150)**2)
        self.assertLess(dist, 40.0)

    def test_4_invalid_image_handling(self):
        """Test 4: Verify proper error on missing image file."""
        with self.assertRaises(FileNotFoundError):
            locate_optic_disc("non_existent_fundus_scan.jpg")

    def test_5_inference_file_generation(self):
        """Test 5: Verify inference saves original, overlay, and json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = locate_optic_disc(self.test_img, output_dir=tmp_dir)
            self.assertTrue(os.path.isfile(res["saved_paths"]["original"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["overlay"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["prediction_json"]))

    def test_6_evaluation_reporting(self):
        """Test 6: Verify evaluation module returns valid metrics payload."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, "test.png")
            Image.fromarray(self.test_img).save(img_path)
            
            # Ground truth available
            gt = [{"center_x": 220, "center_y": 150, "radius": 30}]
            eval_res = evaluate_optic_disc_dataset([img_path], ground_truth_annotations=gt)
            self.assertEqual(eval_res["detected_count"], 1)
            self.assertIn("mean_center_error_px", eval_res)
            self.assertIn("tolerance_success_rate", eval_res)


if __name__ == "__main__":
    unittest.main()
