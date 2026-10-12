"""
Unit Tests for Fovea Localization Module
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

from retinal_analysis.fovea.model import FoveaLocalizer
from retinal_analysis.fovea.inference import locate_fovea
from retinal_analysis.fovea.evaluate import evaluate_fovea_dataset


class TestFoveaLocalization(unittest.TestCase):

    def setUp(self):
        self.localizer = FoveaLocalizer(target_size=(256, 256))
        # Create test fundus with OD on the right side (Right eye OD view)
        self.test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        self.test_img[30:270, 30:270] = [120, 45, 15]
        import cv2
        # Disc on right side
        cv2.circle(self.test_img, (240, 150), 25, (255, 220, 140), -1)
        # Fovea dark spot on left side (temporal)
        cv2.circle(self.test_img, (110, 155), 15, (40, 15, 5), -1)

    def test_1_localizer_initialization(self):
        """Test 1: Verify FoveaLocalizer initialization."""
        self.assertEqual(self.localizer.target_size, (256, 256))

    def test_2_output_schema_and_laterality(self):
        """Test 2: Verify fovea outputs schema and laterality determination."""
        res = self.localizer.locate(self.test_img)
        self.assertIn("detected", res)
        self.assertIn("center", res)
        self.assertIn("roi", res)
        self.assertIn("confidence", res)
        self.assertIn("eye_laterality", res)
        self.assertTrue(res["detected"])
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertIn(res["eye_laterality"], ["OD", "OS", "UNKNOWN"])

    def test_3_invalid_file_handling(self):
        """Test 3: Verify missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            locate_fovea("non_existent_retina.jpg")

    def test_4_inference_outputs_generation(self):
        """Test 4: Verify inference saves fovea visual overlay and json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = locate_fovea(self.test_img, output_dir=tmp_dir)
            self.assertTrue(os.path.isfile(res["saved_paths"]["original"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["overlay"]))
            self.assertTrue(os.path.isfile(res["saved_paths"]["prediction_json"]))

    def test_5_evaluation_metrics(self):
        """Test 5: Verify evaluation reporting."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, "test_fovea.png")
            Image.fromarray(self.test_img).save(img_path)
            
            gt = [{"fovea_x": 110, "fovea_y": 155}]
            eval_res = evaluate_fovea_dataset([img_path], ground_truth_annotations=gt)
            self.assertEqual(eval_res["detected_count"], 1)
            self.assertIn("mean_localization_error_px", eval_res)
            self.assertIn("tolerance_percentage", eval_res)


if __name__ == "__main__":
    unittest.main()
