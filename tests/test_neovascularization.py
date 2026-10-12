"""
Unit Tests for Neovascularization Detection Module
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from retinal_analysis.neovascularization.model import NeovascularizationDetector
from retinal_analysis.neovascularization.inference import check_neovascularization
from retinal_analysis.neovascularization.evaluate import evaluate_neovascularization
from retinal_analysis.neovascularization.train import train_neovascularization_model, DatasetRequiredError


class TestNeovascularizationDetection(unittest.TestCase):

    def setUp(self):
        self.detector = NeovascularizationDetector()
        self.dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)

    def test_1_status_is_pending_dataset(self):
        """Test 1: Verify Neovascularization status is strictly PENDING DATASET + CLINICAL VALIDATION."""
        res = self.detector.detect(self.dummy_img)
        self.assertEqual(res["status"], "PENDING DATASET + CLINICAL VALIDATION")
        self.assertFalse(res["detected"])
        self.assertEqual(res["confidence"], 0.0)

    def test_2_inference_wrapper_consistency(self):
        """Test 2: Verify inference wrapper matches detector output."""
        res = check_neovascularization(self.dummy_img)
        self.assertIn("status", res)
        self.assertIn("PENDING DATASET", res["status"])

    def test_3_evaluation_without_ground_truth(self):
        """Test 3: Verify evaluation reports pending status without certified GT."""
        eval_res = evaluate_neovascularization([])
        self.assertIn("PENDING DATASET", eval_res["status"])
        self.assertIsNone(eval_res["sensitivity"])

    def test_4_training_guard(self):
        """Test 4: Verify training raises DatasetRequiredError without audited dataset."""
        with self.assertRaises(DatasetRequiredError):
            train_neovascularization_model(None)


if __name__ == "__main__":
    unittest.main()
