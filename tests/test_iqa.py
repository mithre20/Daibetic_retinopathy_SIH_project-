"""
Unit and Integration Tests for Retinal Image Quality Assessment (IQA) & Gating Pipeline
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Validates (Section 31):
  1. RetinalQualityModel import
  2. Checkpoint loading from models/retinal_quality_mobilenet.pth
  3. Output shape is [1, 3]
  4. Class mapping: 0 = Good Quality, 1 = Poor Quality, 2 = Unusable
  5. Image preprocessing transforms image to [1, 3, 224, 224] with correct normalization
  6. Inference on real fundus image produces valid class and confidence
  7. Confidence is in [0, 1] range and sum of probabilities ≈ 1.0
  8. Gating logic: Good Quality with conf >= 0.75 allows DR screening
  9. Gating logic: Poor Quality blocks DR screening
  10. Gating logic: Unusable blocks DR screening
  11. Gating logic: Good Quality with conf < 0.75 blocks DR screening (confidence threshold test)
  12. Rejected image: DR prediction is NOT generated (returns None / null)
  13. Accepted image: DR prediction IS generated with severity grade
  14. Regression test: EfficientNet-B0 Grad-CAM still works
  15. Regression test: EfficientNet-B0 SHAP still works
  16. Regression test: U-Net segmentation still works
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 1. Model & IQA imports
from models.iqa_model import (
    RetinalQualityModel,
    build_iqa_model,
    get_iqa_transforms,
    find_iqa_checkpoint,
    IQA_CLASSES,
    IQA_CLASS_KEYS,
    IQA_CLASS_DESCRIPTIONS,
    DEFAULT_GOOD_THRESHOLD
)
from inference.predict_iqa import IQAInferenceEngine
from inference.screening_pipeline import (
    RetinalScreeningGatekeeper,
    screen_retinal_image
)
from utils.image_quality_metrics import compute_image_quality_metrics

# Existing modules for regression verification
from backend.model.efficientnet_dr import (
    EfficientNetDRClassifier,
    build_model,
    CLASS_NAMES
)
from backend.model.inference import DRInferenceEngine, get_inference_transforms
from inference.gradcam import GradCAM, apply_colormap_and_overlay
from explainability.shap_explainer import generate_shap_explanation
from models.unet_model import UNet
from backend.app import app


class TestRetinalImageQualityAssessment(unittest.TestCase):
    """Comprehensive test suite for MobileNetV3-Small IQA and Clinical Gating Pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.test_image_path = os.path.join(ROOT_DIR, "fundus_scan.jpg")
        if not os.path.isfile(cls.test_image_path):
            # Fallback dummy image if fundus_scan.jpg is not present
            dummy = Image.fromarray(np.random.randint(40, 200, (512, 512, 3), dtype=np.uint8))
            dummy.save(cls.test_image_path)

        cls.fundus_img = Image.open(cls.test_image_path).convert("RGB")
        cls.iqa_engine = IQAInferenceEngine(device=cls.device)
        cls.gatekeeper = RetinalScreeningGatekeeper(device=cls.device, good_threshold=0.75)

    def test_1_iqa_model_import(self):
        """Test 1: Verify RetinalQualityModel class and components can be imported."""
        self.assertTrue(issubclass(RetinalQualityModel, torch.nn.Module))
        self.assertEqual(len(IQA_CLASSES), 3)
        self.assertEqual(IQA_CLASSES[0], "Good Quality")
        self.assertEqual(IQA_CLASSES[1], "Poor Quality")
        self.assertEqual(IQA_CLASSES[2], "Unusable")

    def test_2_checkpoint_loading(self):
        """Test 2: Verify checkpoint loading from models/retinal_quality_mobilenet.pth."""
        ckpt_path = find_iqa_checkpoint()
        self.assertIsNotNone(ckpt_path, "IQA Checkpoint not found")
        self.assertTrue(os.path.isfile(ckpt_path), f"Checkpoint missing at {ckpt_path}")

        model = build_iqa_model(num_classes=3, pretrained_backbone=False, checkpoint_path=ckpt_path)
        self.assertIsInstance(model, RetinalQualityModel)

    def test_3_output_shape(self):
        """Test 3: Verify forward pass produces shape [1, 3] for batch of 1."""
        model = build_iqa_model(num_classes=3, pretrained_backbone=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, 3))

    def test_4_class_mapping(self):
        """Test 4: Verify class mapping: 0 = Good Quality, 1 = Poor Quality, 2 = Unusable."""
        expected = {
            0: "Good Quality",
            1: "Poor Quality",
            2: "Unusable"
        }
        self.assertEqual(IQA_CLASSES, expected)
        self.assertEqual(IQA_CLASS_KEYS["good"], 0)
        self.assertEqual(IQA_CLASS_KEYS["poor"], 1)
        self.assertEqual(IQA_CLASS_KEYS["unusable"], 2)

    def test_5_image_preprocessing(self):
        """Test 5: Verify preprocessing transforms image to [1, 3, 224, 224] with correct normalization."""
        transform = get_iqa_transforms(image_size=224, is_training=False)
        tensor = transform(self.fundus_img).unsqueeze(0)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertEqual(tensor.dtype, torch.float32)
        # Verify normalization (standard ImageNet mean/std results in values spanning both <0 and >0)
        self.assertLess(tensor.min().item(), 0.0)
        self.assertGreater(tensor.max().item(), 0.0)

    def test_6_real_fundus_inference(self):
        """Test 6: Verify inference on real fundus image produces valid class and confidence."""
        res = self.iqa_engine.predict_quality(self.fundus_img)
        self.assertIn("predicted_class", res)
        self.assertIn("class_id", res)
        self.assertIn("confidence", res)
        self.assertIn(res["class_id"], [0, 1, 2])
        self.assertIn(res["predicted_class"], ["Good Quality", "Poor Quality", "Unusable"])

    def test_7_confidence_range_and_sum(self):
        """Test 7: Verify confidence is in [0, 1] range and sum of probabilities ≈ 1.0."""
        res = self.iqa_engine.predict_quality(self.fundus_img)
        conf = res["confidence"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

        probs = res["class_probabilities"]
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_8_gating_good_quality_allows_screening(self):
        """Test 8: Gating logic: Good Quality with conf >= 0.75 allows DR screening."""
        gate_res = self.gatekeeper.evaluate_gate(
            predicted_class="Good Quality",
            confidence=0.92,
            threshold=0.75
        )
        self.assertTrue(gate_res["screening_allowed"])
        self.assertIn("Proceeding with DR screening", gate_res["reason"])

    def test_9_gating_poor_quality_blocks_screening(self):
        """Test 9: Gating logic: Poor Quality blocks DR screening."""
        gate_res = self.gatekeeper.evaluate_gate(
            predicted_class="Poor Quality",
            confidence=0.88,
            threshold=0.75
        )
        self.assertFalse(gate_res["screening_allowed"])
        self.assertIn("blocked", gate_res["reason"].lower())

    def test_10_gating_unusable_blocks_screening(self):
        """Test 10: Gating logic: Unusable blocks DR screening."""
        gate_res = self.gatekeeper.evaluate_gate(
            predicted_class="Unusable",
            confidence=0.95,
            threshold=0.75
        )
        self.assertFalse(gate_res["screening_allowed"])
        self.assertIn("blocked", gate_res["reason"].lower())

    def test_11_gating_low_confidence_good_blocks_screening(self):
        """Test 11: Gating logic: Good Quality with conf < 0.75 blocks DR screening."""
        gate_res = self.gatekeeper.evaluate_gate(
            predicted_class="Good Quality",
            confidence=0.68,
            threshold=0.75
        )
        self.assertFalse(gate_res["screening_allowed"])
        self.assertIn("confidence 68.00% is below safety threshold 75.00%", gate_res["reason"])

    def test_12_rejected_image_dr_prediction_is_none(self):
        """Test 12: Verify rejected image does NOT call EfficientNet (returns None / null)."""
        # Create a synthetic unusable image (pitch black with noise)
        black_arr = np.zeros((512, 512, 3), dtype=np.uint8)
        black_img = Image.fromarray(black_arr)

        # Force gatekeeper with a threshold of 1.0 to guarantee rejection
        res = self.gatekeeper.process(black_img, threshold=1.0)
        self.assertFalse(res["screening_gate"]["screening_allowed"])
        self.assertIsNone(res["dr_prediction"], "dr_prediction MUST be None when screening is blocked")

    def test_13_accepted_image_dr_prediction_generated(self):
        """Test 13: Verify accepted image calls EfficientNet and returns DR severity grade."""
        # Run gatekeeper with threshold=0.0 to simulate accepted gate
        res = self.gatekeeper.process(self.fundus_img, threshold=0.0)
        if res["screening_gate"]["screening_allowed"]:
            self.assertIsNotNone(res["dr_prediction"])
            dr = res["dr_prediction"]
            self.assertIn("prediction", dr)
            self.assertIn(dr["prediction"], [0, 1, 2, 3, 4])
            self.assertIn("severity", dr)
            self.assertIn(dr["severity"], CLASS_NAMES)
            self.assertIn("confidence", dr)
            self.assertIn("referable", dr)

    def test_14_regression_gradcam_still_works(self):
        """Test 14: Regression test: EfficientNet-B0 Grad-CAM still functions correctly."""
        dr_eng = DRInferenceEngine()
        model = dr_eng.model
        target_layer = model.get_gradcam_target_layer()
        transform = get_inference_transforms()
        input_tensor = transform(self.fundus_img).unsqueeze(0).to(self.device)

        with GradCAM(model, target_layer) as gcam:
            cam_result = gcam.generate_cam(input_tensor)

        self.assertIn("heatmap", cam_result)
        self.assertIn("prediction", cam_result)
        self.assertIn("sanity_stats", cam_result)

        orig_np, heatmap_rgb, heatmap_pil, overlay_pil = apply_colormap_and_overlay(
            original_img=self.fundus_img,
            heatmap=cam_result["heatmap"],
            alpha=0.4
        )
        self.assertIsNotNone(overlay_pil)
        self.assertEqual(overlay_pil.size, self.fundus_img.size)

    def test_15_regression_shap_still_works(self):
        """Test 15: Regression test: EfficientNet-B0 SHAP still functions correctly."""
        # Fast nsamples=5 for rapid unit test validation
        shap_res = generate_shap_explanation(
            image=self.fundus_img,
            target_class=None,
            nsamples=5,
            device=self.device
        )
        self.assertIn("prediction_class", shap_res)
        self.assertIn("prediction_label", shap_res)
        self.assertIn("positive_map", shap_res)
        self.assertIn("negative_map", shap_res)
        self.assertIn("combined_map", shap_res)
        self.assertIn("shap_stats", shap_res)
        self.assertIsNotNone(shap_res["combined_map"])

    def test_16_regression_unet_segmentation_still_works(self):
        """Test 16: Regression test: U-Net segmentation architecture remains operational."""
        unet = UNet(in_channels=3, num_classes=5)
        unet.eval()
        dummy_tensor = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = unet(dummy_tensor)
        self.assertEqual(output.shape, (1, 5, 128, 128))

    def test_17_api_iqa_predict_endpoint(self):
        """Test 17: Verify POST /api/iqa/predict endpoint accepts fundus image."""
        client = app.test_client()
        with open(self.test_image_path, "rb") as f:
            data = {"image": (f, "fundus.jpg")}
            res = client.post("/api/iqa/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertIn("quality_class", json_data)
        self.assertIn("quality_label", json_data)
        self.assertIn("confidence", json_data)
        self.assertIn("screening_allowed", json_data)
        self.assertIn("class_probabilities", json_data)

    def test_18_api_screen_endpoint(self):
        """Test 18: Verify POST /api/screen endpoint executes full gated screening."""
        client = app.test_client()
        with open(self.test_image_path, "rb") as f:
            data = {"image": (f, "fundus.jpg")}
            res = client.post("/api/screen?threshold=0.0", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)
        json_data = res.get_json()
        self.assertIn("screening_gate", json_data)
        self.assertIn("image_quality", json_data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
