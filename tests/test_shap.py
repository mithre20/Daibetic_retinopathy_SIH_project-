"""
Unit and Integration Tests for SHAP Feature-Level Explainability
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Validates:
  1. SHAP module import
  2. Model loading (EfficientNet-B0 checkpoint)
  3. Correct 5-class DR count & keys
  4. Image preprocessing shape & ImageNet normalization
  5. Non-empty SHAP attribution generation
  6. Numerical validity (No NaN, No Inf)
  7. Output visualization dimensions match original fundus image
  8. Target class support across classes 0, 1, 2, 3, 4
  9. Default explanation targets predicted class
  10. Distinct explanations across differing target classes
  11. Real unseen retinal image testing on fundus_scan.jpg
  12. Backend API /api/explain/shap endpoint response
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

from backend.model.efficientnet_dr import (
    EfficientNetDRClassifier,
    build_model,
    get_device,
    CLASS_NAMES,
    CLASS_KEYS
)
from backend.model.inference import get_inference_transforms
from explainability.shap_explainer import (
    DRModelWrapper,
    find_model_checkpoint,
    generate_shap_explanation,
    compute_shap_maps
)
from explainability.shap_visualizer import (
    create_positive_map,
    create_negative_map,
    create_combined_map,
    create_overlay,
    pil_to_base64
)
from explainability.shap_summary import (
    generate_human_readable_summary,
    compute_lesion_aware_shap_stats,
    compute_shap_unet_overlap,
    MEDICAL_SAFETY_DISCLAIMER
)


class TestSHAPExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.ckpt_path = find_model_checkpoint()
        cls.assertTrue(cls.ckpt_path and os.path.isfile(cls.ckpt_path), "EfficientNet-B0 checkpoint must exist")
        cls.model = build_model(num_classes=5, pretrained=False, weights_path=cls.ckpt_path, device=cls.device)
        cls.model.eval()

        cls.fundus_img_path = os.path.join(ROOT_DIR, "fundus_scan.jpg")
        cls.assertTrue(os.path.isfile(cls.fundus_img_path), "Real fundus test image fundus_scan.jpg must exist")
        cls.real_img = Image.open(cls.fundus_img_path).convert("RGB")

    # Test 1 — SHAP module import
    def test_1_shap_module_import(self):
        """Test 1: Verify shap library and internal explainability modules import successfully."""
        import shap
        self.assertIsNotNone(shap.__version__)
        self.assertTrue(hasattr(shap, "GradientExplainer"), "SHAP must provide GradientExplainer")

    # Test 2 — Model loading
    def test_2_model_loading(self):
        """Test 2: Verify existing trained EfficientNet-B0 checkpoint loads cleanly."""
        self.assertIsInstance(self.model, EfficientNetDRClassifier)
        self.assertIsNotNone(self.ckpt_path)

    # Test 3 — Correct class count
    def test_3_correct_class_count(self):
        """Test 3: Verify model architecture outputs exactly 5 classes matching ICDR scale."""
        self.assertEqual(len(CLASS_NAMES), 5)
        self.assertEqual(len(CLASS_KEYS), 5)
        self.assertEqual(self.model.num_classes, 5)

        dummy_input = torch.zeros((1, 3, 224, 224), device=self.device)
        with torch.no_grad():
            output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 5), "Output logits must have shape [1, 5]")

    # Test 4 — Image preprocessing
    def test_4_image_preprocessing(self):
        """Test 4: Verify image preprocessing generates correct shape (1, 3, 224, 224) and normalization."""
        transform = get_inference_transforms()
        tensor = transform(self.real_img).unsqueeze(0)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        # Verify normalized values are roughly standard normal centered around 0
        self.assertTrue(tensor.min() < 0.0, "Normalized tensor should have negative values")
        self.assertTrue(tensor.max() > 0.0, "Normalized tensor should have positive values")

    # Test 5 — SHAP explanation non-empty values
    def test_5_shap_non_empty_values(self):
        """Test 5: Verify SHAP produces non-empty, non-zero attribution arrays."""
        res = generate_shap_explanation(
            image=self.real_img,
            model=self.model,
            nsamples=15,
            device=self.device
        )
        raw_shap = res["shap_values"]
        self.assertIsInstance(raw_shap, np.ndarray)
        self.assertGreater(raw_shap.size, 0)
        self.assertNotEqual(np.sum(np.abs(raw_shap)), 0.0, "SHAP values cannot be all zeros")

    # Test 6 — Numerical validity (no NaN, no Inf)
    def test_6_numerical_validity(self):
        """Test 6: Verify SHAP attributions contain no NaN or Infinite values."""
        res = generate_shap_explanation(
            image=self.real_img,
            model=self.model,
            nsamples=15,
            device=self.device
        )
        raw_shap = res["shap_values"]
        self.assertFalse(np.isnan(raw_shap).any(), "SHAP values must not contain NaN")
        self.assertFalse(np.isinf(raw_shap).any(), "SHAP values must not contain Inf")
        self.assertFalse(np.isnan(res["positive_map"]).any(), "Positive map must not contain NaN")
        self.assertFalse(np.isnan(res["negative_map"]).any(), "Negative map must not contain NaN")

    # Test 7 — Visualization dimensions
    def test_7_visualization_dimensions(self):
        """Test 7: Verify generated SHAP visualizations match original image dimensions."""
        orig_w, orig_h = self.real_img.size
        res = generate_shap_explanation(
            image=self.real_img,
            model=self.model,
            nsamples=15,
            device=self.device
        )
        pos_img = create_positive_map(res["positive_map"], orig_size=(orig_w, orig_h))
        neg_img = create_negative_map(res["negative_map"], orig_size=(orig_w, orig_h))
        comb_img = create_combined_map(res["combined_map"], orig_size=(orig_w, orig_h))

        self.assertEqual(pos_img.size, (orig_w, orig_h))
        self.assertEqual(neg_img.size, (orig_w, orig_h))
        self.assertEqual(comb_img.size, (orig_w, orig_h))

    # Test 8 — Target class support
    def test_8_target_class_support(self):
        """Test 8: Verify valid target classes 0, 1, 2, 3, 4 are properly handled."""
        for c in range(5):
            res = generate_shap_explanation(
                image=self.real_img,
                model=self.model,
                target_class=c,
                nsamples=10,
                device=self.device
            )
            self.assertEqual(res["target_class"], c)
            self.assertEqual(res["target_label"], CLASS_NAMES[c])

        with self.assertRaises(ValueError):
            generate_shap_explanation(
                image=self.real_img,
                model=self.model,
                target_class=9,
                nsamples=5,
                device=self.device
            )

    # Test 9 — Predicted class explanation
    def test_9_predicted_class_explanation(self):
        """Test 9: Verify default SHAP explanation targets the actual predicted class."""
        res = generate_shap_explanation(
            image=self.real_img,
            model=self.model,
            target_class=None,
            nsamples=15,
            device=self.device
        )
        self.assertEqual(res["target_class"], res["prediction_class"])
        self.assertEqual(res["target_label"], res["prediction_label"])

    # Test 10 — Different target classes produce distinct explanations
    def test_10_different_target_classes(self):
        """Test 10: Verify explaining different classes produces distinct directional attributions."""
        res_c0 = generate_shap_explanation(
            image=self.real_img,
            model=self.model,
            target_class=0,
            nsamples=15,
            device=self.device
        )
        res_c2 = generate_shap_explanation(
            image=self.real_img,
            model=self.model,
            target_class=2,
            nsamples=15,
            device=self.device
        )
        diff = np.abs(res_c0["combined_map"] - res_c2["combined_map"])
        self.assertGreater(np.mean(diff), 1e-4, "Attributions for Class 0 and Class 2 must differ")

    # Test 11 — Real retinal image test
    def test_11_real_retinal_image(self):
        """Test 11: End-to-end execution on unseen fundus_scan.jpg."""
        res = generate_shap_explanation(
            image=self.fundus_img_path,
            nsamples=20,
            device=self.device
        )
        self.assertIn("prediction_class", res)
        self.assertIn("shap_stats", res)
        self.assertGreater(res["shap_stats"]["computation_time_seconds"], 0.0)

    # Test 12 — Backend API endpoint
    def test_12_backend_api_endpoint(self):
        """Test 12: Verify Flask backend /api/explain/shap endpoint returns valid JSON."""
        from backend.app import app
        client = app.test_client()

        with open(self.fundus_img_path, "rb") as f:
            img_bytes = f.read()

        resp = client.post(
            "/api/explain/shap?nsamples=15",
            data=img_bytes,
            content_type="application/octet-stream"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("prediction", data)
        self.assertIn("visualizations", data)
        self.assertIn("positive", data["visualizations"])
        self.assertIn("negative", data["visualizations"])
        self.assertIn("overlay", data["visualizations"])
        self.assertIn("disclaimer", data)


if __name__ == "__main__":
    unittest.main()
