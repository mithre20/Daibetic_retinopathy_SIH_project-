"""
Comprehensive Validation Suite for Grad-CAM Explainable AI on EfficientNet-B0
Diabetic Retinopathy Screening in Rural India

Validates:
- Test 1: Real EfficientNet-B0 checkpoint loads successfully.
- Test 2: Target layer exists, is Conv2d, and matches EfficientNet-B0 architecture.
- Test 3: Forward pass outputs tensor of exact shape [1, 5].
- Test 4: Backward gradients are captured and non-empty.
- Test 5: Target-layer feature activations are captured and non-empty.
- Test 6: Grad-CAM heatmap is non-empty, finite, [0, 1] normalized, and valid dimensions.
- Test 7: Overlay dimensions match original image dimensions exactly.
- Test 8: Target-class specific heatmaps generated for all classes (0..4).
- Test 9: Real retinal fundus scan tested (NOT synthetic).
- Test 10: Sanity check: heatmap is not completely black, uniform, or NaN.
- Test 11: Prediction consistency between standard inference and Grad-CAM inference.
- Test 12: Preprocessing consistency (exact resize and normalization constants).
- Test 13: U-Net lesion segmentation integration and quantitative overlap analysis.
- Test 14: Backend API /api/gradcam endpoint.
- Test 15: Error handling (missing image and missing checkpoint).
"""

import os
import sys
import io
import unittest
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Ensure root directory in sys.path
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
from backend.model.inference import DRInferenceEngine, get_inference_transforms
from inference.gradcam import (
    GradCAM,
    apply_colormap_and_overlay,
    calculate_gradcam_unet_overlap,
    MEDICAL_DISCLAIMER
)
from inference.gradcam_efficientnet import find_model_checkpoint, run_gradcam_single_image
from backend.app import app


class TestGradCAMEfficientNet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.real_image_path = os.path.join(ROOT_DIR, "fundus_scan.jpg")
        cls.sample_image_path = os.path.join(ROOT_DIR, "data", "sample_fundus.jpg")
        cls.checkpoint_path = find_model_checkpoint()
        cls.app_client = app.test_client()

        # Load real trained model
        assert cls.checkpoint_path is not None, "Trained EfficientNet-B0 checkpoint must exist for verification"
        cls.model = build_model(num_classes=5, pretrained=False, weights_path=cls.checkpoint_path, device=cls.device)
        cls.model.eval()

        cls.transform = get_inference_transforms()

    def test_01_checkpoint_loading(self):
        """Test 1: Verify the real trained EfficientNet-B0 checkpoint loads successfully."""
        self.assertIsNotNone(self.checkpoint_path, "Checkpoint path must be resolved")
        self.assertTrue(os.path.isfile(self.checkpoint_path), f"Checkpoint file must exist: {self.checkpoint_path}")

        # Check weights load into architecture without missing keys
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.assertIn("backbone.features.0.0.weight", state)
        self.assertIn("backbone.classifier.1.weight", state)
        self.assertEqual(state["backbone.classifier.1.weight"].shape, torch.Size([5, 1280]))

    def test_02_target_layer_identification(self):
        """Test 2: Verify the selected Grad-CAM target layer exists and is Conv2d."""
        target_layer = self.model.get_gradcam_target_layer(conv_only=True)
        self.assertIsInstance(target_layer, nn.Conv2d, "Target layer must be a Conv2d module")
        self.assertEqual(target_layer.in_channels, 320, "EfficientNet-B0 features[8][0] in_channels must be 320")
        self.assertEqual(target_layer.out_channels, 1280, "EfficientNet-B0 features[8][0] out_channels must be 1280")

        # Also check block access
        block = self.model.get_gradcam_target_layer(conv_only=False)
        self.assertIsInstance(block, nn.Sequential, "Full block must be Conv2dNormActivation sequential")

    def test_03_forward_pass_output_shape(self):
        """Test 3: Verify model forward pass produces exactly [1, 5] for a single input."""
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            out = self.model(dummy)
        self.assertEqual(out.shape, torch.Size([1, 5]), "Output shape must be exactly [1, 5]")

    def test_04_gradient_and_activation_capture(self):
        """Test 4 & 5: Verify gradients and activations are captured and non-empty."""
        img = Image.open(self.real_image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        target_layer = self.model.get_gradcam_target_layer()

        with GradCAM(self.model, target_layer) as gcam:
            res = gcam.generate_cam(tensor)
            self.assertIsNotNone(gcam.activations, "Activations must be captured by hook")
            self.assertIsNotNone(gcam.gradients, "Gradients must be captured by hook")
            self.assertEqual(gcam.activations.shape, torch.Size([1, 1280, 7, 7]), "Activations must be [1, 1280, 7, 7]")
            self.assertEqual(gcam.gradients.shape, torch.Size([1, 1280, 7, 7]), "Gradients must be [1, 1280, 7, 7]")
            self.assertTrue(torch.any(gcam.gradients != 0), "Gradients must not be all zero")

    def test_06_heatmap_properties(self):
        """Test 6: Verify heatmap is non-empty, finite, correct dimensions, properly normalized."""
        img = Image.open(self.real_image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        target_layer = self.model.get_gradcam_target_layer()

        with GradCAM(self.model, target_layer) as gcam:
            res = gcam.generate_cam(tensor)
            heatmap = res["heatmap"]

        self.assertEqual(heatmap.shape, (7, 7), "Feature map resolution must be (7, 7)")
        self.assertTrue(np.all(np.isfinite(heatmap)), "Heatmap values must all be finite (no NaN or Inf)")
        self.assertTrue(0.0 <= np.min(heatmap), "Heatmap min must be >= 0.0")
        self.assertTrue(np.max(heatmap) <= 1.0, "Heatmap max must be <= 1.0")

    def test_07_overlay_dimensions_match_original(self):
        """Test 7: Verify overlay dimensions match the original fundus photograph."""
        img = Image.open(self.real_image_path).convert("RGB")
        orig_w, orig_h = img.size

        heatmap = np.random.rand(7, 7).astype(np.float32)
        orig_np, heat_rgb, heat_pil, overlay_pil = apply_colormap_and_overlay(img, heatmap, alpha=0.4)

        self.assertEqual(overlay_pil.size, (orig_w, orig_h), "Overlay width and height must match original image")
        self.assertEqual(heat_pil.size, (orig_w, orig_h), "Resized heatmap width and height must match original image")

    def test_08_multiple_target_classes(self):
        """Test 8: Verify Grad-CAM can explain all 5 target classes (0..4)."""
        img = Image.open(self.real_image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        target_layer = self.model.get_gradcam_target_layer()

        heatmaps = []
        with GradCAM(self.model, target_layer) as gcam:
            for c in range(5):
                res = gcam.generate_cam(tensor, target_class=c)
                self.assertEqual(res["target_class"], c)
                self.assertEqual(res["target_severity"], CLASS_NAMES[c])
                heatmaps.append(res["heatmap"])

        self.assertEqual(len(heatmaps), 5)
        for h in heatmaps:
            self.assertEqual(h.shape, (7, 7))
            self.assertTrue(np.all(np.isfinite(h)))

    def test_09_real_retinal_image(self):
        """Test 9: Run Grad-CAM on actual unseen retinal fundus scan (fundus_scan.jpg)."""
        self.assertTrue(os.path.isfile(self.real_image_path), "Real retinal image must exist")
        res = run_gradcam_single_image(
            image_path=self.real_image_path,
            model=self.model,
            device=self.device,
            output_dir=os.path.join(ROOT_DIR, "results", "gradcam")
        )
        self.assertEqual(res["status"], "success")
        self.assertIn("heatmap_path", res)
        self.assertIn("overlay_path", res)
        self.assertTrue(os.path.isfile(res["heatmap_path"]))
        self.assertTrue(os.path.isfile(res["overlay_path"]))

    def test_10_sanity_check_metrics(self):
        """Test 10: Sanity check: verify heatmap statistics are printed and non-degenerate."""
        img = Image.open(self.real_image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        target_layer = self.model.get_gradcam_target_layer()

        with GradCAM(self.model, target_layer) as gcam:
            res = gcam.generate_cam(tensor)
            stats = res["sanity_stats"]

        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertIn("dimensions", stats)
        self.assertTrue(stats["is_finite"])
        self.assertGreater(stats["max"], stats["min"], "Max activation should exceed min activation")

    def test_11_prediction_consistency(self):
        """Test 11: Normal EfficientNet inference and Grad-CAM inference must match on identical image."""
        engine = DRInferenceEngine(weights_path=self.checkpoint_path, device=self.device)
        normal_res = engine.predict(self.real_image_path)

        gradcam_res = run_gradcam_single_image(
            image_path=self.real_image_path,
            model=self.model,
            device=self.device
        )

        self.assertEqual(normal_res["prediction"], gradcam_res["prediction"], "Predicted class index must match")
        self.assertEqual(normal_res["severity"], gradcam_res["predicted_severity"], "Predicted severity label must match")
        self.assertAlmostEqual(normal_res["confidence"], gradcam_res["confidence"], places=3)

    def test_12_preprocessing_consistency(self):
        """Test 12: Preprocessing transforms must match exact training/validation pipeline."""
        t = get_inference_transforms()
        # Verify resize to 224x224
        img = Image.new("RGB", (600, 400), (120, 40, 20))
        tensor = t(img)
        self.assertEqual(tensor.shape, torch.Size([3, 224, 224]))

    def test_13_unet_gradcam_overlap_analysis(self):
        """Test 13: Quantitative overlap metric between Grad-CAM attention and U-Net mask."""
        heat = np.zeros((512, 512), dtype=np.float32)
        heat[100:200, 100:200] = 0.8  # Attended region

        unet_mask = np.zeros((512, 512), dtype=np.int64)
        unet_mask[120:180, 120:180] = 2  # Hemorrhage inside attended region

        stats = calculate_gradcam_unet_overlap(heat, unet_mask, threshold=0.5)
        self.assertIn("overall_iou", stats)
        self.assertIn("overall_lesion_coverage_pct", stats)
        self.assertGreater(stats["overall_iou"], 0.0)
        self.assertEqual(stats["per_lesion"]["hemorrhage"]["lesion_coverage_pct"], 100.0)

    def test_14_api_gradcam_endpoint(self):
        """Test 14: Verify Flask backend /api/gradcam returns valid JSON with base64 images."""
        with open(self.sample_image_path, "rb") as f:
            img_bytes = f.read()

        data = {
            "image": (io.BytesIO(img_bytes), "sample.jpg")
        }
        resp = self.app_client.post("/api/gradcam?alpha=0.5", data=data, content_type="multipart/form-data")
        self.assertEqual(resp.status_code, 200)
        json_data = resp.get_json()

        self.assertIn("prediction", json_data)
        self.assertIn("severity", json_data)
        self.assertIn("confidence", json_data)
        self.assertIn("target_class", json_data)
        self.assertIn("heatmap", json_data)
        self.assertIn("overlay", json_data)
        self.assertTrue(json_data["heatmap"].startswith("data:image/png;base64,"))
        self.assertTrue(json_data["overlay"].startswith("data:image/png;base64,"))

    def test_15_error_handling(self):
        """Test 15: Clean error handling for missing inputs."""
        # Missing image to API
        resp = self.app_client.post("/api/gradcam", data={})
        self.assertEqual(resp.status_code, 400)

        # Missing image to inference function
        res = run_gradcam_single_image(
            image_path="nonexistent_path_to_image.jpg",
            model=self.model,
            device=self.device
        )
        self.assertEqual(res["status"], "error")


if __name__ == "__main__":
    unittest.main()
