import os
import sys
import io
import unittest
import numpy as np
from PIL import Image
import torch

# Ensure root in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.efficientnet_dr import EfficientNetDRClassifier, build_model, get_device
from backend.model.inference import DRInferenceEngine, get_inference_transforms
from backend.app import app

class TestEfficientNetDRModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = get_device()
        cls.engine = DRInferenceEngine(device=torch.device("cpu"))
        cls.app_client = app.test_client()

        # Create a valid synthetic retinal fundus test image in memory
        cls.valid_image_bytes = cls._create_synthetic_fundus_image()

    @staticmethod
    def _create_synthetic_fundus_image():
        """Creates an in-memory JPEG fundus-like image (reddish circle on black background)."""
        w, h = 256, 256
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        y, x = np.ogrid[:h, :w]
        center_x, center_y = w // 2, h // 2
        radius = 110
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        arr[mask] = [180, 50, 20] # fundus reddish color
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_model_architecture_and_output_shape(self):
        """Test model initialization, 5-class head, and forward pass output shape."""
        model = EfficientNetDRClassifier(num_classes=5, pretrained=False)
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 5), "Model forward pass must output tensor of shape (batch_size, 5)")

    def test_image_preprocessing(self):
        """Test preprocessing produces normalized (1, 3, 224, 224) tensor."""
        tensor = self.engine.preprocess_image(self.valid_image_bytes)
        self.assertEqual(tensor.shape, (1, 3, 224, 224), "Preprocessed tensor must have shape (1, 3, 224, 224)")
        self.assertIsInstance(tensor, torch.Tensor)

    def test_invalid_image_handling(self):
        """Test that invalid/corrupt image inputs raise ValueError."""
        # 1. Empty bytes
        with self.assertRaises(ValueError):
            self.engine.preprocess_image(b"")

        # 2. Corrupt non-image text bytes
        with self.assertRaises(ValueError):
            self.engine.preprocess_image(b"this is definitely not a fundus image")

        # 3. None input
        with self.assertRaises(ValueError):
            self.engine.preprocess_image(None)

    def test_inference_prediction_structure(self):
        """Test prediction output schema and probability values."""
        result = self.engine.predict(self.valid_image_bytes)
        self.assertIn("prediction", result)
        self.assertIn("severity", result)
        self.assertIn("confidence", result)
        self.assertIn("probabilities", result)
        self.assertIn("referable", result)
        self.assertIn("screening_recommendation", result)
        self.assertIn("disclaimer", result)

        self.assertIn(result["prediction"], [0, 1, 2, 3, 4])
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)

        # Check all 5 class probability keys exist
        expected_keys = ["no_dr", "mild", "moderate", "severe", "proliferative"]
        for k in expected_keys:
            self.assertIn(k, result["probabilities"])
            self.assertTrue(0.0 <= result["probabilities"][k] <= 1.0)

        # Probabilities should sum to approximately 1.0
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_api_health_endpoint(self):
        """Test GET /api/health returns healthy status."""
        response = self.app_client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("classes"), 5)

    def test_api_predict_endpoint_success(self):
        """Test POST /api/predict with valid image file upload."""
        data = {
            "image": (io.BytesIO(self.valid_image_bytes), "test_fundus.jpg")
        }
        response = self.app_client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("prediction", json_data)
        self.assertIn("severity", json_data)
        self.assertIn("confidence", json_data)
        self.assertIn("probabilities", json_data)
        self.assertEqual(len(json_data["probabilities"]), 5)

    def test_api_predict_endpoint_invalid_file(self):
        """Test POST /api/predict with corrupt non-image file returns HTTP 400."""
        data = {
            "image": (io.BytesIO(b"random text file content"), "fake.txt")
        }
        response = self.app_client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)

    def test_api_predict_endpoint_missing_file(self):
        """Test POST /api/predict without image data returns HTTP 400."""
        response = self.app_client.post("/api/predict", data={})
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertIn("error", json_data)

if __name__ == "__main__":
    unittest.main()
