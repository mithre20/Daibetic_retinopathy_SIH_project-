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

from models.resnet50_model import ResNet50DRClassifier, build_resnet50, CLASS_NAMES, CLASS_KEYS
from inference.predict_resnet50 import ResNet50InferenceEngine

class TestResNet50DRModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic in-memory test fundus image
        w, h = 224, 224
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[40:184, 40:184] = [170, 60, 30] # reddish fundus circle
        cls.test_img = Image.fromarray(arr)
        
        # Save a temporary sample image
        cls.tmp_img_path = os.path.join(ROOT_DIR, "tests", "temp_test_fundus.jpg")
        cls.test_img.save(cls.tmp_img_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.tmp_img_path):
            os.remove(cls.tmp_img_path)

    def test_1_model_creation_and_pretrained_backbone(self):
        """Verify ResNet-50 initializes with pretrained backbone and custom 5-class head."""
        model = ResNet50DRClassifier(num_classes=5, pretrained=True)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(model.backbone.fc[1].out_features, 5, "Classifier out_features must equal 5")
        self.assertEqual(model.backbone.fc[1].in_features, 2048, "ResNet-50 in_features must equal 2048")

    def test_2_model_forward_output_shape(self):
        """Verify: Input image -> ResNet50 -> output shape [1, 5]."""
        model = ResNet50DRClassifier(num_classes=5, pretrained=False)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, torch.Size([1, 5]), "Output tensor shape must be exactly [1, 5]")

    def test_3_gradcam_target_layer(self):
        """Verify that the target convolutional layer for Grad-CAM is accessible."""
        model = ResNet50DRClassifier(num_classes=5, pretrained=False)
        target_layer = model.get_gradcam_target_layer()
        self.assertIsInstance(target_layer, torch.nn.Conv2d, "Grad-CAM target layer must be a Conv2d module")

    def test_4_preprocessing_pipeline(self):
        """Verify deterministic preprocessing produces (1, 3, 224, 224) normalized tensor."""
        engine = ResNet50InferenceEngine(device=torch.device("cpu"))
        tensor = engine.preprocess(self.tmp_img_path)
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertIsInstance(tensor, torch.Tensor)

    def test_5_inference_and_probability_output(self):
        """Verify inference returns all 5 classes and probabilities sum to 1.0."""
        engine = ResNet50InferenceEngine(device=torch.device("cpu"))
        res = engine.predict(self.tmp_img_path)
        
        self.assertEqual(res["model"], "ResNet50")
        self.assertIn("prediction", res)
        self.assertIn("severity", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)

        self.assertIn(res["prediction"], [0, 1, 2, 3, 4])
        self.assertEqual(len(res["probabilities"]), 5)

        for key in CLASS_KEYS:
            self.assertIn(key, res["probabilities"])
            self.assertTrue(0.0 <= res["probabilities"][key] <= 1.0)

        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

    def test_6_correct_class_mapping(self):
        """Verify the 5 clinical class names match ICDR definitions."""
        self.assertEqual(CLASS_NAMES[0], "No Diabetic Retinopathy")
        self.assertEqual(CLASS_NAMES[1], "Mild Diabetic Retinopathy")
        self.assertEqual(CLASS_NAMES[2], "Moderate Diabetic Retinopathy")
        self.assertEqual(CLASS_NAMES[3], "Severe Diabetic Retinopathy")
        self.assertEqual(CLASS_NAMES[4], "Proliferative Diabetic Retinopathy")

    def test_7_model_weight_saving_and_loading(self):
        """Verify model weights can be saved and loaded accurately."""
        model = ResNet50DRClassifier(num_classes=5, pretrained=False)
        tmp_weights = os.path.join(ROOT_DIR, "tests", "temp_resnet50.pth")
        try:
            torch.save(model.state_dict(), tmp_weights)
            loaded_model = build_resnet50(num_classes=5, pretrained=False, weights_path=tmp_weights, device=torch.device("cpu"))
            self.assertEqual(loaded_model.backbone.fc[1].out_features, 5)
            del loaded_model
            del model
        finally:
            import gc
            import time
            gc.collect()
            if os.path.exists(tmp_weights):
                for _ in range(5):
                    try:
                        os.remove(tmp_weights)
                        break
                    except PermissionError:
                        time.sleep(0.1)
                        gc.collect()

if __name__ == "__main__":
    unittest.main()
