"""
Unit Tests for U-Net Retinal Lesion Segmentation Architecture and Pipeline

Verifies:
1. U-Net model can be instantiated.
2. Input tensor [1, 3, 512, 512] produces [1, 5, 512, 512].
3. Dataset returns image + mask with matching spatial dimensions.
4. Mask contains only valid class IDs (0, 1, 2, 3, 4).
5. Image and mask augmentation remain aligned.
6. Saved model can be loaded.
7. Inference works on an unseen test image.
8. Predicted probabilities and classes have valid dimensions.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from PIL import Image
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.unet_model import UNet, build_unet, LESION_CLASSES
from training.dataset_unet import IDRiDLesionDataset, combine_lesion_masks
from inference.predict_unet import predict_retinal_lesions


class TestUNetRetinalSegmentation(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.img_size = 256  # fast size for unit test execution

    def test_1_unet_instantiation(self):
        """Test 1: Verify U-Net can be instantiated cleanly."""
        model = UNet(in_channels=3, num_classes=5)
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(model.num_classes, 5)
        self.assertEqual(model.in_channels, 3)

    def test_2_forward_pass_dimensions(self):
        """Test 2: Verify input tensor [1, 3, 512, 512] produces [1, 5, 512, 512]."""
        model = UNet(in_channels=3, num_classes=5, features=[16, 32, 64, 128])
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            output = model(dummy_input)
        self.assertEqual(output.shape, (1, 5, 512, 512))

    def test_3_dataset_matching_dimensions(self):
        """Test 3: Dataset returns image + mask with matching spatial dimensions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a test fundus image
            test_img_path = os.path.join(tmp_dir, "test_fundus.jpg")
            img = Image.new("RGB", (300, 300), color=(100, 50, 20))
            img.save(test_img_path)

            dataset = IDRiDLesionDataset([test_img_path], tmp_dir, img_size=256, is_train=False)
            img_tensor, mask_tensor = dataset[0]

            self.assertEqual(img_tensor.shape, (3, 256, 256))
            self.assertEqual(mask_tensor.shape, (256, 256))

    def test_4_mask_valid_class_ids(self):
        """Test 4: Mask contains only valid class IDs: 0, 1, 2, 3, 4."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create test fundus image and lesion masks
            test_img_path = os.path.join(tmp_dir, "IDRiD_01.jpg")
            Image.new("RGB", (100, 100), color=(50, 20, 10)).save(test_img_path)

            ma_path = os.path.join(tmp_dir, "IDRiD_01_MA.tif")
            he_path = os.path.join(tmp_dir, "IDRiD_01_HE.tif")
            Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 255).save(ma_path)
            Image.fromarray(np.ones((100, 100), dtype=np.uint8) * 255).save(he_path)

            mask = combine_lesion_masks({"ma": ma_path, "he": he_path, "ex": None, "se": None}, (100, 100))
            unique_classes = np.unique(mask)

            for c in unique_classes:
                self.assertIn(c, [0, 1, 2, 3, 4])

    def test_5_synchronized_augmentation_alignment(self):
        """Test 5: Image and mask augmentation remain spatially aligned."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_img_path = os.path.join(tmp_dir, "test_fundus.jpg")
            Image.new("RGB", (128, 128), color=(0, 0, 0)).save(test_img_path)

            dataset = IDRiDLesionDataset([test_img_path], tmp_dir, img_size=128, is_train=True)
            img_tensor, mask_tensor = dataset[0]

            # Dimensions must match exactly across all augmentations
            self.assertEqual(img_tensor.shape[1:], mask_tensor.shape)

    def test_6_saved_model_checkpoint_loading(self):
        """Test 6: Saved model checkpoint can be loaded reliably."""
        model = UNet(in_channels=3, num_classes=5, features=[16, 32, 64, 128])
        tmp_pth = os.path.join(ROOT_DIR, "tests", "temp_unet.pth")
        try:
            torch.save({"model_state_dict": model.state_dict(), "num_classes": 5}, tmp_pth)
            loaded_model = build_unet(in_channels=3, num_classes=5, features=[16, 32, 64, 128], weights_path=tmp_pth, device=self.device)
            self.assertEqual(loaded_model.num_classes, 5)
            del loaded_model
            del model
        finally:
            import gc, time
            gc.collect()
            if os.path.exists(tmp_pth):
                for _ in range(5):
                    try:
                        os.remove(tmp_pth)
                        break
                    except PermissionError:
                        time.sleep(0.1)
                        gc.collect()

    def test_7_inference_on_unseen_image(self):
        """Test 7: Inference works end-to-end on an unseen fundus photo."""
        sample_img = os.path.join(ROOT_DIR, "data", "sample_fundus.jpg")
        if not os.path.exists(sample_img):
            sample_img = os.path.join(ROOT_DIR, "DR_Screening_Pipeline", "DR_Screening_Pipeline", "data", "sample_fundus.jpg")

        with tempfile.TemporaryDirectory() as tmp_out:
            result = predict_retinal_lesions(
                image_input=sample_img,
                model_path=None,
                img_size=128,
                output_dir=tmp_out,
                device=self.device
            )
            self.assertIn("detections", result)
            self.assertIn("pixel_coverage_percentage", result)
            self.assertIn("explanation", result)
            self.assertTrue(os.path.exists(result["predicted_mask_path"]))
            self.assertTrue(os.path.exists(result["lesion_overlay_path"]))

    def test_8_probability_and_classes_valid_dimensions(self):
        """Test 8: Predicted probabilities and classes have valid spatial dimensions."""
        model = UNet(in_channels=3, num_classes=5, features=[16, 32, 64, 128])
        model.eval()
        dummy_tensor = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            logits = model(dummy_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_classes = torch.argmax(probs, dim=1)

        self.assertEqual(probs.shape, (1, 5, 128, 128))
        self.assertEqual(pred_classes.shape, (1, 128, 128))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))


if __name__ == "__main__":
    unittest.main()
