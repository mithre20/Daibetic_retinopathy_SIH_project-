import os
import sys
import json
import argparse
from typing import Union, Dict, Any
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from torchvision import transforms

# Ensure root in path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.resnet50_model import build_resnet50, get_device, CLASS_NAMES, CLASS_KEYS

class ResNet50InferenceEngine:
    """
    Inference Engine for ResNet-50 Diabetic Retinopathy screening model.
    Applies deterministic validation preprocessing:
      1. Resize to (224, 224)
      2. ToTensor()
      3. Normalize with ImageNet mean & std
    """
    def __init__(self, weights_path: str = None, device: torch.device = None):
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        if weights_path is None:
            default_p = os.path.join(ROOT_DIR, "models", "resnet50_dr.pth")
            if os.path.exists(default_p):
                weights_path = default_p

        self.weights_path = weights_path
        self.is_trained = weights_path is not None and os.path.exists(weights_path)
        
        if not self.is_trained:
            print("Notice: ResNet-50 model weights not found. Using pretrained backbone baseline.")

        self.model = build_resnet50(
            num_classes=5,
            pretrained=not self.is_trained,
            weights_path=weights_path if self.is_trained else None,
            device=self.device
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        """Validates and preprocesses image into a normalized tensor."""
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Image not found at: {image_input}")
            try:
                img = Image.open(image_input).convert("RGB")
            except (UnidentifiedImageError, OSError) as e:
                raise ValueError(f"Invalid or corrupted image format: {e}")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return self.transform(img).unsqueeze(0) # (1, 3, 224, 224)

    def predict(self, image_input: Union[str, Image.Image]) -> Dict[str, Any]:
        """
        Runs ResNet-50 prediction and returns formatted JSON dictionary.
        """
        tensor = self.preprocess(image_input).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])

        prob_dict = {
            CLASS_KEYS[i]: round(float(probs[i]), 4)
            for i in range(5)
        }

        return {
            "model": "ResNet50",
            "prediction": pred_idx,
            "severity": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 4),
            "probabilities": prob_dict
        }

def run_prediction(image_path: str, weights_path: str = None):
    engine = ResNet50InferenceEngine(weights_path=weights_path)
    result = engine.predict(image_path)
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResNet-50 DR inference on a fundus image")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal fundus image")
    parser.add_argument("--weights", type=str, default=None, help="Path to resnet50_dr.pth")
    args = parser.parse_args()

    run_prediction(args.image, args.weights)
