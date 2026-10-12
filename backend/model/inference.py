import io
import os
from typing import Union, Dict, Any
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from .efficientnet_dr import (
    EfficientNetDRClassifier,
    build_model,
    get_device,
    CLASS_NAMES,
    CLASS_KEYS
)

RECOMMENDATIONS = {
    0: "No Diabetic Retinopathy detected. Recommend routine annual eye screening.",
    1: "Mild NPDR detected (microaneurysms present). Recommend follow-up screening in 6–12 months with blood sugar control.",
    2: "Moderate NPDR detected. Recommend comprehensive ophthalmology evaluation within 4–8 weeks.",
    3: "Severe NPDR detected. High risk of progression to proliferative DR. Prompt ophthalmology referral within 2–4 weeks.",
    4: "Proliferative Diabetic Retinopathy (PDR) suspected. Urgent referral to a retinal specialist/ophthalmologist within 24–48 hours for immediate intervention (e.g. laser photocoagulation or anti-VEGF)."
}

DISCLAIMER = (
    "AI-assisted screening result only; this is NOT a definitive medical diagnosis. "
    "All findings must be confirmed by a qualified ophthalmologist or eye-care professional."
)

def get_inference_transforms():
    """
    Standard preprocessing transforms matching validation pipeline:
    1. Resize to (224, 224)
    2. Convert to PyTorch Tensor in [0, 1]
    3. Normalize with ImageNet standard mean & std
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class DRInferenceEngine:
    def __init__(self, weights_path: str = None, device: torch.device = None):
        if device is None:
            self.device = get_device()
        else:
            self.device = device
            
        # Determine weights path if not specified
        if weights_path is None:
            default_weights = os.path.join(os.path.dirname(__file__), "model_weights.pth")
            if os.path.exists(default_weights):
                weights_path = default_weights
                
        self.weights_path = weights_path
        self.transform = get_inference_transforms()
        
        # Build model
        self.model = build_model(
            num_classes=5,
            pretrained=True if weights_path is None else False,
            weights_path=weights_path if (weights_path and os.path.exists(weights_path)) else None,
            device=self.device
        )
        self.model.eval()

    def preprocess_image(self, image_input: Union[str, bytes, io.BytesIO, Image.Image]) -> torch.Tensor:
        """
        Loads and validates an image from filepath, raw bytes, BytesIO stream, or PIL Image.
        Returns preprocessed 4D tensor: (1, 3, 224, 224).
        Raises ValueError if image is invalid, corrupted, or empty.
        """
        if image_input is None:
            raise ValueError("No image input provided.")

        try:
            if isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
                return self.transform(img).unsqueeze(0)
            elif isinstance(image_input, (bytes, bytearray)):
                if len(image_input) == 0:
                    raise ValueError("Empty image byte data received.")
                img = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, io.BytesIO):
                image_input.seek(0)
                img = Image.open(image_input)
            elif isinstance(image_input, str):
                if not os.path.isfile(image_input):
                    raise ValueError(f"Image file not found: {image_input}")
                img = Image.open(image_input)
            else:
                # File-like object (e.g., werkzeug FileStorage)
                if hasattr(image_input, "read"):
                    data = image_input.read()
                    if len(data) == 0:
                        raise ValueError("Uploaded file is empty.")
                    img = Image.open(io.BytesIO(data))
                else:
                    raise ValueError(f"Unsupported image input type: {type(image_input)}")

            # Verify and convert to RGB
            img.verify()
            
            # After verify(), re-open to load image data safely
            if isinstance(image_input, (bytes, bytearray)):
                img = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, str):
                img = Image.open(image_input).convert("RGB")
            elif hasattr(image_input, "seek"):
                image_input.seek(0)
                img = Image.open(image_input).convert("RGB")
            else:
                img = img.convert("RGB")
                
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ValueError(f"Invalid or corrupted image format. Please provide a valid JPEG or PNG fundus scan. Details: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to process image: {str(e)}")

        tensor = self.transform(img).unsqueeze(0) # (1, 3, 224, 224)
        return tensor

    def predict(self, image_input: Union[str, bytes, io.BytesIO, Image.Image]) -> Dict[str, Any]:
        """
        Performs inference on a retinal fundus image.
        Returns dictionary conforming exactly to the required JSON schema:
        {
          "prediction": 2,
          "severity": "Moderate Diabetic Retinopathy",
          "confidence": 0.924,
          "probabilities": {
              "no_dr": 0.021,
              "mild": 0.033,
              "moderate": 0.924,
              "severe": 0.017,
              "proliferative": 0.005
          },
          "referable": true,
          "screening_recommendation": "...",
          "disclaimer": "..."
        }
        """
        tensor = self.preprocess_image(image_input).to(self.device)
        
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
            "prediction": pred_idx,
            "severity": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "referable": bool(pred_idx >= 2),
            "screening_recommendation": RECOMMENDATIONS[pred_idx],
            "disclaimer": DISCLAIMER,
            "device": str(self.device)
        }
