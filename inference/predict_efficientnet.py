"""
Inference script for EfficientNet-B0 Diabetic Retinopathy screening model.
Usage:
    python inference/predict_efficientnet.py --image path/to/fundus.jpg [--model models/efficientnet_b0_dr.pth]
"""

import os
import sys
import json
import argparse
from typing import Union, Dict, Any
from PIL import Image
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.inference import DRInferenceEngine
from backend.model.efficientnet_dr import get_device, CLASS_NAMES, CLASS_KEYS

def run_prediction(image_path: str, weights_path: str = None) -> Dict[str, Any]:
    if not os.path.isfile(image_path):
        print(f"INPUT IMAGE NOT FOUND: {image_path}")
        sys.exit(1)

    # Check weights
    if weights_path is None:
        candidates = [
            os.path.join(ROOT_DIR, "models", "efficientnet_b0_dr.pth"),
            os.path.join(ROOT_DIR, "backend", "model", "model_weights.pth"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                weights_path = c
                break

    if weights_path is None or not os.path.isfile(weights_path):
        print("EFFICIENTNET MODEL NOT FOUND")
        sys.exit(1)

    device = get_device()
    engine = DRInferenceEngine(weights_path=weights_path, device=device)
    result = engine.predict(image_path)
    result["model"] = "EfficientNet-B0"
    result["image"] = os.path.basename(image_path)

    print("\n" + "=" * 50)
    print(" EfficientNet-B0 Diabetic Retinopathy Prediction")
    print("=" * 50)
    print(f"Image:          {result['image']}")
    print(f"Predicted DR:   {result['severity']} (Class {result['prediction']})")
    print(f"Confidence:     {result['confidence'] * 100:.1f}%")
    print(f"Referable DR:   {'YES (Specialist Referral Required)' if result['referable'] else 'No'}")
    print("\nClass Probabilities:")
    for k, v in result["probabilities"].items():
        print(f"  {k.replace('_', ' ').title():<18}: {v * 100:.2f}%")
    print(f"\nRecommendation: {result['screening_recommendation']}")
    print(f"\nDisclaimer:     {result['disclaimer']}")
    print("=" * 50)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Diabetic Retinopathy grade using EfficientNet-B0")
    parser.add_argument("--image", type=str, required=True, help="Path to fundus photograph")
    parser.add_argument("--model", type=str, default=None, help="Path to trained EfficientNet-B0 checkpoint")
    args = parser.parse_args()

    run_prediction(args.image, args.model)
