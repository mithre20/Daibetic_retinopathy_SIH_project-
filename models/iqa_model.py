"""
Retinal Image Quality Assessment (IQA) Model
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Architecture:
  Input: Retinal Fundus Photograph (3 x 224 x 224)
  Backbone: MobileNetV3-Small (Pretrained Transfer Learning)
  Head: Global Average Pooling -> Dropout(0.2) -> Linear(in_features -> 3)
  Output: 3 Classes
    0 = Good Quality  (Screening Allowed)
    1 = Poor Quality  (Screening Blocked - Recapture Needed)
    2 = Unusable      (Screening Blocked - Non-Diagnostic)
"""

import os
import sys
from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

QUALITY_CLASSES: List[str] = [
    "Good Quality",
    "Poor Quality",
    "Unusable"
]

QUALITY_KEYS: List[str] = [
    "good",
    "poor",
    "unusable"
]

IQA_CLASSES: Dict[int, str] = {
    0: "Good Quality",
    1: "Poor Quality",
    2: "Unusable"
}

IQA_CLASS_KEYS: Dict[str, int] = {
    "good": 0,
    "poor": 1,
    "unusable": 2
}

IQA_CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "Clear retinal vessels, macula, and optic disc visible with diagnostic clarity.",
    1: "Degraded image due to mild blur, uneven illumination, or minor media opacity.",
    2: "Severely degraded, out of focus, or clipped photograph unsuitable for clinical analysis."
}

DEFAULT_GOOD_THRESHOLD: float = 0.75


def get_device() -> torch.device:
    """Returns CUDA if available, MPS if available on macOS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_iqa_transforms(image_size: int = 224, is_training: bool = False) -> transforms.Compose:
    """
    Standard preprocessing for MobileNetV3 IQA:
      - Resize to (image_size, image_size)
      - Convert to Tensor in [0, 1]
      - Normalize with ImageNet standard mean & std
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class RetinalQualityModel(nn.Module):
    """
    Lightweight MobileNetV3-Small neural network for 3-class Retinal Image Quality Assessment.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.2):
        super(RetinalQualityModel, self).__init__()
        self.num_classes = num_classes

        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self.backbone = mobilenet_v3_small(weights=weights)
        else:
            self.backbone = mobilenet_v3_small(weights=None)

        # In MobileNetV3-Small, the classifier block is:
        # [0] Linear(576, 1024)
        # [1] Hardswish()
        # [2] Dropout(p=0.2, inplace=True)
        # [3] Linear(1024, 1000)
        # Replace the final linear layer (index 3) with a 3-class projection
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape [batch_size, 3]."""
        return self.backbone(x)

    def freeze_backbone(self):
        """Freezes feature extractor layers, training only the classification head."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, deeper_blocks_only: bool = True):
        """
        Unfreezes backbone parameters.
        If deeper_blocks_only is True, only unfreezes final bottleneck stages (blocks 8..11).
        """
        if deeper_blocks_only:
            # Freeze earlier stages, unfreeze last 3 bottleneck blocks
            for i, block in enumerate(self.backbone.features):
                requires_grad = (i >= 8)
                for param in block.parameters():
                    param.requires_grad = requires_grad
        else:
            for param in self.backbone.features.parameters():
                param.requires_grad = True


def build_iqa_model(
    num_classes: int = 3,
    pretrained: bool = True,
    pretrained_backbone: Optional[bool] = None,
    weights_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> RetinalQualityModel:
    """
    Factory function to initialize and load the RetinalQualityModel.
    Supports pretrained_backbone and checkpoint_path keyword aliases.
    """
    if pretrained_backbone is not None:
        pretrained = pretrained_backbone
    if checkpoint_path is not None:
        weights_path = checkpoint_path

    if device is None:
        device = get_device()

    model = RetinalQualityModel(num_classes=num_classes, pretrained=pretrained)

    if weights_path is not None and os.path.isfile(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def find_iqa_checkpoint(custom_path: Optional[str] = None) -> Optional[str]:
    """Finds available IQA checkpoint from canonical project locations."""
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    candidates = [
        os.path.join(ROOT_DIR, "models", "retinal_quality_mobilenet.pth"),
        os.path.join(ROOT_DIR, "training", "checkpoints", "best_iqa_model.pth")
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None
