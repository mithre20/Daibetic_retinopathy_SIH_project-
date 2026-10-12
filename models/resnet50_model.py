import os
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

CLASS_NAMES = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

CLASS_KEYS = [
    "no_dr",
    "mild",
    "moderate",
    "severe",
    "proliferative"
]

def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class ResNet50DRClassifier(nn.Module):
    """
    ResNet-50 Transfer Learning model for 5-class Diabetic Retinopathy screening.
    Classes:
        0 = No Diabetic Retinopathy
        1 = Mild Diabetic Retinopathy
        2 = Moderate Diabetic Retinopathy
        3 = Severe Diabetic Retinopathy
        4 = Proliferative Diabetic Retinopathy

    Explainability Note:
        The target convolutional layer for downstream Grad-CAM is `self.backbone.layer4[-1].conv3`
        (or `self.backbone.layer4`), which is the final residual block prior to AdaptiveAvgPool2d.
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.3):
        super(ResNet50DRClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet50 backbone (transfer learning)
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)
            
        # Replace original ImageNet 1000-class fc layer with a 5-class classification head
        in_features = self.backbone.fc.in_features # 2048 for ResNet50
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self):
        """
        Returns the final convolutional layer of ResNet-50 for Grad-CAM feature mapping.
        Layer Path: self.backbone.layer4[-1].conv3
        """
        return self.backbone.layer4[-1].conv3

    def freeze_backbone(self):
        """Freeze all layers except the classification head."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all layers for full network fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_resnet50(num_classes: int = 5, pretrained: bool = True, weights_path: str = None, device: torch.device = None) -> ResNet50DRClassifier:
    """
    Factory function to initialize ResNet-50 and optionally load trained weights.
    """
    if device is None:
        device = get_device()
        
    model = ResNet50DRClassifier(num_classes=num_classes, pretrained=pretrained)
    
    if weights_path is not None and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"Loaded ResNet-50 weights from: {weights_path}")
        
    model.to(device)
    return model
