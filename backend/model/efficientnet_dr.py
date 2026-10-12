import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

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

class EfficientNetDRClassifier(nn.Module):
    """
    EfficientNet-B0 transfer learning model for 5-class Diabetic Retinopathy classification.
    ICDR Grades:
        0 = No Diabetic Retinopathy
        1 = Mild Diabetic Retinopathy
        2 = Moderate Diabetic Retinopathy
        3 = Severe Diabetic Retinopathy
        4 = Proliferative Diabetic Retinopathy
    """
    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout_rate: float = 0.3):
        super(EfficientNetDRClassifier, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained EfficientNet-B0 backbone (do not train from scratch)
        if pretrained:
            weights = EfficientNet_B0_Weights.DEFAULT
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)
            
        # Replace the original 1000-class classifier head with a 5-class head
        # EfficientNet-B0 classifier input feature dimension is 1280
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_gradcam_target_layer(self, conv_only: bool = True):
        """
        Returns the final convolutional feature layer of EfficientNet-B0 before global pooling.
        If conv_only is True, returns the Conv2d submodule: backbone.features[8][0]
        Otherwise returns the full Conv2dNormActivation block: backbone.features[8]
        Feature output dimensions: [B, 1280, 7, 7]
        """
        if conv_only:
            return self.backbone.features[8][0]
        return self.backbone.features[8]

    def freeze_backbone(self):
        """Freezes all feature extractor layers, leaving only the classifier trainable."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes all parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_model(num_classes: int = 5, pretrained: bool = True, weights_path: str = None, device: torch.device = None) -> EfficientNetDRClassifier:
    """
    Factory function to initialize and optionally load model weights.
    """
    if device is None:
        device = get_device()
        
    model = EfficientNetDRClassifier(num_classes=num_classes, pretrained=pretrained)
    
    if weights_path is not None:
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        # Support loading directly state_dict or wrapped in dict {'model_state_dict': ...}
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        
    model.to(device)
    return model
