"""
U-Net Architecture for Retinal Lesion Segmentation (IDRiD)
Target Classes (5-Class Formulation):
    0 = Background
    1 = Microaneurysm (MA)
    2 = Hemorrhage (HE)
    3 = Hard Exudate (EX)
    4 = Soft Exudate (SE)
"""

import os
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

LESION_CLASSES = {
    0: "Background",
    1: "Microaneurysms",
    2: "Hemorrhages",
    3: "Hard Exudates",
    4: "Soft Exudates"
}

LESION_KEYS = ["background", "microaneurysm", "hemorrhage", "hard_exudate", "soft_exudate"]

# Standard RGB color palette for lesion overlay visualization
# 0: transparent, 1: Red/Pink, 2: Deep Red/Crimson, 3: Vivid Yellow, 4: Cyan/Light Blue
LESION_COLORS = {
    0: (0, 0, 0),        # Background (transparent in overlays)
    1: (255, 50, 50),    # Microaneurysms (Red)
    2: (180, 0, 0),      # Hemorrhages (Dark Red / Crimson)
    3: (255, 230, 0),    # Hard Exudates (Vivid Yellow)
    4: (0, 220, 255)     # Soft Exudates / Cotton Wool Spots (Cyan)
}


def get_device() -> torch.device:
    """Auto-detect CUDA or CPU device as required."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Device: CUDA")
    else:
        device = torch.device("cpu")
        print("Device: CPU")
    return device


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2D then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then DoubleConv with Skip-Connection Concatenation"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        # If bilinear, use normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle potential padding difference if input dimensions are odd
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 Convolution to project feature channels to number of target classes"""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for multi-class retinal lesion segmentation:
    - 4 Downsampling blocks (Encoder)
    - 1 Bottleneck block
    - 4 Upsampling blocks with skip-connections (Decoder)
    - 1x1 OutConv producing [B, num_classes, H, W]
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        features: Optional[List[int]] = None,
        bilinear: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        if features is None:
            features = [64, 128, 256, 512]

        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor)

        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear)
        self.up2 = Up(features[3], features[2] // factor, bilinear)
        self.up3 = Up(features[2], features[1] // factor, bilinear)
        self.up4 = Up(features[1], features[0], bilinear)
        self.outc = OutConv(features[0], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def build_unet(
    in_channels: int = 3,
    num_classes: int = 5,
    features: Optional[List[int]] = None,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    bilinear: bool = False
) -> UNet:
    """Builds and optionally initializes U-Net with trained checkpoint weights."""
    if device is None:
        device = get_device()

    model = UNet(in_channels=in_channels, num_classes=num_classes, features=features, bilinear=bilinear)

    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded U-Net weights from: {weights_path}")

    model.to(device)
    return model
