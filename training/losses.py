"""
Loss functions for retinal lesion segmentation:
- Multi-class Dice Loss
- Combined Loss (Cross-Entropy + Dice Loss)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss with Laplace smoothing.
    Works directly with raw model logits [B, C, H, W] and target class indices [B, H, W].
    """

    def __init__(self, smooth: float = 1e-5, ignore_background: bool = False, weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape [B, C, H, W] (raw unnormalized logits)
            targets: Tensor of shape [B, H, W] (ground truth class indices in {0, ..., C-1})
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # Convert targets to one-hot: [B, C, H, W]
        targets_one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        start_idx = 1 if self.ignore_background else 0
        total_loss = 0.0
        classes_counted = 0

        for c in range(start_idx, num_classes):
            p_c = probs[:, c, :, :]
            t_c = targets_one_hot[:, c, :, :]

            intersection = 2.0 * torch.sum(p_c * t_c) + self.smooth
            cardinality = torch.sum(p_c) + torch.sum(t_c) + self.smooth
            dice_c = intersection / cardinality

            loss_c = 1.0 - dice_c
            if self.weights is not None:
                loss_c = loss_c * self.weights[c]

            total_loss = total_loss + loss_c
            classes_counted += 1

        if classes_counted == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return total_loss / classes_counted


class CombinedLoss(nn.Module):
    """
    Combined segmentation loss:
        Loss = alpha * CrossEntropyLoss + beta * DiceLoss
    Default: alpha = 0.5, beta = 0.5
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        class_weights: Optional[torch.Tensor] = None,
        dice_smooth: float = 1e-5
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(smooth=dice_smooth, weights=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        return self.alpha * loss_ce + self.beta * loss_dice
