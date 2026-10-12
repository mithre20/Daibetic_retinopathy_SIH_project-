"""
U-Net Retinal Lesion Segmentation Training Pipeline (IDRiD)

Supported Lesion Classes:
    0 = Background
    1 = Microaneurysms
    2 = Hemorrhages
    3 = Hard Exudates
    4 = Soft Exudates
"""

import os
import sys
import json
import argparse
import copy
import time
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.unet_model import UNet, get_device, LESION_CLASSES
from training.losses import CombinedLoss
from training.dataset_unet import (
    DEFAULT_IDRID_PATH,
    locate_idrid_dataset,
    audit_idrid_dataset,
    IDRiDLesionDataset,
    create_idrid_splits,
    calculate_training_class_weights
)


def compute_epoch_metrics(logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 5) -> Tuple[float, float]:
    """Computes Mean Dice and Mean IoU over batch."""
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    dices = []
    ious = []

    # Calculate for all classes
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)

        intersection = (pred_c & target_c).sum().float().item()
        union = (pred_c | target_c).sum().float().item()
        cardinality = pred_c.sum().float().item() + target_c.sum().float().item()

        dice_c = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
        iou_c = (intersection + 1e-5) / (union + 1e-5)

        dices.append(dice_c)
        ious.append(iou_c)

    return float(sum(dices) / len(dices)), float(sum(ious) / len(ious))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_samples = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dice, iou = compute_epoch_metrics(logits, masks)
        running_dice += dice * batch_size
        running_iou += iou * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_dice = running_dice / max(total_samples, 1)
    epoch_iou = running_iou / max(total_samples, 1)
    return epoch_loss, epoch_dice, epoch_iou


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item() * batch_size
            dice, iou = compute_epoch_metrics(logits, masks)
            running_dice += dice * batch_size
            running_iou += iou * batch_size
            total_samples += batch_size

    val_loss = running_loss / max(total_samples, 1)
    val_dice = running_dice / max(total_samples, 1)
    val_iou = running_iou / max(total_samples, 1)
    return val_loss, val_dice, val_iou


def train_unet(
    dataset_path: str = DEFAULT_IDRID_PATH,
    epochs: int = 50,
    batch_size: int = 4,
    img_size: int = 512,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    patience: int = 7,
    output_model_path: Optional[str] = None
) -> Dict[str, Any]:
    # Reproducibility seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = get_device()
    print("=" * 70)
    print(" U-Net Retinal Lesion Segmentation Training")
    print(f" Target Classes (5): {list(LESION_CLASSES.values())}")
    print(f" Image Resolution   : {img_size}x{img_size}")
    print(f" Learning Rate      : {learning_rate}")
    print(f" Optimizer          : AdamW (weight_decay={weight_decay})")
    print(f" Random Seed        : {seed}")
    print("=" * 70)

    # 1. Dataset Discovery & Audit
    resolved_path = dataset_path
    if not os.path.exists(resolved_path):
        discovered = locate_idrid_dataset()
        if discovered:
            resolved_path = discovered
            print(f"Located IDRiD dataset at alternative path: {resolved_path}")

    audit_result = audit_idrid_dataset(resolved_path)
    if audit_result.get("status") != "FOUND":
        print("\n" + "!" * 70)
        print(" TRAINING CANNOT PROCEED: DATASET NOT FOUND")
        print(" The real IDRiD segmentation dataset is required. Training has not been fabricated.")
        print("!" * 70)
        return {"status": "DATASET_NOT_FOUND", "message": audit_result.get("error")}

    fundus_images = audit_result["fundus_images"]
    split_save_path = os.path.join(ROOT_DIR, "results", "unet_split_info.json")
    train_paths, val_paths, test_paths = create_idrid_splits(
        fundus_images,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        save_path=split_save_path
    )

    print(f"\nDataset Splits (Zero Data Leakage):")
    print(f"  Training Split   : {len(train_paths)} images")
    print(f"  Validation Split : {len(val_paths)} images")
    print(f"  Test Split       : {len(test_paths)} images")

    # 2. Datasets and DataLoaders
    train_dataset = IDRiDLesionDataset(train_paths, resolved_path, img_size=img_size, is_train=True)
    val_dataset = IDRiDLesionDataset(val_paths, resolved_path, img_size=img_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 3. Class Weights (computed ONLY from training set)
    print("\nCalculating class weights from training set...")
    class_weights = calculate_training_class_weights(train_dataset, num_classes=5).to(device)

    # 4. Architecture & Loss
    model = UNet(in_channels=3, num_classes=5, bilinear=False).to(device)
    criterion = CombinedLoss(alpha=0.5, beta=0.5, class_weights=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    if output_model_path is None:
        output_model_path = os.path.join(ROOT_DIR, "models", "unet_retinal_lesion.pth")
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    best_val_dice = 0.0
    best_epoch = 0
    best_weights = None
    epochs_no_improve = 0

    print("\nStarting U-Net Training Loop...\n")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_loss, t_dice, t_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_dice, v_iou = validate(model, val_loader, criterion, device)
        scheduler.step(v_dice)

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {t_loss:.4f} | Train Mean Dice: {t_dice:.4f} | Train Mean IoU: {t_iou:.4f} || "
              f"Val Loss: {v_loss:.4f} | Val Mean Dice: {v_dice:.4f} | Val Mean IoU: {v_iou:.4f} | LR: {curr_lr:.1e}")

        # Checkpointing if best validation Dice score
        if v_dice > best_val_dice:
            best_val_dice = v_dice
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            # Save checkpoint
            torch.save({
                "epoch": best_epoch,
                "model_state_dict": best_weights,
                "val_dice": best_val_dice,
                "val_iou": v_iou,
                "val_loss": v_loss,
                "num_classes": 5,
                "img_size": img_size,
                "seed": seed
            }, output_model_path)
            print(f"  --> Saved new best U-Net checkpoint (Val Mean Dice: {best_val_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without validation Dice improvement.")
                break

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(" U-Net Training Complete!")
    print(f" Elapsed Time           : {total_time:.1f}s")
    print(f" Best Epoch             : {best_epoch}")
    print(f" Best Validation Mean Dice: {best_val_dice:.4f}")
    print(f" Best Model Saved to   : {output_model_path}")
    print("=" * 70)

    # Save model configuration metadata
    config_path = os.path.join(os.path.dirname(output_model_path), "unet_config.json")
    config_data = {
        "model_architecture": "Standard U-Net (4 Encoder, 1 Bottleneck, 4 Decoder blocks)",
        "num_classes": 5,
        "class_mapping": LESION_CLASSES,
        "input_resolution": [img_size, img_size],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "dataset": "IDRiD (Indian Diabetic Retinopathy Image Dataset)",
        "dataset_path": resolved_path,
        "random_seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_trained": epochs,
        "loss_function": "Combined (0.5 * CrossEntropy + 0.5 * Multi-class Dice Loss)",
        "optimizer": "AdamW",
        "weight_decay": weight_decay,
        "splits": {
            "train_count": len(train_paths),
            "val_count": len(val_paths),
            "test_count": len(test_paths)
        },
        "best_epoch": best_epoch,
        "best_val_mean_dice": round(float(best_val_dice), 4),
        "device": str(device)
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved configuration metadata to: {config_path}")

    return {
        "status": "SUCCESS",
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "model_path": output_model_path,
        "config_path": config_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net on IDRiD Retinal Lesion Segmentation Dataset")
    parser.add_argument("--dataset", type=str, default=DEFAULT_IDRID_PATH, help="Path to IDRiD segmentation dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--img-size", type=int, default=512, help="Input image dimension (512 or 256)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--output", type=str, default=None, help="Output .pth file path")
    args = parser.parse_args()

    train_unet(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        patience=args.patience,
        output_model_path=args.output
    )
