import os
import sys
import argparse
import copy
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score

# Ensure root is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.model.efficientnet_dr import EfficientNetDRClassifier, get_device
from training.dataset import create_data_loaders, DEFAULT_DATASET_PATH

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1

def train_model(
    data_dir: str = DEFAULT_DATASET_PATH,
    epochs: int = 15,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    freeze_backbone: bool = False,
    output_dir: str = None
):
    device = get_device()
    print("=" * 60)
    print(" Diabetic Retinopathy Screening — EfficientNet-B0 Training")
    print(f" Device: {device} ({'GPU Acceleration' if device.type == 'cuda' else 'CPU'})")
    print(f" Dataset Path: {data_dir}")
    print("=" * 60)

    # 1. Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size
    )

    # 2. Build model with transfer learning
    model = EfficientNetDRClassifier(num_classes=5, pretrained=True)
    if freeze_backbone:
        print("Freezing EfficientNet-B0 feature backbone; training classifier head only.")
        model.freeze_backbone()
    model.to(device)

    # 3. Class-weighted Cross-Entropy loss for handling severe DR imbalance
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    print(f"Class imbalance weights: {[round(w, 2) for w in class_weights]}")

    # 4. Optimizer and scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Output paths
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "backend", "model")
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(ROOT_DIR, "training", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_weights = None
    epochs_no_improve = 0

    print("\nStarting Transfer Learning Fine-Tuning...\n")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr_curr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Train Macro-F1: {train_f1:.3f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro-F1: {val_f1:.3f} | LR: {lr_curr:.1e}")

        # Checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            # Save checkpoint
            ckpt_path = os.path.join(checkpoints_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_weights,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_acc": val_acc
            }, ckpt_path)
            
            # Save production weights for inference engine
            prod_weights_path = os.path.join(output_dir, "model_weights.pth")
            torch.save(best_weights, prod_weights_path)

            # Also save to canonical models directory matching resnet50_dr.pth
            canonical_path = os.path.join(models_dir, "efficientnet_b0_dr.pth")
            torch.save(best_weights, canonical_path)
            print(f"  --> Saved new best model checkpoint (Val Loss: {val_loss:.4f}, Val F1: {val_f1:.3f})")
            print(f"      Saved to: {prod_weights_path}")
            print(f"      Saved to: {canonical_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without validation improvement.")
                break

    print("\n" + "=" * 60)
    print(" Training Complete!")
    print(f" Best Validation Loss: {best_val_loss:.4f} | Best Validation Macro-F1: {best_val_f1:.3f}")
    print(f" Model weights saved to: {os.path.join(output_dir, 'model_weights.pth')}")
    print(f" Canonical model saved to: {os.path.join(models_dir, 'efficientnet_b0_dr.pth')}")
    print("=" * 60)

    return model, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 on Diabetic Retinopathy Dataset")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_PATH, help="Path to fundus dataset")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--freeze_backbone", action="store_true", default=False, help="Freeze backbone features and train classifier head only")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save model weights")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
        output_dir=args.output_dir
    )
