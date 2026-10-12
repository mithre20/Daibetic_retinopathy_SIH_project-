import os
import sys
import argparse
import copy
from collections import Counter
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ensure root directory in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.resnet50_model import ResNet50DRClassifier, get_device, CLASS_NAMES
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
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
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
    epoch_prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    epoch_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1

def train_resnet50(
    data_dir: str = DEFAULT_DATASET_PATH,
    model_save_path: str = None,
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    freeze_backbone: bool = False
):
    # Check dataset existence
    if not os.path.exists(data_dir):
        print("DATASET NOT FOUND")
        sys.exit(1)

    device = get_device()
    print("=" * 70)
    print(" RESNET-50 BENCHMARK TRAINING -- DIABETIC RETINOPATHY")
    print(f" Device: {device} ({'GPU Acceleration' if device.type == 'cuda' else 'CPU'})")
    print(f" Dataset Path: {data_dir}")
    print("=" * 70)

    if model_save_path is None:
        model_save_path = os.path.join(ROOT_DIR, "models", "resnet50_dr.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # 1. Load data splits (reuses exact same splits as EfficientNet)
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        val_split=0.15,
        test_split=0.15,
        random_seed=42
    )

    train_labels = train_loader.dataset.labels
    train_counts = Counter(train_labels)

    # Report class distribution in training dataset ONLY
    print("\nTraining Dataset Class Distribution (Weights derived strictly from training data):")
    for c in range(5):
        print(f"  Class {c} ({CLASS_NAMES[c]}): {train_counts.get(c, 0)} images")

    print(f"\nCalculated Class Weights for Loss: {[round(w, 4) for w in class_weights]}")

    # 2. Build ResNet-50 with transfer learning
    model = ResNet50DRClassifier(num_classes=5, pretrained=True)
    if freeze_backbone:
        print("Freezing ResNet-50 feature backbone; training classifier head only.")
        model.freeze_backbone()
    model.to(device)

    # 3. Class-weighted Cross-Entropy loss
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # 4. AdamW optimizer and ReduceLROnPlateau scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0

    print("\nStarting ResNet-50 Fine-Tuning...\n")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_f1)

        # Print required per-epoch metrics
        print(f"Epoch: {epoch:02d}/{epochs:02d}")
        print(f"  Training Loss:        {train_loss:.4f}")
        print(f"  Training Accuracy:    {train_acc * 100:.2f}%")
        print(f"  Validation Loss:      {val_loss:.4f}")
        print(f"  Validation Accuracy:  {val_acc * 100:.2f}%")
        print(f"  Validation Precision: {val_prec:.4f}")
        print(f"  Validation Recall:    {val_rec:.4f}")
        print(f"  Validation F1:        {val_f1:.4f}")
        print("-" * 50)

        # Save best model based on validation F1-score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0

            torch.save(best_weights, model_save_path)
            print(f"  >>> [Checkpoint] New best ResNet-50 saved (Val F1: {val_f1:.4f}) -> {model_save_path}\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without validation F1 improvement.")
                break

    print("\n" + "=" * 70)
    print(" ResNet-50 Training Complete!")
    print(f" Best Validation Macro-F1: {best_val_f1:.4f}")
    print(f" Model weights saved to:   {model_save_path}")
    print("=" * 70)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-50 on Diabetic Retinopathy Dataset")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATASET_PATH, help="Path to fundus dataset")
    parser.add_argument("--model_save_path", type=str, default=None, help="Output path for .pth model weights")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for AdamW")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for AdamW")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze feature extractor and train classifier head only")
    args = parser.parse_args()

    train_resnet50(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone
    )
