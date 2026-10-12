"""
Training script for Calibrated Retinal Image Quality Assessment (IQA)
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Trains MobileNetV3-Small to accurately recognize:
  Class 0: Good Quality (clear, diagnostic fundus images with sharp vessels & optic disc)
  Class 1: Poor Quality (mild-to-moderate blur, uneven illumination, contrast loss)
  Class 2: Unusable (severe blur, extreme dark/overexposure, obstruction)
"""

import os
import sys
import glob
import random
import time
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.iqa_model import build_iqa_model, get_iqa_transforms, get_device


def make_poor_quality(img: Image.Image) -> Image.Image:
    """Simulates mild to moderate optical degradation (Poor Quality)."""
    degraded = img.copy()
    choice = random.choice(["blur", "contrast", "shadow", "mixed"])
    if choice in ["blur", "mixed"]:
        radius = random.uniform(2.5, 4.5)
        degraded = degraded.filter(ImageFilter.GaussianBlur(radius=radius))
    if choice in ["contrast", "mixed"]:
        factor = random.uniform(0.45, 0.65)
        degraded = ImageEnhance.Contrast(degraded).enhance(factor)
    if choice in ["shadow", "mixed"]:
        arr = np.array(degraded, dtype=np.float32)
        h, w = arr.shape[:2]
        if random.random() > 0.5:
            grad = np.linspace(0.3, 1.0, w, dtype=np.float32)
            grad = np.tile(grad, (h, 1))
        else:
            grad = np.linspace(0.3, 1.0, h, dtype=np.float32)
            grad = np.tile(grad[:, None], (1, w))
        arr = arr * grad[:, :, None]
        degraded = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return degraded


def make_unusable_quality(img: Image.Image) -> Image.Image:
    """Simulates severe optical degradation (Unusable)."""
    degraded = img.copy()
    choice = random.choice(["heavy_blur", "dark", "overexposed", "occlusion"])
    if choice == "heavy_blur":
        degraded = degraded.filter(ImageFilter.GaussianBlur(radius=random.uniform(8.0, 16.0)))
    elif choice == "dark":
        degraded = ImageEnhance.Brightness(degraded).enhance(random.uniform(0.08, 0.22))
    elif choice == "overexposed":
        degraded = ImageEnhance.Brightness(degraded).enhance(random.uniform(2.5, 4.0))
        degraded = ImageEnhance.Contrast(degraded).enhance(0.4)
    elif choice == "occlusion":
        arr = np.array(degraded)
        h, w = arr.shape[:2]
        patch_h = int(h * random.uniform(0.6, 0.85))
        arr[:patch_h, :] = np.random.randint(0, 15, (patch_h, w, 3), dtype=np.uint8)
        degraded = Image.fromarray(arr)
    return degraded


class RetinalQualityCalibrationDataset(Dataset):
    def __init__(self, image_paths, transform, samples_per_class=400):
        self.transform = transform
        self.samples = []
        
        selected_paths = image_paths[:samples_per_class] if len(image_paths) >= samples_per_class else image_paths
        
        # Class 0: Good Quality (clean original images)
        for p in selected_paths:
            self.samples.append((p, 0))
            
        # Class 1: Poor Quality (mildly degraded)
        for p in selected_paths:
            self.samples.append((p, 1))
            
        # Class 2: Unusable (severely degraded)
        for p in selected_paths:
            self.samples.append((p, 2))
            
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        
        if label == 1:
            img = make_poor_quality(img)
        elif label == 2:
            img = make_unusable_quality(img)
            
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)


def train_calibrated_iqa(epochs=4, batch_size=32, lr=0.001):
    device = get_device()
    print(f"Calibrating IQA MobileNetV3 model on device: {device}")
    
    # 1. Collect clinical fundus images
    pattern = os.path.join(ROOT_DIR, "DR_Screening_Pipeline", "DR_Screening_Pipeline", "data", "**", "*.jpg")
    img_paths = glob.glob(pattern, recursive=True)
    if not img_paths:
        pattern = os.path.join(ROOT_DIR, "**", "*.jpg")
        img_paths = [p for p in glob.glob(pattern, recursive=True) if "fundus" in p.lower() or "data" in p.lower()]
    
    print(f"Found {len(img_paths)} clinical fundus image candidates.")
    random.seed(42)
    random.shuffle(img_paths)
    
    # 2. Build Dataset & Loader
    transform = get_iqa_transforms(image_size=224, is_training=True)
    dataset = RetinalQualityCalibrationDataset(img_paths, transform=transform, samples_per_class=350)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Created Calibration Dataset with {len(dataset)} balanced multi-class samples.")
    
    # 3. Model & Optimizer
    # Initialize with pretrained MobileNetV3-Small backbone
    model = build_iqa_model(num_classes=3, pretrained=True, device=device)
    model.freeze_backbone()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.backbone.classifier.parameters(), lr=lr, weight_decay=1e-4)
    
    # 4. Training Loop (Warmup classifier head)
    print("\n--- Training IQA Quality Head ---")
    model.train()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}% | Time: {time.time()-t0:.1f}s")
        
    # Unfreeze top bottleneck blocks (Stage 2 fine-tuning for 2 epochs)
    print("\n--- Fine-tuning Top Feature Layers ---")
    model.unfreeze_backbone(deeper_blocks_only=True)
    fine_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr * 0.2,
        weight_decay=1e-4
    )
    for epoch in range(1, 3):
        t0 = time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            fine_optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            fine_optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            
        print(f"Fine-tune Epoch {epoch}/2 | Loss: {running_loss/total:.4f} | Accuracy: {(correct/total)*100:.2f}% | Time: {time.time()-t0:.1f}s")

    # 5. Save Calibrated Weights
    checkpoint_path = os.path.join(ROOT_DIR, "models", "retinal_quality_mobilenet.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved calibrated weights to: {checkpoint_path}")
    
    # 6. Test on fundus_scan.jpg
    model.eval()
    val_transform = get_iqa_transforms(image_size=224, is_training=False)
    test_img_path = os.path.join(ROOT_DIR, "fundus_scan.jpg")
    if os.path.isfile(test_img_path):
        test_pil = Image.open(test_img_path).convert("RGB")
        test_t = val_transform(test_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            test_logits = model(test_t)
            probs = torch.softmax(test_logits, dim=1)[0].cpu().numpy()
            
        print("\nVerification on fundus_scan.jpg:")
        print(f"  Good Quality (0): {probs[0]*100:.2f}%")
        print(f"  Poor Quality (1): {probs[1]*100:.2f}%")
        print(f"  Unusable     (2): {probs[2]*100:.2f}%")


if __name__ == "__main__":
    train_calibrated_iqa(epochs=4, batch_size=32)
