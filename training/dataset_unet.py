"""
IDRiD Retinal Lesion Segmentation Dataset Loader & Preprocessing

Target Classes:
    0 = Background
    1 = Microaneurysm (MA)
    2 = Hemorrhage (HE)
    3 = Hard Exudate (EX)
    4 = Soft Exudate (SE)

Expected IDRiD Segmentation Directory Layout (from https://idrid.grand-challenge.org/Data/):
    <DATASET_DIR>/
    ├── 1. Original Images/
    │   ├── a. Training Set/
    │   │   ├── IDRiD_01.jpg ...
    │   └── b. Testing Set/
    │       ├── IDRiD_55.jpg ...
    └── 2. Groundtruths/
        ├── a. Training Set/
        │   ├── 1. Microaneurysms/ (e.g. IDRiD_01_MA.tif)
        │   ├── 2. Haemorrhages/   (e.g. IDRiD_01_HE.tif)
        │   ├── 3. Hard Exudates/  (e.g. IDRiD_01_EX.tif)
        │   └── 4. Soft Exudates/  (e.g. IDRiD_01_SE.tif)
        └── b. Testing Set/
            ├── 1. Microaneurysms/
            ├── 2. Haemorrhages/
            ├── 3. Hard Exudates/
            └── 4. Soft Exudates/
"""

import os
import glob
import json
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

DEFAULT_IDRID_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "idrid_segmentation")
)

# Standard ImageNet normalization parameters
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

# Conflict resolution priority (highest index overrides lower index on spatial overlap)
# Priority: Soft Exudates (4) > Hard Exudates (3) > Hemorrhages (2) > Microaneurysms (1) > Background (0)
CONFLICT_PRIORITY_ORDER = [1, 2, 3, 4]


def locate_idrid_dataset(search_dirs: Optional[List[str]] = None) -> Optional[str]:
    """Search for IDRiD segmentation dataset in standard project locations."""
    if search_dirs is None:
        search_dirs = [
            DEFAULT_IDRID_PATH,
            os.path.join(os.path.expanduser("~"), "Downloads", "IDRiD_segmentation"),
            os.path.join(os.path.expanduser("~"), "Downloads", "idrid_segmentation"),
            os.path.join(os.path.expanduser("~"), "Downloads", "A. Segmentation"),
            os.path.join(os.path.expanduser("~"), "Downloads", "dataset", "segmentation"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DR_Screening_Pipeline", "DR_Screening_Pipeline", "data", "idrid")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "IDRiD")),
        ]

    for candidate in search_dirs:
        if candidate and os.path.isdir(candidate):
            # Check for images and groundtruths subfolders or matching files
            has_images = any(
                os.path.isdir(os.path.join(candidate, sub))
                for sub in ["1. Original Images", "Original Images", "images", "Images"]
            )
            has_gt = any(
                os.path.isdir(os.path.join(candidate, sub))
                for sub in ["2. Groundtruths", "Groundtruths", "groundtruth", "masks"]
            )
            if has_images or has_gt or len(glob.glob(os.path.join(candidate, "**", "*.tif"), recursive=True)) > 0:
                return candidate
    return None


def audit_idrid_dataset(data_dir: str) -> Dict[str, Any]:
    """
    Performs rigorous dataset integrity audit before any training begins.
    If the dataset is missing or incomplete, clearly reports DATASET NOT FOUND.
    """
    print("=" * 70)
    print(" DATASET AUDIT: IDRiD Retinal Lesion Segmentation")
    print(f" Checking path: {data_dir}")
    print("=" * 70)

    if not os.path.exists(data_dir):
        msg = (
            f"\n[!] DATASET NOT FOUND: Directory '{data_dir}' does not exist.\n\n"
            "To train the U-Net segmentation model with real Indian Diabetic Retinopathy data:\n"
            "1. Download the official IDRiD segmentation dataset from:\n"
            "   https://idrid.grand-challenge.org/Data/\n"
            "2. Extract 'A. Segmentation' to one of the following locations:\n"
            f"   - {DEFAULT_IDRID_PATH}\n"
            f"   - {os.path.join(os.path.expanduser('~'), 'Downloads', 'IDRiD_segmentation')}\n"
            "3. Expected subfolders:\n"
            "   - '1. Original Images/a. Training Set/'\n"
            "   - '2. Groundtruths/a. Training Set/' with (Microaneurysms, Haemorrhages, Hard Exudates, Soft Exudates)\n"
        )
        print(msg)
        return {"status": "NOT_FOUND", "error": msg, "data_dir": data_dir}

    # Discover image files
    img_patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif"]
    all_images = []
    for pat in img_patterns:
        all_images.extend(glob.glob(os.path.join(data_dir, "**", pat), recursive=True))

    # Separate original fundus images from annotation tif masks
    fundus_images = [
        f for f in all_images
        if not any(k in f.lower() for k in ["_ma.", "_he.", "_ex.", "_se.", "groundtruth", "mask", "microaneurysm"])
    ]

    all_masks = [
        f for f in all_images
        if any(k in f.lower() for k in ["_ma.", "_he.", "_ex.", "_se.", "groundtruth", "mask", "microaneurysm", "haemorrhage", "exudate"])
    ]

    print(f"Total fundus photos found : {len(fundus_images)}")
    print(f"Total lesion masks found  : {len(all_masks)}")

    if len(fundus_images) == 0:
        msg = (
            f"\n[!] DATASET NOT FOUND: No valid retinal fundus images found inside '{data_dir}'.\n"
            "Please ensure original IDRiD fundus images are placed in the directory.\n"
        )
        print(msg)
        return {"status": "NOT_FOUND", "error": msg, "data_dir": data_dir}

    return {
        "status": "FOUND",
        "total_images": len(fundus_images),
        "total_masks": len(all_masks),
        "fundus_images": fundus_images,
        "mask_files": all_masks
    }


def find_lesion_masks_for_image(image_path: str, data_dir: str) -> Dict[str, Optional[str]]:
    """
    Finds corresponding lesion masks (MA, HE, EX, SE) for a given fundus image file.
    IDRiD naming pattern:
      Image: IDRiD_01.jpg
      MA: IDRiD_01_MA.tif
      HE: IDRiD_01_HE.tif
      EX: IDRiD_01_EX.tif
      SE: IDRiD_01_SE.tif
    """
    base = os.path.splitext(os.path.basename(image_path))[0]
    masks = {
        "ma": None,  # Microaneurysms
        "he": None,  # Hemorrhages
        "ex": None,  # Hard Exudates
        "se": None   # Soft Exudates
    }

    # Search in common groundtruth locations
    search_dirs = [
        os.path.dirname(image_path),
        os.path.join(data_dir, "2. Groundtruths"),
        os.path.join(data_dir, "Groundtruths"),
        os.path.join(data_dir, "masks"),
        data_dir
    ]

    for s_dir in search_dirs:
        if not os.path.isdir(s_dir):
            continue
        for root, _, files in os.walk(s_dir):
            for f in files:
                f_lower = f.lower()
                if not f_lower.endswith((".tif", ".png", ".jpg")):
                    continue
                if base.lower() in f_lower:
                    if any(k in f_lower for k in ["_ma", "microaneurysm"]):
                        masks["ma"] = os.path.join(root, f)
                    elif any(k in f_lower for k in ["_he", "haemorrhage", "hemorrhage"]):
                        masks["he"] = os.path.join(root, f)
                    elif any(k in f_lower for k in ["_ex", "hard exudate", "hard_exudate"]):
                        masks["ex"] = os.path.join(root, f)
                    elif any(k in f_lower for k in ["_se", "soft exudate", "soft_exudate", "cotton"]):
                        masks["se"] = os.path.join(root, f)

    return masks


def combine_lesion_masks(
    mask_paths: Dict[str, Optional[str]],
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Combines individual lesion masks into a single multi-class integer mask:
        0 = Background
        1 = Microaneurysm (MA)
        2 = Hemorrhage (HE)
        3 = Hard Exudate (EX)
        4 = Soft Exudate (SE)

    Conflict resolution:
        If multiple lesions overlap on the same pixel, we apply deterministic priority:
        SE (4) > EX (3) > HE (2) > MA (1) > BG (0).
        This guarantees reproducible, consistent ground truth without silent data corruption.
    """
    combined = np.zeros(target_size, dtype=np.int64)

    lesion_mapping = [
        ("ma", 1),
        ("he", 2),
        ("ex", 3),
        ("se", 4)
    ]

    for key, class_id in lesion_mapping:
        path = mask_paths.get(key)
        if path and os.path.exists(path):
            mask_img = Image.open(path)
            # Crucial: NEAREST NEIGHBOR interpolation for mask resizing to prevent class index corruption
            mask_resized = mask_img.resize((target_size[1], target_size[0]), Image.NEAREST)
            mask_arr = np.array(mask_resized)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            # Binarize: any positive pixel in lesion mask is foreground
            binary_mask = (mask_arr > 0)
            combined[binary_mask] = class_id

    return combined


class IDRiDLesionDataset(Dataset):
    """
    PyTorch Dataset for multi-class retinal lesion segmentation.
    Handles image-mask pairing, synchronized geometric augmentation,
    bilinear image resizing, nearest-neighbor mask resizing, and normalization.
    """

    def __init__(
        self,
        image_paths: List[str],
        data_dir: str,
        img_size: int = 512,
        is_train: bool = True,
        normalize_mean: Optional[List[float]] = None,
        normalize_std: Optional[List[float]] = None
    ):
        self.image_paths = image_paths
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train
        self.mean = normalize_mean or IMAGE_NET_MEAN
        self.std = normalize_std or IMAGE_NET_STD

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        # Find and combine lesion masks
        mask_paths = find_lesion_masks_for_image(img_path, self.data_dir)
        orig_w, orig_h = image.size
        # Resize image: Bilinear
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Build multi-class combined mask at target resolution (using Nearest-Neighbor internally)
        combined_mask = combine_lesion_masks(mask_paths, (self.img_size, self.img_size))
        mask_pil = Image.fromarray(combined_mask.astype(np.uint8))

        # Synchronized Data Augmentation for training split
        if self.is_train:
            # 1. Random Horizontal Flip (synchronized on image and mask)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask_pil = TF.hflip(mask_pil)

            # 2. Random Vertical Flip (synchronized on image and mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask_pil = TF.vflip(mask_pil)

            # 3. Small Random Rotation (-15 to +15 deg, nearest neighbor for mask)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask_pil = TF.rotate(mask_pil, angle, interpolation=TF.InterpolationMode.NEAREST)

            # 4. Color transformations on IMAGE ONLY (never corrupt mask IDs)
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.85, 1.15)
                contrast_factor = random.uniform(0.85, 1.15)
                image = TF.adjust_brightness(image, brightness_factor)
                image = TF.adjust_contrast(image, contrast_factor)

        # Convert image to Float Tensor [3, H, W] in [0, 1] and normalize
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)

        # Convert mask to Long Tensor [H, W] with integer class labels {0, 1, 2, 3, 4}
        mask_tensor = torch.from_numpy(np.array(mask_pil, dtype=np.int64))

        return image_tensor, mask_tensor


def create_idrid_splits(
    image_paths: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    save_path: Optional[str] = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Creates strict 70/15/15 train/val/test splits with zero data leakage.
    Saves split metadata to JSON for reproducibility.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Splits must sum to 1.0"
    rng = random.Random(seed)
    shuffled = sorted(image_paths)  # deterministic pre-sort
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train:n_train + n_val]
    test_paths = shuffled[n_train + n_val:]

    # Verify zero data leakage
    set_train = set(train_paths)
    set_val = set(val_paths)
    set_test = set(test_paths)
    assert len(set_train.intersection(set_val)) == 0, "Data leakage between Train and Val!"
    assert len(set_train.intersection(set_test)) == 0, "Data leakage between Train and Test!"
    assert len(set_val.intersection(set_test)) == 0, "Data leakage between Val and Test!"

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        split_data = {
            "seed": seed,
            "total_images": n,
            "train_count": len(train_paths),
            "val_count": len(val_paths),
            "test_count": len(test_paths),
            "train_images": [os.path.basename(p) for p in train_paths],
            "val_images": [os.path.basename(p) for p in val_paths],
            "test_images": [os.path.basename(p) for p in test_paths]
        }
        with open(save_path, "w") as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved dataset split metadata to: {save_path}")

    return train_paths, val_paths, test_paths


def calculate_training_class_weights(dataset: Dataset, num_classes: int = 5) -> torch.Tensor:
    """
    Calculates class weights ONLY from the training set.
    Prevents background pixel domination over tiny retinal lesions (e.g. microaneurysms).
    """
    pixel_counts = torch.zeros(num_classes, dtype=torch.float64)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    for _, masks in loader:
        for c in range(num_classes):
            pixel_counts[c] += torch.sum(masks == c).item()

    total_pixels = pixel_counts.sum()
    print("Training set pixel counts per class:")
    class_names = ["Background", "Microaneurysm", "Hemorrhage", "Hard Exudate", "Soft Exudate"]
    for c in range(num_classes):
        pct = (pixel_counts[c] / max(total_pixels, 1.0)) * 100
        print(f"  Class {c} ({class_names[c]:14s}): {int(pixel_counts[c]):10d} pixels ({pct:.3f}%)")

    # Inverse frequency weights, normalized to mean = 1.0
    weights = total_pixels / (num_classes * (pixel_counts + 1e-6))
    weights = weights / weights.mean()
    # Clip extreme weight ratios to prevent training instability
    weights = torch.clamp(weights, min=0.1, max=50.0).float()
    return weights
