import os
import glob
import pandas as pd
from typing import Tuple, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split

DEFAULT_DATASET_PATH = os.environ.get(
    "DATASET_PATH",
    os.path.join(os.path.dirname(__file__), "..", "DR_Screening_Pipeline", "DR_Screening_Pipeline", "data")
)

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """Data augmentation for training images."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Standard evaluation transforms for validation and test sets."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

class RetinalDRDataset(Dataset):
    """
    PyTorch Dataset for Retinal Fundus Images.
    Accepts:
    - image_paths: List[str]
    - labels: List[int]
    - transform: torchvision.transforms
    """
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error reading image file {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        label = int(self.labels[idx])
        return image, label

def load_dataset_samples(data_dir: str) -> Tuple[List[str], List[int]]:
    """
    Auto-detects and loads fundus samples from either:
    1. Folder-based layout (e.g. data_dir/0, data_dir/1 ... or data_dir/train/0, data_dir/train/1)
    2. CSV-based layout (e.g. labels.csv or train.csv with image_path/diagnosis or image/grade)
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path does not exist: {data_dir}")

    image_paths = []
    labels = []

    # Check 1: CSV file in data_dir (e.g. labels.csv, train.csv)
    csv_candidates = [
        os.path.join(data_dir, "labels.csv"),
        os.path.join(data_dir, "train", "labels.csv"),
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "annotations.csv")
    ]
    csv_file = None
    for cand in csv_candidates:
        if os.path.exists(cand):
            csv_file = cand
            break

    if csv_file is not None:
        print(f"Loading dataset using metadata CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        # Normalize column names
        col_map = {col.lower().strip(): col for col in df.columns}
        
        # Identify path column
        path_col = None
        for name in ["image_path", "image", "filename", "filepath", "id_code"]:
            if name in col_map:
                path_col = col_map[name]
                break

        # Identify label column
        label_col = None
        for name in ["diagnosis", "grade", "label", "dr_grade", "target"]:
            if name in col_map:
                label_col = col_map[name]
                break

        if path_col is None or label_col is None:
            raise ValueError(f"Could not identify image and label columns in {csv_file}. Columns found: {list(df.columns)}")

        base_search_dirs = [os.path.dirname(csv_file), data_dir, os.path.join(data_dir, "train"), os.path.join(data_dir, "images")]

        for _, row in df.iterrows():
            fname = str(row[path_col]).strip()
            # If no extension, try adding .png or .jpg
            cand_names = [fname]
            if not any(fname.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                cand_names.extend([f"{fname}.png", f"{fname}.jpg", f"{fname}.jpeg"])

            found_path = None
            for b_dir in base_search_dirs:
                for c_name in cand_names:
                    p = os.path.join(b_dir, c_name)
                    if os.path.isfile(p):
                        found_path = p
                        break
                if found_path:
                    break

            if found_path:
                lbl = int(row[label_col])
                if 0 <= lbl <= 4:
                    image_paths.append(found_path)
                    labels.append(lbl)

    # Check 2: Folder-based layout (subfolders named '0', '1', '2', '3', '4' or '0_NoDR', etc.)
    if len(image_paths) == 0:
        print(f"Scanning for folder-based class structure in: {data_dir}")
        class_dirs = {}
        # Search directly or in train/
        search_roots = [data_dir, os.path.join(data_dir, "train")]
        for s_root in search_roots:
            if not os.path.exists(s_root):
                continue
            for entry in os.listdir(s_root):
                full_p = os.path.join(s_root, entry)
                if os.path.isdir(full_p):
                    # Check if entry maps to 0-4
                    digit = None
                    if entry in ["0", "1", "2", "3", "4"]:
                        digit = int(entry)
                    elif entry.startswith(("0_", "1_", "2_", "3_", "4_")):
                        digit = int(entry[0])
                    elif "no_dr" in entry.lower():
                        digit = 0
                    elif "mild" in entry.lower():
                        digit = 1
                    elif "moderate" in entry.lower():
                        digit = 2
                    elif "severe" in entry.lower():
                        digit = 3
                    elif "proliferat" in entry.lower():
                        digit = 4

                    if digit is not None and digit not in class_dirs:
                        class_dirs[digit] = full_p

        if len(class_dirs) > 0:
            for cls_idx, folder in sorted(class_dirs.items()):
                imgs = []
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                    imgs.extend(glob.glob(os.path.join(folder, ext)))
                    imgs.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
                for p in imgs:
                    image_paths.append(p)
                    labels.append(cls_idx)

    if len(image_paths) == 0:
        raise ValueError(f"No valid fundus images found in {data_dir}. Please ensure either folder-per-class or CSV layout exists.")

    print(f"Loaded {len(image_paths)} total samples across {len(set(labels))} classes from {data_dir}")
    return image_paths, labels

def create_data_loaders(
    data_dir: str = DEFAULT_DATASET_PATH,
    batch_size: int = 16,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 0,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, List[float]]:
    """
    Creates stratified Train, Validation, and Test DataLoaders.
    Guarantees:
    - Zero data leakage (test set completely held out)
    - Stratified splits maintaining class proportions
    - Returns class weights for handling class imbalance
    """
    image_paths, labels = load_dataset_samples(data_dir)

    # 1. Stratified split into (train + val) and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=test_split,
        stratify=labels,
        random_state=random_seed
    )

    # 2. Stratified split into train and val
    val_rel_size = val_split / (1.0 - test_split)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_rel_size,
        stratify=train_val_labels,
        random_state=random_seed
    )

    print(f"Split sizes -> Train: {len(train_paths)}, Validation: {len(val_paths)}, Test: {len(test_paths)}")

    # Compute class weights for imbalanced cross-entropy loss: N / (num_classes * count)
    class_counts = pd.Series(train_labels).value_counts().to_dict()
    total_train = len(train_labels)
    num_classes = 5
    class_weights = []
    for c in range(num_classes):
        count = class_counts.get(c, 1)
        w = total_train / (num_classes * count)
        class_weights.append(w)

    train_ds = RetinalDRDataset(train_paths, train_labels, transform=get_train_transforms())
    val_ds = RetinalDRDataset(val_paths, val_labels, transform=get_val_transforms())
    test_ds = RetinalDRDataset(test_paths, test_labels, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, class_weights
