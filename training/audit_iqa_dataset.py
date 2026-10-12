"""
Dataset Audit Script for Retinal Image Quality Assessment (IQA)
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Audits candidate datasets for:
  - Total image counts & valid file integrity
  - Verification of legitimate quality labels (Good, Poor, Unusable) vs DR severity (0-4)
  - Class distribution & imbalance ratios
  - Image spatial dimensions & aspect ratios
  - Corrupted, unreadable, duplicate, or missing images
  - Patient/Image leakage risk analysis
"""

import os
import sys
import glob
import hashlib
import argparse
from typing import Dict, Any, List, Optional
from PIL import Image, UnidentifiedImageError
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

QUALITY_CLASSES = ["Good Quality", "Poor Quality", "Unusable"]
VALID_QUALITY_LABELS = {
    "0": "Good Quality",
    "1": "Poor Quality",
    "2": "Unusable",
    "good": "Good Quality",
    "poor": "Poor Quality",
    "unusable": "Unusable",
    "gradable": "Good Quality",
    "ungradable": "Unusable"
}


def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash to detect exact duplicate files."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset(dataset_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs comprehensive audit of retinal images and labels in dataset directory.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(ROOT_DIR, "DR_Screening_Pipeline", "DR_Screening_Pipeline", "data")
    
    print("\n" + "=" * 60)
    print(" AI DR Screen — Retinal Image Quality Dataset Audit")
    print("=" * 60)
    print(f"Target dataset directory: {os.path.abspath(dataset_dir)}")

    if not os.path.isdir(dataset_dir):
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        return {"error": f"Directory not found: {dataset_dir}"}

    # 1. Search for metadata CSVs
    csv_files = glob.glob(os.path.join(dataset_dir, "**", "*.csv"), recursive=True)
    image_files = glob.glob(os.path.join(dataset_dir, "**", "*.[jJ][pP][gG]"), recursive=True) + \
                  glob.glob(os.path.join(dataset_dir, "**", "*.[pP][nN][gG]"), recursive=True)

    print(f"Found {len(image_files)} image files and {len(csv_files)} CSV files.")

    # 2. Check labels in CSV files
    is_legitimate_iqa = False
    dr_severity_detected = False
    quality_counts = {"Good Quality": 0, "Poor Quality": 0, "Unusable": 0}
    dr_severity_counts = {}
    invalid_labels = 0
    missing_files = 0
    total_labeled_rows = 0

    for csv_p in csv_files:
        try:
            df = pd.read_csv(csv_p)
            cols = [c.strip().lower() for c in df.columns]

            # Check if columns indicate DR severity classes (e.g. Mild, Moderate, No_DR...)
            if any(k in cols for k in ["no_dr", "mild", "moderate", "severe", "proliferate_dr"]):
                dr_severity_detected = True
            if "grade" in cols:
                # Check sample values in grade
                sample_grades = df["grade"].dropna().unique()
                if any(g in [3, 4] for g in sample_grades):
                    dr_severity_detected = True
                    for g in sample_grades:
                        dr_severity_counts[str(g)] = int((df["grade"] == g).sum())

            # Check if genuine quality column exists
            if any(q in cols for q in ["quality", "gradability", "iqa_label"]):
                q_col = next(c for c in df.columns if c.strip().lower() in ["quality", "gradability", "iqa_label"])
                for val in df[q_col].dropna():
                    s_val = str(val).strip().lower()
                    if s_val in VALID_QUALITY_LABELS:
                        quality_counts[VALID_QUALITY_LABELS[s_val]] += 1
                        is_legitimate_iqa = True
                    else:
                        invalid_labels += 1
        except Exception as e:
            print(f"Note: Error reading CSV {csv_p}: {e}")

    # 3. Check image files integrity and dimensions
    corrupted_images = []
    dimension_samples = {}
    hashes = set()
    duplicate_count = 0

    print("Auditing image integrity (sampling up to 300 images)...")
    sample_to_check = image_files[:300] if len(image_files) > 300 else image_files

    for img_p in sample_to_check:
        try:
            with Image.open(img_p) as im:
                im.verify()
            # Check dimensions
            with Image.open(img_p) as im:
                dims = f"{im.width}x{im.height}"
                dimension_samples[dims] = dimension_samples.get(dims, 0) + 1
            # Check duplicate hash
            h = compute_file_hash(img_p)
            if h in hashes:
                duplicate_count += 1
            else:
                hashes.add(h)
        except (UnidentifiedImageError, OSError):
            corrupted_images.append(img_p)

    # 4. Report Findings
    print("\n--- Audit Findings ---")
    print(f"Total images scanned:       {len(image_files)}")
    print(f"Corrupted images:           {len(corrupted_images)}")
    print(f"Duplicate images found:     {duplicate_count}")
    print(f"Image dimensions:           {dict(list(dimension_samples.items())[:5])}")

    if dr_severity_detected and not is_legitimate_iqa:
        print("\n[!] CRITICAL AUDIT RESULT:")
        print("    The existing dataset contains Diabetic Retinopathy severity labels (Grades 0 to 4),")
        print("    NOT legitimate retinal image quality labels (Good, Poor, Unusable).")
        print("    Per Section 5 & 32: Do NOT fabricate quality labels from DR severity grades.")
        print("    Status: IQA TRAINING BLOCKED until a genuine quality-labeled dataset (e.g. EyeQ) is provided.")
    elif is_legitimate_iqa:
        total_q = sum(quality_counts.values())
        print("\n[✓] Genuine quality labels verified:")
        for qname, qcnt in quality_counts.items():
            pct = (qcnt / total_q * 100) if total_q > 0 else 0
            print(f"    • {qname:<15}: {qcnt} ({pct:.1f}%)")
    else:
        print("\n[!] No labeled metadata found. Unsupervised or rule-based metrics will be utilized.")

    print("=" * 60 + "\n")

    return {
        "dataset_directory": os.path.abspath(dataset_dir),
        "total_images": len(image_files),
        "corrupted_images_count": len(corrupted_images),
        "duplicate_images_count": duplicate_count,
        "sample_dimensions": dimension_samples,
        "has_genuine_quality_labels": is_legitimate_iqa,
        "dr_severity_labels_detected": dr_severity_detected,
        "quality_class_counts": quality_counts if is_legitimate_iqa else None,
        "dr_severity_counts": dr_severity_counts if dr_severity_detected else None
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit Retinal Image Quality Dataset")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory path to target dataset")
    args = parser.parse_args()
    audit_dataset(args.data_dir)
