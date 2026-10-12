"""
Retinal Image Quality Assessment (IQA) Auxiliary Metrics
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India

Calculates objective classical image quality measurements:
  1. Sharpness: Variance of Laplacian (focus / motion blur)
  2. Illumination Uniformity: Grid tile mean spread & over/underexposure
  3. Field of View (FOV): Retinal aperture coverage ratio (Otsu binarization)
  4. Contrast: Root-Mean-Square (RMS) luminance standard deviation
  5. Exposure Extreme Fraction: Percentage of pixels at dark/saturated extremes
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Union, Tuple, List, Optional


def calculate_sharpness(img_gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Computes sharpness using the Variance of Laplacian.
    Higher values indicate sharp retinal vascular structures; low values indicate focus or motion blur.
    Returns (raw_variance, normalized_score [0, 1]).
    """
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    if mask is not None and np.sum(mask) > 100:
        raw_val = float(np.var(lap[mask > 0]))
    else:
        raw_val = float(np.var(lap))
    
    # Normalized score calibrated for retinal fundus imaging (200.0+ is generally sharp)
    norm_score = float(np.clip(raw_val / 250.0, 0.0, 1.0))
    return raw_val, norm_score


def calculate_fov_coverage(img_gray: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Estimates the percentage of the frame covered by the circular retinal aperture.
    Uses lenient Otsu thresholding to segment the illuminated retina from camera mask borders.
    Returns (coverage_ratio [0, 1], binary_mask).
    """
    h, w = img_gray.shape
    total_pixels = h * w
    
    # Threshold to isolate illuminated fundus from dark camera border
    _, thresh = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)
    # Morphological closing to seal internal vessel/fovea gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    if num_labels > 1:
        # Index 0 is background; find largest foreground component
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        retina_mask = (labels == largest_idx).astype(np.uint8) * 255
        coverage = float(stats[largest_idx, cv2.CC_STAT_AREA] / total_pixels)
    else:
        retina_mask = np.zeros_like(img_gray)
        coverage = 0.0

    return coverage, retina_mask


def calculate_illumination_and_exposure(
    img_gray: np.ndarray,
    mask: Optional[np.ndarray] = None,
    grid_size: int = 4
) -> Tuple[float, float, float]:
    """
    Evaluates illumination uniformity across a grid and checks for extreme under/over-exposure.
    Returns (uniformity_score [0, 1], mean_brightness, extreme_pixel_fraction).
    """
    h, w = img_gray.shape
    rh, rw = h // grid_size, w // grid_size
    tile_means = []

    for r in range(grid_size):
        for c in range(grid_size):
            tile = img_gray[r*rh : (r+1)*rh, c*rw : (c+1)*rw]
            if mask is not None:
                tile_mask = mask[r*rh : (r+1)*rh, c*rw : (c+1)*rw]
                if np.sum(tile_mask) > 50:
                    tile_means.append(float(np.mean(tile[tile_mask > 0])))
                    continue
            tile_means.append(float(np.mean(tile)))

    if len(tile_means) > 0 and np.mean(tile_means) > 0:
        spread = float(np.std(tile_means) / (np.mean(tile_means) + 1e-6))
        uniformity = float(np.clip(1.0 - (spread / 0.6), 0.0, 1.0))
    else:
        uniformity = 0.0

    # Mean brightness in retinal zone
    if mask is not None and np.sum(mask) > 100:
        retina_pixels = img_gray[mask > 0]
    else:
        retina_pixels = img_gray

    mean_b = float(np.mean(retina_pixels))
    dark_frac = float(np.mean(retina_pixels < 20))
    sat_frac = float(np.mean(retina_pixels > 245))
    extreme_frac = dark_frac + sat_frac

    return uniformity, mean_b, extreme_frac


def calculate_contrast(img_gray: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """
    Computes RMS contrast (standard deviation of pixel intensities).
    Returns (raw_std, normalized_contrast [0, 1]).
    """
    if mask is not None and np.sum(mask) > 100:
        pixels = img_gray[mask > 0]
    else:
        pixels = img_gray
    raw_std = float(np.std(pixels))
    norm_score = float(np.clip(raw_std / 55.0, 0.0, 1.0))
    return raw_std, norm_score


def compute_image_quality_metrics(
    image: Union[str, Image.Image, np.ndarray]
) -> Dict[str, Any]:
    """
    Computes comprehensive classical image quality indicators for a retinal fundus scan.
    Returns structured metrics and human-readable defect indicators.
    """
    if isinstance(image, str):
        pil_img = Image.open(image).convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
        img_np = np.array(pil_img)
    elif isinstance(image, np.ndarray):
        img_np = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Extract green channel (best contrast in ophthalmology) and grayscale
    if img_np.ndim == 3:
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_green = img_np[:, :, 1]
    else:
        img_gray = img_np
        img_green = img_np

    # 1. FOV Coverage
    fov_cov, retina_mask = calculate_fov_coverage(img_gray)
    fov_score = float(np.clip(fov_cov / 0.55, 0.0, 1.0)) # 55%+ of frame is typical for fundus

    # 2. Sharpness on green channel within retinal mask
    raw_sharp, sharp_score = calculate_sharpness(img_green, retina_mask)

    # 3. Illumination Uniformity & Exposure
    illum_score, mean_b, extreme_frac = calculate_illumination_and_exposure(img_green, retina_mask)

    # 4. RMS Contrast
    raw_contrast, contrast_score = calculate_contrast(img_green, retina_mask)

    # Defect indicators detection
    issues: List[str] = []
    if sharp_score < 0.30:
        issues.append("Image blur / out-of-focus capture")
    if fov_score < 0.45:
        issues.append("Insufficient retinal field coverage / partial eyelid occlusion")
    if mean_b < 35.0:
        issues.append("Severe underexposure / dark image")
    elif mean_b > 220.0:
        issues.append("Severe overexposure / bleached retina")
    if extreme_frac > 0.25:
        issues.append("Excessive saturation or dark clipping")
    if illum_score < 0.35:
        issues.append("Uneven illumination / camera vignetting / glare")
    if contrast_score < 0.25:
        issues.append("Low contrast / hazy media")

    # Composite rule-based quality score
    composite_score = float(
        0.35 * sharp_score +
        0.25 * fov_score +
        0.20 * illum_score +
        0.20 * contrast_score
    )

    return {
        "sharpness_score": round(sharp_score, 4),
        "sharpness_raw_var": round(raw_sharp, 2),
        "illumination_score": round(illum_score, 4),
        "mean_brightness": round(mean_b, 2),
        "fov_score": round(fov_score, 4),
        "fov_coverage_ratio": round(fov_cov, 4),
        "contrast_score": round(contrast_score, 4),
        "contrast_raw_std": round(raw_contrast, 2),
        "extreme_pixel_ratio": round(extreme_frac, 4),
        "composite_rule_score": round(composite_score, 4),
        "quality_issues_detected": issues,
        "is_acceptable_by_rules": bool(composite_score >= 0.50 and len(issues) <= 1)
    }
