"""
Retinal Image Preprocessing Utilities for AI DR Screen
Handles FOV extraction, green-channel isolation, CLAHE, and illumination normalization.
"""

import os
import io
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image
import cv2


def load_fundus_image(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """Loads fundus image into RGB uint8 numpy array [H, W, 3]."""
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"Image not found at path: {image_input}")
        img_pil = Image.open(image_input).convert("RGB")
        return np.array(img_pil)
    elif isinstance(image_input, bytes):
        img_pil = Image.open(io.BytesIO(image_input)).convert("RGB")
        return np.array(img_pil)
    elif isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        return image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def extract_retinal_fov_mask(image_rgb: np.ndarray, threshold: int = 15) -> np.ndarray:
    """
    Computes a binary Field-of-View (FOV) mask isolating the circular retina from black borders.
    Returns uint8 mask where 255 = retinal tissue, 0 = background border.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Fill small holes and smooth boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest connected component (the circular retina)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_idx).astype(np.uint8) * 255
    return mask


def extract_green_channel(image_rgb: np.ndarray) -> np.ndarray:
    """
    Extracts the green channel (Channel index 1 in RGB).
    The green channel exhibits maximal absorption contrast for hemoglobin.
    """
    return image_rgb[:, :, 1]


def apply_clahe(
    image_gray: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization to single-channel image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_gray)


def illumination_equalization(image_gray: np.ndarray, fov_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Equalizes non-uniform illumination by subtracting a heavily blurred background estimate.
    """
    # Large Gaussian kernel (approx 10% of image width)
    ksize = int(max(image_gray.shape) * 0.08)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 31)
    
    background = cv2.GaussianBlur(image_gray, (ksize, ksize), 0)
    normalized = cv2.subtract(image_gray, background)
    equalized = cv2.add(normalized, 128)
    
    if fov_mask is not None:
        equalized = cv2.bitwise_and(equalized, equalized, mask=fov_mask)
    return equalized


def preprocess_fundus(
    image_input: Union[str, bytes, Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (512, 512)
) -> dict:
    """
    Unified standard preprocessing routine for all retinal analysis modules.
    
    Returns:
        dict containing:
            - 'original_rgb': Original RGB numpy array [H, W, 3]
            - 'resized_rgb': Resized RGB numpy array [target_H, target_W, 3]
            - 'green_channel': Resized green channel [target_H, target_W]
            - 'clahe_green': Contrast-enhanced green channel [target_H, target_W]
            - 'fov_mask': Binary FOV mask [target_H, target_W]
            - 'scale_x': Ratio original_W / target_W
            - 'scale_y': Ratio original_H / target_H
    """
    original_rgb = load_fundus_image(image_input)
    orig_h, orig_w = original_rgb.shape[:2]
    
    resized_rgb = cv2.resize(original_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    fov_mask = extract_retinal_fov_mask(resized_rgb)
    
    green = extract_green_channel(resized_rgb)
    clahe_green = apply_clahe(green)
    
    return {
        "original_rgb": original_rgb,
        "resized_rgb": resized_rgb,
        "green_channel": green,
        "clahe_green": clahe_green,
        "fov_mask": fov_mask,
        "orig_shape": (orig_h, orig_w),
        "target_size": target_size,
        "scale_x": orig_w / float(target_size[0]),
        "scale_y": orig_h / float(target_size[1])
    }
