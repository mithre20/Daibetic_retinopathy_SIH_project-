"""
Visualization Module for SHAP Feature-Level Attributions
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
"""

import io
import base64
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


def get_colormap(name: str):
    """Safely retrieves Matplotlib colormap across library versions."""
    if hasattr(matplotlib, "colormaps"):
        return matplotlib.colormaps[name]
    return plt.get_cmap(name)


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    """Encodes PIL Image to Base64 Data URI."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"


def resize_heatmap_to_image(
    heatmap: np.ndarray,
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Resizes a 2D float heatmap (H, W) to (target_w, target_h) using bicubic interpolation.
    """
    target_w, target_h = target_size
    img = Image.fromarray(heatmap.astype(np.float32))
    resized = img.resize((target_w, target_h), resample=Image.BICUBIC)
    return np.array(resized)


def create_positive_map(
    positive_attr: np.ndarray,
    orig_size: Optional[Tuple[int, int]] = None,
    colormap_name: str = "YlOrRd"
) -> Image.Image:
    """
    Visualizes positive SHAP attributions (features pushing towards target class)
    using warm colors (yellow to dark red).
    """
    if orig_size is not None:
        pos_res = resize_heatmap_to_image(positive_attr, orig_size)
    else:
        pos_res = positive_attr.copy()

    pos_norm = np.clip(pos_res, 0.0, 1.0)
    cmap = get_colormap(colormap_name)
    rgba = cmap(pos_norm) # (H, W, 4) in [0, 1]
    rgb_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb_uint8)


def create_negative_map(
    negative_attr: np.ndarray,
    orig_size: Optional[Tuple[int, int]] = None,
    colormap_name: str = "Blues"
) -> Image.Image:
    """
    Visualizes negative SHAP attributions (features pushing away from target class)
    using cool colors (light blue to deep dark blue).
    """
    if orig_size is not None:
        neg_res = resize_heatmap_to_image(negative_attr, orig_size)
    else:
        neg_res = negative_attr.copy()

    neg_norm = np.clip(neg_res, 0.0, 1.0)
    cmap = get_colormap(colormap_name)
    rgba = cmap(neg_norm)
    rgb_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb_uint8)


def create_combined_map(
    combined_attr: np.ndarray,
    orig_size: Optional[Tuple[int, int]] = None,
    colormap_name: str = "coolwarm"
) -> Image.Image:
    """
    Visualizes signed SHAP attributions [-1, 1] using a diverging colormap:
      Cool (Blue) = Negative contribution
      White/Neutral = Near zero contribution
      Warm (Red) = Positive contribution
    """
    if orig_size is not None:
        comb_res = resize_heatmap_to_image(combined_attr, orig_size)
    else:
        comb_res = combined_attr.copy()

    comb_clipped = np.clip(comb_res, -1.0, 1.0)
    # Map [-1, 1] to [0, 1] for colormap indexing
    norm_01 = (comb_clipped + 1.0) / 2.0
    cmap = get_colormap(colormap_name)
    rgba = cmap(norm_01)
    rgb_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb_uint8)


def create_overlay(
    original_img: Union[str, Image.Image],
    combined_attr: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "coolwarm",
    show_colorbar: bool = True
) -> Image.Image:
    """
    Creates an overlay of the signed SHAP diverging map on top of the original fundus scan.
    Low-attribution regions (|attr| < 0.08) are masked to preserve original retinal details.
    Includes a scientific diverging legend.
    """
    if isinstance(original_img, str):
        orig_pil = Image.open(original_img).convert("RGB")
    else:
        orig_pil = original_img.convert("RGB")

    orig_w, orig_h = orig_pil.size
    orig_np = np.array(orig_pil, dtype=np.float32) / 255.0

    # Resize attribution map to original dimensions
    comb_res = resize_heatmap_to_image(combined_attr, (orig_w, orig_h))
    comb_clipped = np.clip(comb_res, -1.0, 1.0)

    # Colorize with diverging colormap
    cmap = get_colormap(colormap_name)
    norm_01 = (comb_clipped + 1.0) / 2.0
    colored_heat = cmap(norm_01)[:, :, :3] # (H, W, 3)

    # Alpha mask: attenuate overlay where attribution is near zero
    importance_weight = np.clip((np.abs(comb_clipped) - 0.05) / 0.95, 0.0, 1.0)
    effective_alpha = alpha * importance_weight[:, :, np.newaxis]

    # Alpha blend
    blended = (1.0 - effective_alpha) * orig_np + effective_alpha * colored_heat
    blended_uint8 = (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)
    blended_img = Image.fromarray(blended_uint8)

    if not show_colorbar:
        return blended_img

    # Render figure with professional scientific legend bar
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")
    ax.imshow(blended_uint8)
    ax.axis("off")

    # Diverging colorbar with labeled ticks
    norm = Normalize(vmin=-1.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.03,
        aspect=30
    )
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    cbar.set_ticklabels([
        "Negative (−)",
        "Moderate (−)",
        "Neutral (0)",
        "Moderate (+)",
        "Positive (+)"
    ], fontsize=8, color="#cbd5e1")
    cbar.set_label("SHAP Feature Contribution", fontsize=9, color="#f8fafc", weight="bold", labelpad=6)
    cbar.ax.tick_params(labelsize=7.5, colors="#cbd5e1")
    cbar.outline.set_edgecolor("#334155")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")
