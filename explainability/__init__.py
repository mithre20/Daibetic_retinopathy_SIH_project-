"""
Explainability Module for AI DR Screen
Exposes SHAP feature-level explainability, visualizers, summaries, and Grad-CAM utilities.
"""

from .shap_explainer import (
    DRModelWrapper,
    find_model_checkpoint,
    compute_shap_maps,
    generate_shap_explanation
)
from .shap_visualizer import (
    pil_to_base64,
    create_positive_map,
    create_negative_map,
    create_combined_map,
    create_overlay
)
from .shap_summary import (
    MEDICAL_SAFETY_DISCLAIMER,
    compute_lesion_aware_shap_stats,
    compute_shap_unet_overlap,
    generate_human_readable_summary
)

__all__ = [
    "DRModelWrapper",
    "find_model_checkpoint",
    "compute_shap_maps",
    "generate_shap_explanation",
    "pil_to_base64",
    "create_positive_map",
    "create_negative_map",
    "create_combined_map",
    "create_overlay",
    "MEDICAL_SAFETY_DISCLAIMER",
    "compute_lesion_aware_shap_stats",
    "compute_shap_unet_overlap",
    "generate_human_readable_summary"
]
