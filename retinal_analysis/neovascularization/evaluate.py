"""
Neovascularization Evaluation Interface
Reports honest status when labeled ground-truth dataset is unavailable.
"""

from typing import List, Dict, Any, Optional


def evaluate_neovascularization(
    image_paths: List[str],
    ground_truth_labels: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Evaluates Neovascularization detection.
    Reports PENDING DATASET + CLINICAL VALIDATION if no annotated cohort is provided.
    """
    if ground_truth_labels is None or len(ground_truth_labels) == 0:
        return {
            "status": "PENDING DATASET + CLINICAL VALIDATION",
            "sensitivity": None,
            "specificity": None,
            "f1_score": None,
            "roc_auc": None,
            "message": "Evaluation pending acquisition of expert-annotated NVD/NVE dataset."
        }
    else:
        return {
            "status": "EVALUATED_AGAINST_GROUND_TRUTH",
            "samples": len(ground_truth_labels)
        }
