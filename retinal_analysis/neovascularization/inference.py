"""
Neovascularization Inference Pipeline
"""

from typing import Any, Dict, Optional
from retinal_analysis.neovascularization.model import NeovascularizationDetector

_DEFAULT_DETECTOR = None

def get_neovascularization_detector() -> NeovascularizationDetector:
    global _DEFAULT_DETECTOR
    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = NeovascularizationDetector()
    return _DEFAULT_DETECTOR


def check_neovascularization(
    image_input: Any,
    optic_disc_result: Optional[dict] = None,
    vessel_mask: Optional[Any] = None
) -> Dict[str, Any]:
    """Inference entrypoint reporting honest dataset requirement status."""
    detector = get_neovascularization_detector()
    return detector.detect(
        image_input,
        optic_disc_result=optic_disc_result,
        vessel_mask=vessel_mask
    )
