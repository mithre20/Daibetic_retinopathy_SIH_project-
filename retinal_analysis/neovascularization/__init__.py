"""
Neovascularization Detection Module
Status: PENDING DATASET + CLINICAL VALIDATION
"""

from retinal_analysis.neovascularization.model import NeovascularizationDetector
from retinal_analysis.neovascularization.inference import check_neovascularization

__all__ = ["NeovascularizationDetector", "check_neovascularization"]
