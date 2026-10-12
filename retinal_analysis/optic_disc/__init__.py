"""
Optic Disc Localization & Segmentation Module
"""

from retinal_analysis.optic_disc.model import OpticDiscDetector
from retinal_analysis.optic_disc.inference import locate_optic_disc

__all__ = ["OpticDiscDetector", "locate_optic_disc"]
