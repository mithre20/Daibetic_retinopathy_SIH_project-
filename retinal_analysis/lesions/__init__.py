"""
Retinal Lesion Analysis Sub-Package
Dedicated modules for Microaneurysms, Hemorrhages, Hard Exudates, and Soft Exudates.
"""

from retinal_analysis.lesions.microaneurysm import detect_microaneurysms, MicroaneurysmDetector
from retinal_analysis.lesions.hemorrhage import analyze_hemorrhages, HemorrhageAnalyzer
from retinal_analysis.lesions.exudate import segment_exudates, ExudateSegmenter
from retinal_analysis.lesions.evaluation import evaluate_lesions

__all__ = [
    "detect_microaneurysms",
    "MicroaneurysmDetector",
    "analyze_hemorrhages",
    "HemorrhageAnalyzer",
    "segment_exudates",
    "ExudateSegmenter",
    "evaluate_lesions"
]
