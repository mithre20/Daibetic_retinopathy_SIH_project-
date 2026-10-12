"""
Retinal Analysis Package for AI DR Screen
Explainable AI for Diabetic Retinopathy Screening in Rural India

Sub-packages:
- optic_disc: Optic Disc Localization & Segmentation
- fovea: Fovea & Macular Center Localization
- vessels: Retinal Vessel Segmentation
- lesions: Microaneurysms, Hemorrhages, Hard/Soft Exudates
- neovascularization: Neovascularization Detection (Pending Dataset)
- pipeline: Unified Retinal Evidence Pipeline
"""

from retinal_analysis.pipeline import UnifiedRetinalPipeline, analyze_retinal_structures

__all__ = [
    "UnifiedRetinalPipeline",
    "analyze_retinal_structures"
]
