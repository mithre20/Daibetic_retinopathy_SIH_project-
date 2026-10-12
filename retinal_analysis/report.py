"""
Explainable Retinal Evidence Report Generator
AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
"""

from typing import Dict, Any, Optional

REPORT_DISCLAIMER = (
    "Important: These findings are AI-generated screening evidence and are not a definitive diagnosis. "
    "Clinical review by a certified ophthalmologist or eye-care professional is strictly required."
)


def generate_retinal_evidence_report(
    dr_result: Optional[Dict[str, Any]],
    retinal_evidence: Dict[str, Any],
    gradcam_available: bool = True,
    shap_available: bool = True
) -> str:
    """
    Assembles formatted, honest explainable evidence report according to clinical screening standards.
    """
    lines = []
    lines.append("==================================================")
    lines.append("        AI DR SCREEN — RETINAL ANALYSIS")
    lines.append("==================================================")
    lines.append("")
    
    # 1. DR Classification
    lines.append("DR Classification")
    lines.append("------------------")
    if dr_result:
        severity = dr_result.get("severity", "Unknown")
        conf = dr_result.get("confidence", 0.0) * 100.0
        lines.append(f"Grade:      {severity}")
        lines.append(f"Confidence: {conf:.1f}%")
        if "referable" in dr_result:
            lines.append(f"Referable:  {'Yes' if dr_result['referable'] else 'No'}")
    else:
        lines.append("Grade:      Blocked or Not Evaluated (IQA Gated)")
    lines.append("")
    
    # 2. Retinal Structures
    lines.append("Retinal Structures")
    lines.append("------------------")
    structs = retinal_evidence.get("retinal_structures", {})
    od = structs.get("optic_disc", {})
    fovea = structs.get("fovea", {})
    vessels = structs.get("vessels", {})
    
    od_txt = "Detected" if od.get("detected") else "Not detected"
    fovea_txt = "Detected" if fovea.get("detected") else "Not detected"
    vessel_txt = f"Available ({vessels.get('vessel_density_pct', 0.0):.1f}% density)" if vessels.get("available") else "Unavailable"
    
    lines.append(f"Optic Disc:          {od_txt}")
    lines.append(f"Fovea:               {fovea_txt}")
    lines.append(f"Vessel Segmentation: {vessel_txt}")
    lines.append("")
    
    # 3. Lesion Evidence
    lines.append("Lesion Evidence")
    lines.append("----------------")
    lesions = retinal_evidence.get("lesions", {})
    ma = lesions.get("microaneurysms", {})
    he = lesions.get("hemorrhages", {})
    hard_ex = lesions.get("hard_exudates", {})
    soft_ex = lesions.get("soft_exudates", {})
    nv = retinal_evidence.get("neovascularization", {})
    
    ma_txt = f"Detected ({ma.get('count', 0)} candidates)" if ma.get("detected") else "Not detected"
    he_txt = f"Detected ({he.get('count', 0)} regions, {he.get('area_percentage', 0.0):.2f}% area)" if he.get("detected") else "Not detected"
    hard_txt = f"Detected ({hard_ex.get('area_percentage', 0.0):.2f}% area)" if hard_ex.get("detected") else "Not detected"
    soft_txt = f"Detected ({soft_ex.get('area_percentage', 0.0):.2f}% area)" if soft_ex.get("detected") else "Not detected"
    
    if nv.get("detected"):
        nv_txt = "Detected"
    elif "PENDING" in str(nv.get("status", "")):
        nv_txt = "Not evaluated (Pending Dataset)"
    else:
        nv_txt = "Not detected"
        
    lines.append(f"Microaneurysms:     {ma_txt}")
    lines.append(f"Hemorrhages:        {he_txt}")
    lines.append(f"Hard Exudates:      {hard_txt}")
    lines.append(f"Soft Exudates:      {soft_txt}")
    lines.append(f"Neovascularization: {nv_txt}")
    lines.append("")
    
    # 4. Explainability
    lines.append("Explainability")
    lines.append("----------------")
    lines.append(f"Grad-CAM: {'Available' if gradcam_available else 'Unavailable'}")
    lines.append(f"SHAP:     {'Available' if shap_available else 'Unavailable'}")
    lines.append("")
    
    # 5. Disclaimer
    lines.append("Important:")
    lines.append(REPORT_DISCLAIMER)
    lines.append("==================================================")
    
    return "\n".join(lines)
