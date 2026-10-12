"""
Unit Tests for Final Requirement Audit, Referable DR, and Grad-CAM Evidence
AI DR Screen / Retina_Rovers: Explainable AI for Diabetic Retinopathy Screening in Rural India
"""

import os
import json
import csv
import unittest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestFinalRequirementAudit(unittest.TestCase):

    def test_dr_validation_artifacts(self):
        """Verify Part 1 Referable DR validation artifacts exist and adhere to standards."""
        dr_dir = os.path.join(ROOT_DIR, "results", "dr_validation")
        self.assertTrue(os.path.exists(dr_dir), f"Directory missing: {dr_dir}")

        # Metrics JSON
        json_p = os.path.join(dr_dir, "metrics.json")
        self.assertTrue(os.path.exists(json_p), f"metrics.json missing: {json_p}")
        with open(json_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metrics", data)
        m = data["metrics"]
        for key in ["sensitivity", "specificity", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "npv"]:
            self.assertIn(key, m, f"Missing metric {key}")

        self.assertIn("requirement", data)
        self.assertIn("status", data["requirement"])
        self.assertEqual(data["requirement"]["status"], "FAIL — TARGET NOT MET")

        # Visualizations & Report
        self.assertTrue(os.path.exists(os.path.join(dr_dir, "confusion_matrix.png")))
        self.assertTrue(os.path.exists(os.path.join(dr_dir, "roc_curve.png")))
        self.assertTrue(os.path.exists(os.path.join(dr_dir, "precision_recall_curve.png")))
        self.assertTrue(os.path.exists(os.path.join(dr_dir, "metrics.csv")))
        self.assertTrue(os.path.exists(os.path.join(dr_dir, "report.md")))

    def test_gradcam_lesion_overlap(self):
        """Verify Part 9 Grad-CAM & Lesion Overlap artifact and scientific statement."""
        overlap_p = os.path.join(ROOT_DIR, "results", "retinal_analysis", "gradcam_lesion_overlap.json")
        self.assertTrue(os.path.exists(overlap_p), f"gradcam_lesion_overlap.json missing: {overlap_p}")
        with open(overlap_p, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("scientific_statement", data)
        self.assertIn("highlights image regions that contributed", data["scientific_statement"])
        self.assertIn("prohibited_claim", data)
        self.assertIn("does NOT prove", data["prohibited_claim"])
        self.assertIn("evaluation_summary", data)
        self.assertIn("pointing_game_accuracy_pct", data["evaluation_summary"])

    def test_final_requirement_compliance_matrix(self):
        """Verify Part 22 22-row requirement audit table."""
        json_p = os.path.join(ROOT_DIR, "results", "final_requirement_audit.json")
        csv_p = os.path.join(ROOT_DIR, "results", "final_requirement_audit.csv")
        md_p = os.path.join(ROOT_DIR, "docs", "final_requirement_compliance.md")

        self.assertTrue(os.path.exists(json_p), f"Missing {json_p}")
        self.assertTrue(os.path.exists(csv_p), f"Missing {csv_p}")
        self.assertTrue(os.path.exists(md_p), f"Missing {md_p}")

        with open(json_p, "r", encoding="utf-8") as f:
            table = json.load(f)

        self.assertEqual(len(table), 22, f"Expected 22 requirements in audit table, got {len(table)}")

        expected_reqs = [
            "Referable DR sensitivity",
            "Referable DR specificity",
            "Optic disc localization",
            "Fovea localization",
            "Vessel segmentation",
            "Microaneurysm detection",
            "Exudate segmentation",
            "Hemorrhage analysis",
            "Neovascularization",
            "Sub-pixel MA",
            "Grad-CAM validation",
            "Clinical usefulness",
            "Review time",
            "Simulink",
            "100k patients/year",
            "Bandwidth",
            "Throughput",
            "Review capacity",
            "Resource optimization",
            "Published benchmarks",
            "Ablation study",
            "Clinical validation"
        ]

        actual_reqs = [r["Requirement"] for r in table]
        self.assertEqual(actual_reqs, expected_reqs)

        # Confirm honest reporting of Referable DR sensitivity failure
        sens_row = next(r for r in table if r["Requirement"] == "Referable DR sensitivity")
        self.assertEqual(sens_row["Status"], "FAIL")

        # Confirm honest reporting of human trial requirements
        time_row = next(r for r in table if r["Requirement"] == "Review time")
        self.assertEqual(time_row["Status"], "PENDING")


if __name__ == "__main__":
    unittest.main()
