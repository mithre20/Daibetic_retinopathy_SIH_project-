# Clinical Mapping & Validation Plan

## ICDR severity scale implemented

| Grade | Name | Clinical definition (ICDR) | How this pipeline approximates it |
|---|---|---|---|
| 0 | No DR | No abnormalities | No lesions detected by any module |
| 1 | Mild NPDR | Microaneurysms only | `maCount >= 1`, no hemorrhages/exudates |
| 2 | Moderate NPDR | More than mild, less than severe | `maCount >= 5` OR any dot hemorrhage OR exudate area > 0.2% FOV |
| 3 | Severe NPDR | 4-2-1 rule (severe hemorrhages in 4 quadrants, OR venous beading in 2, OR IRMA in 1) | Proxied by blot-hemorrhage burden + exudate extent (see caveat below — true 4-2-1 needs quadrant-registered masks, a documented simplification) |
| 4 | PDR | Neovascularization and/or vitreous/preretinal hemorrhage | `detectNeovascularization` flag with confidence gate |

**Caveat on Grade 3**: the real ICDR 4-2-1 rule requires per-quadrant
lesion counts and specific patterns (venous beading, IRMA) that need
dedicated detectors this prototype does not yet implement. The current
Grade-3 rule is a documented, conservative proxy — treat any pipeline
output of Grade 2 or 3 as "route to human grading," not as a final severe/
moderate distinction a clinician should rely on without review.

## Why sensitivity is prioritized over specificity

The task brief's targets (>90% sensitivity, >85% specificity for referable
DR) reflect standard DR screening program targets (e.g. NHS Diabetic Eye
Screening Programme, UK). In a screening context, a false negative (missed
referable case) risks preventable blindness; a false positive costs one
extra ophthalmologist review. `gradeDRSeverity.m`'s fusion logic is
deliberately asymmetric: on CNN/rule disagreement it never grades *down*
from a strong rule-based severe signal, and defers to manual review rather
than silently picking a grade.

## Validation study design (recommended, not yet executed)

1. **Retrospective bench validation** — run `evaluatePerformance.m` against
   held-out IDRiD/Messidor-2 test splits (see `DATASETS.md`). Report
   sensitivity/specificity/AUC for referable DR, and 5-class accuracy.
2. **Reader study** — have 2-3 independent ophthalmologists grade the same
   test set (blinded to pipeline output); compute pipeline-vs-consensus
   agreement (Cohen's kappa) in addition to sensitivity/specificity against
   any single reference grader, since inter-grader disagreement on DR
   grading is itself well documented in the literature.
3. **Explainability usability study** — time how long it takes reviewing
   ophthalmologists to reach a decision using `generateClinicalReport.m`
   outputs vs. the raw image alone; target is the <30s figure in the brief.
   Also collect a simple 1-5 "was the highlighted region clinically
   relevant" rating per case to quantify Grad-CAM/lesion-correlation
   usefulness beyond the automated overlap metric in `generateGradCAM.m`.
4. **Prospective field pilot** — deploy at 1-2 PHCs on the actual portable
   camera hardware, compare `assessImageQuality.m` rejection rate and final
   grading performance against the retrospective numbers (field images are
   reliably harder than curated public datasets).
5. **Regulatory** — in India, an AI-based diagnostic-support device of this
   kind falls under CDSCO's Medical Device Rules; classification and
   required clinical evidence should be confirmed with a regulatory
   consultant before any patient-facing deployment.

## Benchmark citation caveat

`evaluatePerformance.m` prints a comparison table against IDx-DR, Google
ARDA, and an EyePACS baseline CNN. These numbers are **illustrative
reference points** reconstructed from general knowledge of the published
literature, not verified against the primary papers in this session — before
using them in any report, presentation, or regulatory document, look up and
cite the specific paper/trial each number comes from.
