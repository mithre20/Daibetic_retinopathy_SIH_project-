/* ============================================================
   Documentation content — renders documentation sections
   ============================================================ */

const docsContent = {
    overview: `
        <h2>Pipeline Architecture</h2>
        <p>The DR Screening Pipeline processes fundus images through five stages, designed to run standalone (each module can be tested/tuned independently) and end-to-end.</p>

        <pre>
 Fundus image (portable camera, field conditions)
        │
        ▼
 ┌───────────────────────────┐
 │ 1. Quality Assessment      │  assessImageQuality.m
 │    + Adaptive Enhancement  │  enhanceImage.m
 └───────────┬───────────────┘
             │ (reject → recapture message)
             ▼
 ┌───────────────────────────┐
 │ 2. Structure Segmentation  │  detectOpticDiscFovea.m
 │    - Optic disc / fovea    │  segmentVessels.m
 │    - Vessels               │  detectMicroaneurysms.m
 │    - Microaneurysms        │  segmentExudates.m
 │    - Exudates              │  detectHemorrhages.m
 │    - Hemorrhages           │  detectNeovascularization.m
 │    - Neovascularization    │
 └───────────┬───────────────┘
             ▼
 ┌───────────────────────────┐
 │ 3. DR Severity Grading     │  buildDRClassifier.m
 │    CNN (global) ⊕ lesion   │  trainDRModel.m
 │    rule-score (local),     │  gradeDRSeverity.m
 │    fused → ICDR grade 0-4  │
 └───────────┬───────────────┘
             ▼
 ┌───────────────────────────┐
 │ 4. Explainability          │  generateGradCAM.m
 │    Grad-CAM + lesion map + │  generateClinicalReport.m
 │    confidence + PDF report │
 └───────────┬───────────────┘
             ▼
 ┌───────────────────────────┐
 │ 5. Performance Validation  │  evaluatePerformance.m
 └───────────────────────────┘</pre>

        <h3>Required MATLAB Toolboxes (original)</h3>
        <ul>
            <li>Image Processing Toolbox</li>
            <li>Computer Vision Toolbox</li>
            <li>Deep Learning Toolbox (with pretrained ResNet50/EfficientNet-b0)</li>
            <li>Statistics and Machine Learning Toolbox</li>
            <li>Simulink (+ SimEvents recommended)</li>
        </ul>

        <div class="callout info">This web version ports the pipeline to JavaScript/Canvas API. Image processing runs entirely in-browser with simplified algorithms. The rule-based grading pathway is fully functional; the CNN branch requires the MATLAB version with a trained network.</div>

        <h3>Folder Structure (Original MATLAB Project)</h3>
        <pre>
src/preprocessing/     Quality gate + enhancement
src/segmentation/      Anatomical + lesion segmentation
src/classification/    CNN+rule fusion grading (ICDR 0-4)
src/explainability/    Grad-CAM, clinical PDF report
src/evaluation/        Sensitivity/specificity/AUC vs. benchmarks
simulink/              Telemedicine throughput & capacity model
docs/                  Dataset notes, clinical mapping, validation plan
main_pipeline.m        End-to-end orchestration for one image</pre>
    `,

    clinical: `
        <h2>Clinical Mapping — ICDR Severity Scale</h2>
        <p>Output strictly follows the International Clinical DR (ICDR) severity scale, Levels 0–4.</p>

        <table>
            <thead>
                <tr><th>Grade</th><th>Name</th><th>ICDR Clinical Definition</th><th>Pipeline Approximation</th></tr>
            </thead>
            <tbody>
                <tr><td>0</td><td>No DR</td><td>No abnormalities</td><td>No lesions detected by any module</td></tr>
                <tr><td>1</td><td>Mild NPDR</td><td>Microaneurysms only</td><td><code>maCount ≥ 1</code>, no hemorrhages/exudates</td></tr>
                <tr><td>2</td><td>Moderate NPDR</td><td>More than mild, less than severe</td><td><code>maCount ≥ 5</code> OR any dot hemorrhage OR exudate area > 0.2% FOV</td></tr>
                <tr><td>3</td><td>Severe NPDR</td><td>4-2-1 rule</td><td>Proxied by blot-hemorrhage burden + exudate extent</td></tr>
                <tr><td>4</td><td>PDR</td><td>Neovascularization</td><td><code>detectNeovascularization</code> flag with confidence gate</td></tr>
            </tbody>
        </table>

        <div class="callout warning"><strong>Caveat on Grade 3:</strong> The real ICDR 4-2-1 rule requires per-quadrant lesion counts and specific patterns (venous beading, IRMA) that need dedicated detectors this prototype does not yet implement. The current Grade-3 rule is a documented, conservative proxy.</div>

        <h3>Why Sensitivity is Prioritized Over Specificity</h3>
        <p>The targets (>90% sensitivity, >85% specificity for referable DR) reflect standard DR screening program targets (e.g. NHS Diabetic Eye Screening Programme). In a screening context, a false negative (missed referable case) risks preventable blindness; a false positive costs one extra ophthalmologist review.</p>
        <p>The grading fusion logic is deliberately asymmetric: on CNN/rule disagreement it never grades <em>down</em> from a strong rule-based severe signal, and defers to manual review rather than silently picking a grade.</p>
    `,

    datasets: `
        <h2>Datasets for Training & Validation</h2>
        <p>No patient images ship with this project. Point the training/evaluation scripts at one of these public, research-license datasets:</p>

        <table>
            <thead>
                <tr><th>Dataset</th><th>Images</th><th>Labels</th><th>Notes</th></tr>
            </thead>
            <tbody>
                <tr><td><strong>IDRiD</strong></td><td>516</td><td>ICDR grade + pixel-level masks</td><td>Indian population; has lesion-level segmentation ground truth</td></tr>
                <tr><td><strong>Messidor-2</strong></td><td>1,748</td><td>ICDR grade + referable DR</td><td>Widely used in FDA/CE submissions</td></tr>
                <tr><td><strong>APTOS 2019</strong></td><td>~3,662</td><td>ICDR grade</td><td>Larger, more label noise; good for pretraining</td></tr>
                <tr><td><strong>EyePACS</strong></td><td>~88,000</td><td>ICDR grade</td><td>Largest; heavily class-imbalanced (~73% Grade 0)</td></tr>
            </tbody>
        </table>

        <h3>Recommended Training Recipe</h3>
        <ol>
            <li>Pretrain / fine-tune the CNN branch on EyePACS + APTOS (volume)</li>
            <li>Fine-tune again on IDRiD + Messidor-2 (population/label-quality match)</li>
            <li>Hold out IDRiD's test split for final evaluation — never report metrics on training data</li>
            <li>Capture a local pilot-site sample (~200-500 images) on the actual camera for field validation</li>
        </ol>

        <h3>CSV Label Format</h3>
        <pre>image,grade
IDRiD_001.jpg,2
IDRiD_002.jpg,0
...</pre>
        <p><code>grade</code> in {0,1,2,3,4} per the ICDR scale.</p>
    `,

    validation: `
        <h2>Clinical Validation Plan</h2>

        <h3>Validation Study Design (Recommended)</h3>
        <ol>
            <li><strong>Retrospective bench validation</strong> — run <code>evaluatePerformance.m</code> against held-out IDRiD/Messidor-2 test splits. Report sensitivity/specificity/AUC for referable DR, and 5-class accuracy.</li>
            <li><strong>Reader study</strong> — have 2-3 independent ophthalmologists grade the same test set (blinded to pipeline output); compute pipeline-vs-consensus agreement (Cohen's kappa).</li>
            <li><strong>Explainability usability study</strong> — time how long it takes reviewing ophthalmologists to reach a decision using the clinical report outputs vs. the raw image alone; target is the &lt;30s figure.</li>
            <li><strong>Prospective field pilot</strong> — deploy at 1-2 PHCs on the actual portable camera hardware, compare rejection rate and grading performance against retrospective numbers.</li>
            <li><strong>Regulatory</strong> — in India, CDSCO Medical Device Rules; classification and required clinical evidence should be confirmed with a regulatory consultant.</li>
        </ol>

        <h3>Benchmark Comparison Points</h3>
        <table>
            <thead><tr><th>System</th><th>Sensitivity</th><th>Specificity</th><th>Note</th></tr></thead>
            <tbody>
                <tr><td>IDx-DR (pivotal trial)</td><td>87.2%</td><td>90.7%</td><td>FDA-approved; verify exact figures before citing</td></tr>
                <tr><td>Google ARDA</td><td>~90%</td><td>~90%</td><td>Reported range across validation cohorts</td></tr>
                <tr><td>EyePACS baseline CNN</td><td>~85%</td><td>~85%</td><td>Representative single-CNN baseline</td></tr>
            </tbody>
        </table>

        <div class="callout warning"><strong>Benchmark citation caveat:</strong> These numbers are illustrative reference points from general knowledge of the published literature, not verified against the primary papers. Before using them in any report or regulatory document, look up and cite the specific paper/trial.</div>
    `,

    simulink: `
        <h2>Simulink Workflow Simulation</h2>
        <p>Two complementary artifacts are provided for telemedicine capacity modeling:</p>

        <h3>1. <code>capacityPlanningAnalysis.m</code> — Analytical Queueing Model</h3>
        <ul>
            <li>Runs in plain MATLAB, no Simulink license required</li>
            <li>Uses M/M/1 and M/M/c (Erlang-C) queueing formulas</li>
            <li>Directly computes minimum number of servers needed</li>
            <li>Produces a sensitivity table for staffing decisions</li>
            <li><strong>Best for:</strong> "How many of X do we need?" — quick and reproducible answers</li>
        </ul>

        <div class="callout info">This web version includes a full JavaScript port of the analytical queueing model in the Capacity Planning tab. It produces identical results to the MATLAB version.</div>

        <h3>2. <code>buildScreeningWorkflowModel.m</code> — Simulink Block Diagram</h3>
        <ul>
            <li>Must be run inside MATLAB (creates <code>ScreeningWorkflow.slx</code>)</li>
            <li>Visual, editable block diagram: arrival → upload → AI → referral split → review → outcomes</li>
            <li>Uses Transport Delay + Saturation blocks as a continuous-time approximation</li>
            <li>Can be extended with SimEvents for stochastic discrete-event simulation</li>
            <li><strong>Best for:</strong> "What if" scenarios and stakeholder communication</li>
        </ul>

        <h3>Recommended Workflow</h3>
        <ol>
            <li>Run the analytical model first to get sizing numbers</li>
            <li>Feed those numbers into the Simulink model for visualization</li>
            <li>If SimEvents is available, extend for stochastic modeling</li>
        </ol>
    `,

    code: `
        <h2>Source Code Reference</h2>
        <p>All source files in the original MATLAB project, grouped by component:</p>

        <h3>Preprocessing</h3>
        <table>
            <thead><tr><th>File</th><th>Purpose</th></tr></thead>
            <tbody>
                <tr><td><code>assessImageQuality.m</code></td><td>Quality gate — sharpness (Laplacian variance), illumination uniformity (tile grid), FOV coverage (Otsu + connected components), contrast (RMS). Weighted score → accept/enhance/reject.</td></tr>
                <tr><td><code>enhanceImage.m</code></td><td>Adaptive enhancement — Lab color space, illumination normalization (Gaussian background subtraction), CLAHE on luminance, bilateral denoising.</td></tr>
            </tbody>
        </table>

        <h3>Segmentation</h3>
        <table>
            <thead><tr><th>File</th><th>Purpose</th></tr></thead>
            <tbody>
                <tr><td><code>detectOpticDiscFovea.m</code></td><td>Optic disc via Hough circles on bright channel; fovea via darkest-region search offset from disc.</td></tr>
                <tr><td><code>segmentVessels.m</code></td><td>Multi-scale Frangi vesselness (Hessian eigenvalue analysis) + hysteresis thresholding.</td></tr>
                <tr><td><code>detectMicroaneurysms.m</code></td><td>Rotating morphological top-hat + shape filtering + sub-pixel parabolic refinement.</td></tr>
                <tr><td><code>segmentExudates.m</code></td><td>Morphological reconstruction residual + eccentricity filtering (reject vessel highlights).</td></tr>
                <tr><td><code>detectHemorrhages.m</code></td><td>Dark-lesion top-hat + dot/blot typing by area + eccentricity + solidity.</td></tr>
                <tr><td><code>detectNeovascularization.m</code></td><td>Vessel-caliber irregularity + local density anomaly heuristic (screening flag only).</td></tr>
            </tbody>
        </table>

        <h3>Classification</h3>
        <table>
            <thead><tr><th>File</th><th>Purpose</th></tr></thead>
            <tbody>
                <tr><td><code>buildDRClassifier.m</code></td><td>Transfer-learning CNN (ResNet50/EfficientNet-b0/DenseNet201) with 5-class ICDR head.</td></tr>
                <tr><td><code>trainDRModel.m</code></td><td>Fine-tuning with inverse-frequency class weighting and augmentation.</td></tr>
                <tr><td><code>gradeDRSeverity.m</code></td><td>CNN + rule-score fusion with safety-net logic (never grades down from severe signal).</td></tr>
            </tbody>
        </table>

        <h3>Explainability</h3>
        <table>
            <thead><tr><th>File</th><th>Purpose</th></tr></thead>
            <tbody>
                <tr><td><code>generateGradCAM.m</code></td><td>Grad-CAM attention map (or lesion-density heatmap fallback) + lesion-correlation scoring.</td></tr>
                <tr><td><code>generateClinicalReport.m</code></td><td>One-page annotated report for &lt;30s ophthalmologist triage.</td></tr>
            </tbody>
        </table>

        <h3>Evaluation & Simulation</h3>
        <table>
            <thead><tr><th>File</th><th>Purpose</th></tr></thead>
            <tbody>
                <tr><td><code>evaluatePerformance.m</code></td><td>Sensitivity/specificity/AUC for referable DR + benchmark comparison table.</td></tr>
                <tr><td><code>capacityPlanningAnalysis.m</code></td><td>M/M/c analytical queueing model for district program sizing.</td></tr>
                <tr><td><code>buildScreeningWorkflowModel.m</code></td><td>Programmatic Simulink .slx model builder for the screening workflow.</td></tr>
            </tbody>
        </table>
    `
};

/** Initialize documentation tab */
function initDocs() {
    const content = document.getElementById('docs-content');
    const tabs = document.querySelectorAll('.doc-tab');

    function showDoc(id) {
        content.innerHTML = docsContent[id] || '<p>Documentation not found.</p>';
        tabs.forEach(t => t.classList.toggle('active', t.dataset.doc === id));
    }

    tabs.forEach(tab => {
        tab.addEventListener('click', () => showDoc(tab.dataset.doc));
    });

    // Show initial doc
    showDoc('overview');
}

window.initDocs = initDocs;
