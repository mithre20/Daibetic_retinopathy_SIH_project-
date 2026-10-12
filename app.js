/* ============================================================
   DR Screening Pipeline — Main Application Controller
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    // ===================== Background canvas =====================
    const bgCanvas = document.getElementById('bg-canvas');
    const bgCtx = bgCanvas.getContext('2d');
    let particles = [];

    function resizeBg() {
        bgCanvas.width = window.innerWidth;
        bgCanvas.height = window.innerHeight;
    }
    resizeBg();
    window.addEventListener('resize', resizeBg);

    function createParticles() {
        particles = [];
        const count = Math.floor(window.innerWidth * window.innerHeight / 15000);
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * bgCanvas.width,
                y: Math.random() * bgCanvas.height,
                r: Math.random() * 1.5 + 0.3,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                alpha: Math.random() * 0.3 + 0.05
            });
        }
    }
    createParticles();

    function animateBg() {
        bgCtx.clearRect(0, 0, bgCanvas.width, bgCanvas.height);
        for (const p of particles) {
            p.x += p.vx;
            p.y += p.vy;
            if (p.x < 0) p.x = bgCanvas.width;
            if (p.x > bgCanvas.width) p.x = 0;
            if (p.y < 0) p.y = bgCanvas.height;
            if (p.y > bgCanvas.height) p.y = 0;

            bgCtx.beginPath();
            bgCtx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            bgCtx.fillStyle = `rgba(110, 231, 183, ${p.alpha})`;
            bgCtx.fill();
        }

        // Draw faint connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    bgCtx.beginPath();
                    bgCtx.moveTo(particles[i].x, particles[i].y);
                    bgCtx.lineTo(particles[j].x, particles[j].y);
                    bgCtx.strokeStyle = `rgba(110, 231, 183, ${0.04 * (1 - dist / 120)})`;
                    bgCtx.lineWidth = 0.5;
                    bgCtx.stroke();
                }
            }
        }
        requestAnimationFrame(animateBg);
    }
    animateBg();

    // ===================== Tab Navigation =====================
    const tabs = document.querySelectorAll('.nav-tab');
    const panels = document.querySelectorAll('.panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            panels.forEach(p => {
                p.classList.toggle('active', p.id === `panel-${target}`);
            });
        });
    });

    // ===================== File Upload =====================
    const dropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('file-input');
    const previewArea = document.getElementById('upload-preview-area');
    const previewCanvas = document.getElementById('preview-canvas');
    const previewFilename = document.getElementById('preview-filename');
    const previewDimensions = document.getElementById('preview-dimensions');
    const btnChangeImage = document.getElementById('btn-change-image');
    const runSection = document.getElementById('run-section');
    const btnRun = document.getElementById('btn-run-pipeline');

    let pipeline = null;
    let currentFile = null;

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('drag-over');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
    });
    btnChangeImage.addEventListener('click', () => {
        fileInput.click();
    });

    async function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        currentFile = file;
        pipeline = new DRPipeline();

        try {
            const info = await pipeline.loadImage(file);

            // Show preview
            const pCtx = previewCanvas.getContext('2d');
            previewCanvas.width = info.width;
            previewCanvas.height = info.height;
            pCtx.putImageData(info.imageData, 0, 0);

            previewFilename.textContent = file.name;
            previewDimensions.textContent = `${info.width} × ${info.height} px (processing resolution)`;

            dropzone.style.display = 'none';
            previewArea.style.display = 'flex';
            runSection.style.display = 'block';

            // Hide previous results
            document.getElementById('pipeline-progress').style.display = 'none';
            document.getElementById('results-section').style.display = 'none';
        } catch (err) {
            alert('Failed to load image: ' + err.message);
        }
    }

    // ===================== Medical Report & Health Analysis Handlers =====================
    window.currentHealthReportData = null;
    window.currentManualVitals = null;

    const reportDropzone = document.getElementById('report-dropzone');
    const reportFileInput = document.getElementById('report-file-input');
    const btnBrowseReport = document.getElementById('btn-browse-report');
    const btnClearReport = document.getElementById('btn-clear-report');
    const reportPreviewStatus = document.getElementById('report-preview-status');
    const reportFilenameDisplay = document.getElementById('report-filename-display');
    const reportStatusMsg = document.getElementById('report-status-msg');

    if (btnBrowseReport && reportFileInput) {
        btnBrowseReport.addEventListener('click', (e) => {
            e.stopPropagation();
            reportFileInput.click();
        });
    }
    if (reportDropzone && reportFileInput) {
        reportDropzone.addEventListener('click', () => reportFileInput.click());
        reportDropzone.addEventListener('dragover', (e) => { e.preventDefault(); reportDropzone.style.background = 'rgba(56,189,248,0.08)'; });
        reportDropzone.addEventListener('dragleave', () => { reportDropzone.style.background = 'none'; });
        reportDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            reportDropzone.style.background = 'none';
            if (e.dataTransfer.files.length > 0) handleReportFile(e.dataTransfer.files[0]);
        });
        reportFileInput.addEventListener('change', () => {
            if (reportFileInput.files.length > 0) handleReportFile(reportFileInput.files[0]);
        });
    }

    if (btnClearReport) {
        btnClearReport.addEventListener('click', (e) => {
            e.stopPropagation();
            window.currentHealthReportData = null;
            if (reportFileInput) reportFileInput.value = '';
            if (reportPreviewStatus) reportPreviewStatus.style.display = 'none';
            updateHealthReportSummaryCards();
        });
    }

    async function handleReportFile(file) {
        const ext = file.name.split('.').pop().toLowerCase();
        if (!['pdf', 'jpg', 'jpeg', 'png'].includes(ext)) {
            alert('Unsupported report format. Please upload a PDF, JPG, or PNG file.');
            return;
        }

        if (reportPreviewStatus) reportPreviewStatus.style.display = 'block';
        if (reportFilenameDisplay) reportFilenameDisplay.textContent = file.name;
        if (reportStatusMsg) reportStatusMsg.textContent = 'Extracting clinical text & parameters...';

        try {
            const formData = new FormData();
            formData.append('report', file);
            const resp = await fetch('/api/analyze-report', { method: 'POST', body: formData });
            if (resp.ok) {
                const data = await resp.json();
                window.currentHealthReportData = data;
                if (reportStatusMsg) {
                    const paramsFound = [];
                    if (data.glucose && Object.keys(data.glucose).length > 0) paramsFound.push('Glucose');
                    if (data.blood_pressure) paramsFound.push('Blood Pressure');
                    const summaryTxt = paramsFound.length > 0 
                        ? `Extracted: ${paramsFound.join(', ')} (Confidence: ${(data.extraction_confidence*100).toFixed(0)}%)` 
                        : 'No standard vitals recognized in document.';
                    reportStatusMsg.textContent = summaryTxt;
                }
                populateEditInputsFromReport(data);
                updateHealthReportSummaryCards();
            } else {
                const errData = await resp.json().catch(() => ({}));
                if (reportStatusMsg) reportStatusMsg.textContent = `Extraction notice: ${errData.error || 'Failed to parse'}`;
            }
        } catch (err) {
            console.warn('Report extraction request error:', err);
            if (reportStatusMsg) reportStatusMsg.textContent = 'Report loaded locally (server analysis unavailable).';
        }
    }

    // Manual Vitals save
    const btnSaveManual = document.getElementById('btn-save-manual-vitals');
    if (btnSaveManual) {
        btnSaveManual.addEventListener('click', () => {
            const sys = parseInt(document.getElementById('manual-bp-systolic').value, 10);
            const dia = parseInt(document.getElementById('manual-bp-diastolic').value, 10);
            const gType = document.getElementById('manual-glucose-type').value;
            const gVal = parseFloat(document.getElementById('manual-glucose-value').value);
            const gUnit = document.getElementById('manual-glucose-unit').value;

            window.currentManualVitals = {
                manual_bp: (!isNaN(sys) && !isNaN(dia)) ? { systolic: sys, diastolic: dia, unit: 'mmHg' } : null,
                manual_glucose: (!isNaN(gVal)) ? { test_type: gType, value: gVal, unit: gType === 'hba1c' ? '%' : gUnit } : null
            };

            updateHealthReportSummaryCards();
            alert('Manual health vitals recorded.');
        });
    }

    // Toggle and Confirm Verification
    const btnToggleEdit = document.getElementById('btn-toggle-edit-vitals');
    const editContainer = document.getElementById('edit-vitals-container');
    if (btnToggleEdit && editContainer) {
        btnToggleEdit.addEventListener('click', () => {
            const isHidden = editContainer.style.display === 'none';
            editContainer.style.display = isHidden ? 'block' : 'none';
            btnToggleEdit.textContent = isHidden ? '✕ Close Edit' : '✎ Edit Values';
        });
    }

    const btnConfirmVitals = document.getElementById('btn-confirm-vitals');
    if (btnConfirmVitals) {
        btnConfirmVitals.addEventListener('click', () => {
            const gVal = parseFloat(document.getElementById('edit-glucose-val').value);
            const a1cVal = parseFloat(document.getElementById('edit-hba1c-val').value);
            const sysVal = parseInt(document.getElementById('edit-bp-sys').value, 10);
            const diaVal = parseInt(document.getElementById('edit-bp-dia').value, 10);

            if (!window.currentHealthReportData) {
                window.currentHealthReportData = { success: true, glucose: {}, blood_pressure: null };
            }

            if (!isNaN(gVal)) {
                window.currentHealthReportData.glucose.fasting = {
                    parameter: 'Fasting Blood Glucose',
                    value: gVal,
                    unit: 'mg/dL',
                    source: 'Operator Verified',
                    status: (gVal > 125 ? 'ABOVE_RANGE' : gVal < 70 ? 'BELOW_RANGE' : 'WITHIN_RANGE'),
                    interpretation: 'Verified by operator.'
                };
            }
            if (!isNaN(a1cVal)) {
                window.currentHealthReportData.glucose.hba1c = {
                    parameter: 'HbA1c',
                    value: a1cVal,
                    unit: '%',
                    source: 'Operator Verified',
                    status: (a1cVal >= 6.5 ? 'ABOVE_RANGE' : 'WITHIN_RANGE'),
                    interpretation: 'Verified by operator.'
                };
            }
            if (!isNaN(sysVal) && !isNaN(diaVal) && sysVal > diaVal) {
                window.currentHealthReportData.blood_pressure = {
                    systolic: sysVal,
                    diastolic: diaVal,
                    unit: 'mmHg',
                    formatted: `${sysVal}/${diaVal} mmHg`,
                    source: 'Operator Verified',
                    status: (sysVal >= 140 || diaVal >= 90 ? 'ELEVATED' : 'STANDARD'),
                    interpretation: 'Verified by operator.'
                };
            }

            if (editContainer) editContainer.style.display = 'none';
            if (btnToggleEdit) btnToggleEdit.textContent = '✎ Edit Values';
            updateHealthReportSummaryCards();
            alert('Extracted values confirmed.');
        });
    }

    function populateEditInputsFromReport(reportData) {
        if (!reportData) return;
        const g = reportData.glucose || {};
        const fasting = g.fasting || g.random || g.general;
        const hba1c = g.hba1c;
        const bp = reportData.blood_pressure;

        const elG = document.getElementById('edit-glucose-val');
        const elA = document.getElementById('edit-hba1c-val');
        const elSys = document.getElementById('edit-bp-sys');
        const elDia = document.getElementById('edit-bp-dia');

        if (elG && fasting) elG.value = fasting.value;
        if (elA && hba1c) elA.value = hba1c.value;
        if (elSys && bp) elSys.value = bp.systolic;
        if (elDia && bp) elDia.value = bp.diastolic;
    }

    function updateHealthReportSummaryCards() {
        const badge = document.getElementById('health-data-source-badge');
        const elGVal = document.getElementById('health-glucose-val');
        const elGSub = document.getElementById('health-glucose-sub');
        const elGStat = document.getElementById('health-glucose-status');

        const elAVal = document.getElementById('health-hba1c-val');
        const elASub = document.getElementById('health-hba1c-sub');
        const elAStat = document.getElementById('health-hba1c-status');

        const elBPVal = document.getElementById('health-bp-val');
        const elBPSrc = document.getElementById('health-bp-source');
        const elBPStat = document.getElementById('health-bp-status');

        // Combined summary fields
        const sumG = document.getElementById('summary-glucose-val');
        const sumA = document.getElementById('summary-hba1c-val');
        const sumBP = document.getElementById('summary-bp-val');

        let rep = window.currentHealthReportData;
        let man = window.currentManualVitals;

        // Glucose
        let gObj = null;
        if (rep && rep.glucose) {
            gObj = rep.glucose.fasting || rep.glucose.random || rep.glucose.postprandial || rep.glucose.general;
        }
        if (!gObj && man && man.manual_glucose && man.manual_glucose.test_type !== 'hba1c') {
            gObj = {
                parameter: `${man.manual_glucose.test_type.toUpperCase()} Glucose`,
                value: man.manual_glucose.value,
                unit: man.manual_glucose.unit,
                source: 'Manual Entry',
                status: 'RECORDED'
            };
        }

        if (gObj && elGVal) {
            elGVal.textContent = `${gObj.value} ${gObj.unit}`;
            if (elGSub) elGSub.textContent = gObj.reference_range ? `Ref: ${gObj.reference_range}` : `Source: ${gObj.source || 'Report'}`;
            if (elGStat) {
                elGStat.textContent = gObj.status || 'RECORDED';
                elGStat.style.color = gObj.status === 'ABOVE_RANGE' ? '#f59e0b' : '#10b981';
            }
            if (sumG) sumG.textContent = `${gObj.value} ${gObj.unit} (${gObj.source || 'Report'})`;
        } else if (elGVal) {
            elGVal.textContent = '—';
            if (elGSub) elGSub.textContent = 'Ref: —';
            if (elGStat) { elGStat.textContent = 'Not Provided'; elGStat.style.color = '#94a3b8'; }
            if (sumG) sumG.textContent = 'Not provided';
        }

        // HbA1c
        let a1cObj = null;
        if (rep && rep.glucose && rep.glucose.hba1c) {
            a1cObj = rep.glucose.hba1c;
        }
        if (!a1cObj && man && man.manual_glucose && man.manual_glucose.test_type === 'hba1c') {
            a1cObj = {
                parameter: 'HbA1c',
                value: man.manual_glucose.value,
                unit: '%',
                source: 'Manual Entry',
                status: 'RECORDED'
            };
        }

        if (a1cObj && elAVal) {
            elAVal.textContent = `${a1cObj.value} %`;
            if (elASub) elASub.textContent = a1cObj.reference_range ? `Ref: ${a1cObj.reference_range}` : `Source: ${a1cObj.source || 'Report'}`;
            if (elAStat) {
                elAStat.textContent = a1cObj.status || 'RECORDED';
                elAStat.style.color = a1cObj.status === 'ABOVE_RANGE' ? '#f59e0b' : '#10b981';
            }
            if (sumA) sumA.textContent = `${a1cObj.value} % (${a1cObj.source || 'Report'})`;
        } else if (elAVal) {
            elAVal.textContent = '—';
            if (elASub) elASub.textContent = 'Ref: —';
            if (elAStat) { elAStat.textContent = 'Not Provided'; elAStat.style.color = '#94a3b8'; }
            if (sumA) sumA.textContent = 'Not provided';
        }

        // Blood Pressure
        let bpObj = null;
        if (rep && rep.blood_pressure) {
            bpObj = rep.blood_pressure;
        }
        if (!bpObj && man && man.manual_bp) {
            bpObj = {
                systolic: man.manual_bp.systolic,
                diastolic: man.manual_bp.diastolic,
                unit: 'mmHg',
                formatted: `${man.manual_bp.systolic}/${man.manual_bp.diastolic} mmHg`,
                source: 'Manual Entry',
                status: (man.manual_bp.systolic >= 140 || man.manual_bp.diastolic >= 90 ? 'ELEVATED' : 'STANDARD')
            };
        }

        if (bpObj && elBPVal) {
            elBPVal.textContent = bpObj.formatted || `${bpObj.systolic}/${bpObj.diastolic} ${bpObj.unit || 'mmHg'}`;
            if (elBPSrc) elBPSrc.textContent = `Source: ${bpObj.source || 'Report'}`;
            if (elBPStat) {
                elBPStat.textContent = bpObj.status || 'STANDARD';
                elBPStat.style.color = bpObj.status === 'ELEVATED' ? '#ef4444' : bpObj.status === 'BORDERLINE' ? '#f59e0b' : '#10b981';
            }
            if (sumBP) sumBP.textContent = `${bpObj.formatted || bpObj.systolic + '/' + bpObj.diastolic + ' mmHg'} (${bpObj.source || 'Report'})`;
        } else if (elBPVal) {
            elBPVal.textContent = '—';
            if (elBPSrc) elBPSrc.textContent = 'Source: —';
            if (elBPStat) { elBPStat.textContent = 'Not Provided'; elBPStat.style.color = '#94a3b8'; }
            if (sumBP) sumBP.textContent = 'Not provided';
        }

        if (badge) {
            if (rep && man && man.manual_bp) {
                badge.textContent = 'HYBRID: REPORT + MANUAL';
                badge.style.color = '#a855f7';
            } else if (rep) {
                badge.textContent = `REPORT: ${rep.file_type || 'UPLOADED'}`;
                badge.style.color = '#38bdf8';
            } else if (man) {
                badge.textContent = 'MANUAL CLINICAL ENTRY';
                badge.style.color = '#f59e0b';
            } else {
                badge.textContent = 'NO REPORT LOADED';
                badge.style.color = '#94a3b8';
            }
        }
    }

    // ===================== Run Pipeline =====================
    btnRun.addEventListener('click', async () => {
        if (!pipeline) return;

        btnRun.disabled = true;
        const progress = document.getElementById('pipeline-progress');
        const bar = document.getElementById('pipeline-bar');
        progress.style.display = 'block';
        document.getElementById('results-section').style.display = 'none';

        // Reset steps
        for (let i = 1; i <= 5; i++) {
            const step = document.getElementById(`step-${i}`);
            step.className = 'progress-step';
            document.getElementById(`step-${i}-status`).textContent = '';
        }
        bar.style.width = '0%';

        // Scroll to progress
        progress.scrollIntoView({ behavior: 'smooth', block: 'start' });

        let currentStep = 0;
        const result = await pipeline.runPipeline((step, msg) => {
            if (step !== currentStep) {
                // Mark previous as done
                if (currentStep > 0) {
                    document.getElementById(`step-${currentStep}`).className = 'progress-step done';
                }
                currentStep = step;
                document.getElementById(`step-${step}`).className = 'progress-step active';
            }
            document.getElementById(`step-${step}-status`).textContent = msg;
            bar.style.width = `${(step / 5) * 100}%`;
        });

        // Attempt Deep Learning Screening via PyTorch backend (/api/screen)
        let dlResult = null;
        let iqaResult = null;
        let gateResult = null;
        let referralResult = null;

        if (currentFile) {
            try {
                const formData = new FormData();
                formData.append('image', currentFile);
                const resp = await fetch('/api/screen', { method: 'POST', body: formData });
                if (resp.ok) {
                    const screenData = await resp.json();
                    iqaResult = screenData.image_quality;
                    gateResult = screenData.screening_gate;
                    dlResult = screenData.dr_prediction;
                    referralResult = screenData.referral_decision;
                    window.currentIqaResult = iqaResult;
                    window.currentGateResult = gateResult;
                    window.currentReferralResult = referralResult;
                    console.log('Screening Pipeline output:', screenData);

                    // Update Step 1 with IQA status
                    if (iqaResult) {
                        const s1Status = document.getElementById('step-1-status');
                        if (s1Status) {
                            s1Status.textContent = `${iqaResult.predicted_class} (${(iqaResult.confidence * 100).toFixed(0)}%)`;
                        }
                    }
                }
            } catch (err) {
                console.info('Backend /api/screen offline, running client-side engine fallback.');
            }
        }

        // Mark last step done
        document.getElementById(`step-${currentStep}`).className = 'progress-step done';
        bar.style.width = '100%';

        // Render results
        await new Promise(r => setTimeout(r, 300));
        renderResults(result, dlResult, iqaResult, gateResult, referralResult);
        btnRun.disabled = false;
    });

    // ===================== Render Results =====================
    window.currentPipelineResult = null;
    window.currentDlResult = null;

    function renderResults(result, dlResult, iqaResult, gateResult, referralResult) {
        window.currentPipelineResult = result;
        window.currentDlResult = dlResult;
        window.currentReferralResult = referralResult;

        // Broadcast screening context for Chatbot Assistant
        try {
            const chatCtx = {
                dr_grade: dlResult ? (dlResult.predicted_class !== undefined ? dlResult.predicted_class : dlResult.prediction) : (result ? result.dr_grade : null),
                dr_label: dlResult ? (dlResult.predicted_label || dlResult.severity) : (result ? result.dr_label : null),
                dr_confidence: dlResult ? dlResult.confidence : (result ? result.dr_confidence : null),
                risk_level: referralResult ? referralResult.risk_level : null,
                referral_priority: referralResult ? referralResult.referral_priority : null,
                iqa_status: iqaResult ? (iqaResult.predicted_class || iqaResult.status) : null,
                screening_allowed: gateResult ? gateResult.screening_allowed : true,
                unet_findings: window.currentUnetResult || null
            };
            localStorage.setItem('aidr_screening_context', JSON.stringify(chatCtx));
            if (window.parent && window.parent !== window) {
                window.parent.postMessage({ action: 'screeningUpdated', context: chatCtx }, '*');
            }
        } catch (e) {}

        const section = document.getElementById('results-section');
        section.style.display = 'block';

        const iqaCard = document.getElementById('iqa-card');
        const blockedCard = document.getElementById('screening-blocked-card');
        const gradeCard = document.getElementById('grade-card');
        const dlCard = document.getElementById('efficientnet-dl-card');
        const gradcamCard = document.getElementById('gradcam-explainability-card');
        const shapCard = document.getElementById('shap-explainability-card');
        const otherCards = document.querySelectorAll('.results-section > .glass-card:not(#iqa-card):not(#screening-blocked-card)');

        const screeningAllowed = gateResult ? gateResult.screening_allowed : (result && result.status !== 'rejected');

        // Populate IQA card
        if (iqaResult && iqaCard) {
            const statusBadge = document.getElementById('iqa-status-badge');
            const classText = document.getElementById('iqa-class-text');
            const confText = document.getElementById('iqa-conf-text');
            const gateAction = document.getElementById('iqa-gate-action');
            const recaptureAlert = document.getElementById('iqa-recapture-alert');
            const recaptureReason = document.getElementById('iqa-recapture-reason');
            const recList = document.getElementById('iqa-recommendations-list');

            const isGood = iqaResult.quality_status === 'ACCEPTED' && iqaResult.screening_allowed;

            if (isGood) {
                iqaCard.style.borderLeft = '4px solid #10b981';
                if (statusBadge) {
                    statusBadge.textContent = 'ACCEPTED: DR SCREENING ALLOWED';
                    statusBadge.style.background = 'rgba(16,185,129,0.15)';
                    statusBadge.style.color = '#10b981';
                    statusBadge.style.border = '1px solid rgba(16,185,129,0.3)';
                }
                if (classText) {
                    classText.textContent = iqaResult.predicted_class;
                    classText.style.color = '#10b981';
                }
                if (confText) {
                    confText.textContent = `Confidence: ${(iqaResult.confidence * 100).toFixed(1)}% (Threshold: ${(iqaResult.threshold * 100).toFixed(1)}%)`;
                }
                if (gateAction) {
                    gateAction.textContent = 'Proceed to DR Severity Grading';
                    gateAction.style.color = '#e2e8f0';
                }
                if (recaptureAlert) recaptureAlert.style.display = 'none';
            } else {
                const badgeColor = iqaResult.predicted_class === 'Unusable' ? '#ef4444' : '#f59e0b';
                iqaCard.style.borderLeft = `4px solid ${badgeColor}`;
                if (statusBadge) {
                    statusBadge.textContent = 'REJECTED: DR SCREENING BLOCKED';
                    statusBadge.style.background = 'rgba(239,68,68,0.15)';
                    statusBadge.style.color = '#f87171';
                    statusBadge.style.border = '1px solid rgba(239,68,68,0.3)';
                }
                if (classText) {
                    classText.textContent = iqaResult.predicted_class;
                    classText.style.color = badgeColor;
                }
                if (confText) {
                    confText.textContent = `Confidence: ${(iqaResult.confidence * 100).toFixed(1)}% (Threshold: ${(iqaResult.threshold * 100).toFixed(1)}%)`;
                }
                if (gateAction) {
                    gateAction.textContent = 'Recapture Fundus Image Required';
                    gateAction.style.color = '#fca5a5';
                }
                if (recaptureAlert) {
                    recaptureAlert.style.display = 'block';
                    if (recaptureReason && gateResult) {
                        recaptureReason.textContent = gateResult.reason || 'Image quality insufficient for automated diagnostic screening.';
                    }
                    if (recList && iqaResult.recommendations) {
                        recList.innerHTML = '<strong style="color:#fecaca;">Operator Recommendations:</strong><ul style="margin:4px 0 0 18px; padding:0;">' +
                            iqaResult.recommendations.map(r => `<li>${r}</li>`).join('') + '</ul>';
                    }
                }
            }

            // Classical metrics
            if (iqaResult.classical_metrics) {
                const cm = iqaResult.classical_metrics;
                const setVal = (id, val) => {
                    const el = document.getElementById(id);
                    if (el && val !== undefined) el.textContent = typeof val === 'number' ? val.toFixed(2) : val;
                };
                setVal('metric-sharpness-val', cm.sharpness_laplacian);
                setVal('metric-illum-val', cm.illumination_uniformity);
                setVal('metric-fov-val', `${(cm.fov_coverage_ratio * 100).toFixed(1)}%`);
                setVal('metric-contrast-val', cm.rms_contrast);
                setVal('metric-clipping-val', `${(cm.clipping_ratio * 100).toFixed(2)}%`);
            }
        }

        // If screening is BLOCKED by IQA gatekeeper
        if (!screeningAllowed) {
            if (blockedCard) blockedCard.style.display = 'block';
            if (gradeCard) gradeCard.style.display = 'none';
            if (dlCard) dlCard.style.display = 'none';
            if (gradcamCard) gradcamCard.style.display = 'none';
            if (shapCard) shapCard.style.display = 'none';
            otherCards.forEach(c => { c.style.display = 'none'; });
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
            return;
        }

        // Screening is ALLOWED: reveal diagnostic cards
        if (blockedCard) blockedCard.style.display = 'none';
        if (gradeCard) gradeCard.style.display = 'flex';
        if (dlCard) dlCard.style.display = 'block';
        if (gradcamCard) gradcamCard.style.display = 'block';
        if (shapCard) shapCard.style.display = 'block';
        otherCards.forEach(c => { c.style.display = 'block'; });

        const g = result ? result.grading : { icdrGrade: 0, gradeLabel: 'No DR', confidence: 0.9, referable: false };
        const qc = result ? result.qc : {};
        const gradeColors = ['#22c55e', '#84cc16', '#f59e0b', '#f97316', '#ef4444'];
        
        // Use deep learning prediction if available from backend, else client engine
        const finalGrade = dlResult ? dlResult.prediction : g.icdrGrade;
        const finalLabel = dlResult ? dlResult.severity : g.gradeLabel;
        const finalConfidence = dlResult ? dlResult.confidence : g.confidence;
        const finalReferable = dlResult ? dlResult.referable : g.referable;
        const gc = gradeColors[finalGrade] || '#22c55e';

        // Grade card
        const gradeBadge = document.getElementById('grade-badge');
        gradeBadge.style.background = gc;
        gradeBadge.style.boxShadow = `0 0 30px ${gc}44`;
        document.getElementById('grade-number').textContent = finalGrade;
        document.getElementById('grade-label').textContent = finalLabel;
        document.getElementById('grade-label').style.color = gc;

        const referTag = document.getElementById('referable-tag');
        referTag.textContent = finalReferable ? 'REFERABLE — Refer to Ophthalmologist' : 'Not Referable';
        referTag.className = finalReferable ? 'tag referable' : 'tag not-referable';

        const reviewTag = document.getElementById('review-tag');
        if (g.needsManualReview || finalGrade >= 3) {
            reviewTag.style.display = 'inline-block';
            reviewTag.textContent = 'Priority Review Needed';
            reviewTag.className = 'tag review-needed';
        } else {
            reviewTag.style.display = 'none';
        }

        document.getElementById('confidence-fill').style.width = `${finalConfidence * 100}%`;
        document.getElementById('confidence-value').textContent = `${(finalConfidence * 100).toFixed(0)}%`;

        // Render Deep Learning Probabilities & Recommendations
        const dlBadge = document.getElementById('dl-engine-badge');
        const recText = document.getElementById('screening-recommendation-text');
        
        let probs = { no_dr: 0.85, mild: 0.10, moderate: 0.03, severe: 0.01, proliferative: 0.01 };
        let recommendation = "Routine annual screening recommended. Maintain glycemic control.";
        
        if (dlResult && dlResult.probabilities) {
            probs = dlResult.probabilities;
            recommendation = dlResult.screening_recommendation || recommendation;
            if (dlBadge) dlBadge.textContent = 'EfficientNet-B0 (PyTorch Server Active)';
        } else {
            // Estimate distribution from client grade if backend offline
            probs = { no_dr: 0.02, mild: 0.03, moderate: 0.02, severe: 0.01, proliferative: 0.01 };
            const keys = ['no_dr', 'mild', 'moderate', 'severe', 'proliferative'];
            probs[keys[finalGrade]] = finalConfidence;
            const remainder = Math.max(0, 1.0 - finalConfidence);
            keys.forEach((k, idx) => {
                if (idx !== finalGrade) probs[k] = remainder / 4;
            });
            if (dlBadge) dlBadge.textContent = 'Client Rule Engine (Offline Fallback)';
        }

        // Update probability rows
        const probMapping = [
            { id: 0, val: probs.no_dr || 0 },
            { id: 1, val: probs.mild || 0 },
            { id: 2, val: probs.moderate || 0 },
            { id: 3, val: probs.severe || 0 },
            { id: 4, val: probs.proliferative || 0 }
        ];

        probMapping.forEach(item => {
            const bar = document.getElementById(`prob-bar-${item.id}`);
            const text = document.getElementById(`prob-val-${item.id}`);
            const pct = (item.val * 100).toFixed(1);
            if (bar) bar.style.width = `${pct}%`;
            if (text) text.textContent = `${pct}%`;
        });

        if (recText) recText.textContent = recommendation;

        // Populate Deterministic Screening Recommendation Card
        const referralCard = document.getElementById('screening-recommendation-card');
        if (referralCard) {
            let refData = referralResult;
            if (!refData) {
                // Client-side fallback rule engine
                const grade = finalGrade;
                const conf = finalConfidence;
                let risk = "Low Risk";
                let priority = "Routine";
                let action = "Routine follow-up";
                let ruleStr = "DR Grade 0 → Low Risk";
                let why = "";
                if (grade === 0) {
                    risk = "Low Risk"; priority = "Routine"; action = "Routine follow-up"; ruleStr = "DR Grade 0 → Low Risk";
                } else if (grade === 1 || grade === 2) {
                    risk = "Moderate Risk"; priority = "Recommended"; action = "Eye examination recommended"; ruleStr = "DR Grade 1–2 → Moderate Risk";
                } else {
                    risk = "High Risk"; priority = "Priority"; action = "Priority ophthalmologist referral"; ruleStr = "DR Grade 3–4 → High Risk";
                }
                const isLowConf = conf < 0.50;
                if (isLowConf) {
                    if (grade >= 3) {
                        why = `Why this recommendation?\n\nDetected DR grade: ${finalLabel} (Grade ${grade})\nDecision rule: DR Grade 3–4 → High Risk\nRecommendation: Priority ophthalmologist referral.\n\nPrediction confidence is low (${(conf * 100).toFixed(1)}%); professional review is especially important. High-severity screening signal is not downgraded.`;
                    } else {
                        action = "Professional review recommended";
                        why = `Why this recommendation?\n\nDetected DR grade: ${finalLabel} (Grade ${grade})\nDecision rule: Low-confidence prediction (${(conf * 100).toFixed(1)}% < 50% threshold).\nRecommendation: High-priority professional review recommended because the model prediction has low confidence.`;
                    }
                } else {
                    why = `Why this recommendation?\n\nDetected DR grade: ${finalLabel} (Grade ${grade})\nDecision rule: ${ruleStr}\nRecommendation: ${action}.`;
                }
                refData = {
                    dr_grade: grade,
                    dr_label: finalLabel,
                    dr_confidence: conf,
                    risk_level: risk,
                    referral_priority: priority,
                    recommendation: action,
                    decision_status: isLowConf ? "LOW_CONFIDENCE" : "VALID",
                    why_recommendation: why
                };
            }

            const riskColors = {
                "Low Risk": "#10b981",
                "Moderate Risk": "#f59e0b",
                "High Risk": "#ef4444"
            };
            const rColor = riskColors[refData.risk_level] || "#06d6a0";
            referralCard.style.borderLeft = `4px solid ${rColor}`;

            const elGrade = document.getElementById('referral-dr-grade');
            const elConf = document.getElementById('referral-confidence');
            const elRisk = document.getElementById('referral-risk-level');
            const elPrio = document.getElementById('referral-priority-tag');
            const elAct = document.getElementById('referral-action-text');
            const elWhy = document.getElementById('referral-why-text');
            const elBadge = document.getElementById('referral-status-badge');

            if (elGrade) elGrade.textContent = `${refData.dr_label || finalLabel} (Grade ${refData.dr_grade !== undefined ? refData.dr_grade : finalGrade})`;
            if (elConf) elConf.textContent = `Confidence: ${((refData.dr_confidence !== undefined ? refData.dr_confidence : finalConfidence) * 100).toFixed(1)}%`;
            if (elRisk) {
                elRisk.textContent = refData.risk_level || '—';
                elRisk.style.color = rColor;
            }
            if (elPrio) elPrio.textContent = `Priority: ${refData.referral_priority || '—'}`;
            if (elAct) elAct.textContent = refData.recommendation || '—';
            if (elWhy) elWhy.textContent = refData.why_recommendation || 'Recommendation generated based on validated DR grade.';
            if (elBadge) {
                if (refData.decision_status === 'LOW_CONFIDENCE') {
                    elBadge.textContent = 'LOW CONFIDENCE: REVIEW ADVISED';
                    elBadge.style.background = 'rgba(245,158,11,0.15)';
                    elBadge.style.color = '#f59e0b';
                    elBadge.style.border = '1px solid rgba(245,158,11,0.3)';
                } else {
                    elBadge.textContent = 'VALID DECISION';
                    elBadge.style.background = 'rgba(16,185,129,0.15)';
                    elBadge.style.color = '#10b981';
                    elBadge.style.border = '1px solid rgba(16,185,129,0.3)';
                }
            }
        }

        // QC rings
        const qcMetrics = [
            { id: 'overall', val: qc.score },
            { id: 'sharp', val: qc.sharpness },
            { id: 'illum', val: qc.illumination },
            { id: 'fov', val: qc.fov },
            { id: 'contrast', val: qc.contrast }
        ];
        for (const m of qcMetrics) {
            const ring = document.getElementById(`qc-ring-${m.id}`);
            const pct = Math.round(m.val * 100);
            ring.style.setProperty('--pct', `${pct}%`);
            // Set ring color based on value
            const ringColor = m.val >= 0.7 ? '#22c55e' : m.val >= 0.4 ? '#f59e0b' : '#ef4444';
            ring.style.background = `conic-gradient(${ringColor} 0%, ${ringColor} ${pct}%, rgba(255,255,255,0.06) ${pct}%, rgba(255,255,255,0.06) 100%)`;
            document.getElementById(`qc-val-${m.id}`).textContent = pct;
        }

        const qcMsg = document.getElementById('qc-message');
        if (qc.score >= 0.75) {
            qcMsg.textContent = '✓ Good quality — no enhancement needed';
            qcMsg.className = 'qc-message pass';
        } else if (qc.score >= 0.35) {
            qcMsg.textContent = '⚠ Borderline quality — adaptive enhancement applied';
            qcMsg.className = 'qc-message borderline';
        } else {
            qcMsg.textContent = '✕ Poor quality — image may need recapture';
            qcMsg.className = 'qc-message fail';
        }

        // Lesion table
        const tbody = document.getElementById('lesion-tbody');
        const rs = g.ruleScore;
        const lesionRows = [
            ['Microaneurysms', rs.maCount, rs.maCount === 0 ? 'none' : rs.maCount < 5 ? 'mild' : 'moderate'],
            ['Dot Hemorrhages', rs.dotCount, rs.dotCount === 0 ? 'none' : 'moderate'],
            ['Blot Hemorrhages', rs.blotCount, rs.blotCount === 0 ? 'none' : rs.blotCount < 5 ? 'moderate' : 'severe'],
            ['Exudate Area (% FOV)', `${(rs.exudateFrac * 100).toFixed(3)}%`, rs.exudateFrac < 0.002 ? 'none' : 'moderate'],
            ['Neovascularization', rs.nvSuspected ? 'SUSPECTED' : 'Not detected', rs.nvSuspected ? 'severe' : 'none']
        ];

        const severityColors = { none: '#22c55e', mild: '#84cc16', moderate: '#f59e0b', severe: '#ef4444' };

        tbody.innerHTML = lesionRows.map(([name, value, severity]) => `
            <tr>
                <td>${name}</td>
                <td>${value}</td>
                <td><span class="severity-dot" style="background:${severityColors[severity]}"></span>${severity}</td>
            </tr>
        `).join('');

        // Render image panels
        pipeline.renderOriginal(document.getElementById('result-original-canvas'));
        pipeline.renderVessels(document.getElementById('result-vessel-canvas'), result.vessels.mask);
        pipeline.renderLesions(document.getElementById('result-lesion-canvas'), result.lesionData, result.structures.disc);
        pipeline.renderHeatmap(document.getElementById('result-heatmap-canvas'), result.heatmap);

        // Report
        document.getElementById('report-content').textContent = result.reportText;

        // Download report
        document.getElementById('btn-download-report').onclick = () => {
            downloadReport(result);
        };

        // Update Combined Patient Screening Summary fields
        const sumDR = document.getElementById('summary-dr-grade');
        const sumConf = document.getElementById('summary-dr-conf');
        const sumRef = document.getElementById('summary-referral-text');

        if (sumDR) sumDR.textContent = finalLabel;
        if (sumConf) sumConf.textContent = `${(finalConfidence * 100).toFixed(1)}%`;
        if (sumRef && window.currentReferralResult) {
            sumRef.innerHTML = `<strong>${window.currentReferralResult.risk_level}:</strong> ${window.currentReferralResult.recommendation}. <br><span style="color:#94a3b8; font-size:0.8rem;">${window.currentReferralResult.why_recommendation.replace(/\\n/g, ' ')}</span>`;
        } else if (sumRef) {
            sumRef.textContent = 'Routine follow-up recommended based on screening rules.';
        }

        updateHealthReportSummaryCards();

        // Trigger Grad-CAM Visual Explanation from PyTorch Backend if screening is allowed
        if (currentFile && (!window.currentGateResult || window.currentGateResult.screening_allowed)) {
            fetchGradcamExplanation(currentFile);
        }

        // Auto-save screening record to Patient Portal
        if (screeningAllowed) {
            handlePatientScreeningAutoSave(result, dlResult, iqaResult, referralResult, finalGrade, finalLabel, finalConfidence, finalReferable);
        }

        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ===================== Patient Portal Screening Persistence =====================
    window.currentSavedScreeningId = null;

    async function handlePatientScreeningAutoSave(result, dlResult, iqaResult, referralResult, finalGrade, finalLabel, finalConfidence, finalReferable) {
        const resultsSection = document.getElementById('results-section');
        if (!resultsSection) return;

        // Clean up previous banner if present
        const oldBanner = document.getElementById('patient-portal-banner');
        if (oldBanner) oldBanner.remove();

        const currentUser = window.PatientAuth ? window.PatientAuth.getCurrentUser() : null;

        if (!currentUser) {
            // Render non-blocking prompt for unauthenticated patients
            const guestBanner = document.createElement('div');
            guestBanner.id = 'patient-portal-banner';
            guestBanner.innerHTML = `
                <div style="background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(99, 102, 241, 0.05)); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 14px 20px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                    <div style="display:flex; align-items:center; gap: 12px;">
                        <span style="font-size: 1.4rem;">👤</span>
                        <div>
                            <div style="font-weight: 600; color: #e2e8f0; font-size: 0.95rem;">Are you a patient undergoing screening?</div>
                            <div style="font-size: 0.8rem; color: #94a3b8;">Sign in or create an account to save this screening directly to your longitudinal medical record and progression history.</div>
                        </div>
                    </div>
                    <div style="display:flex; gap: 8px;">
                        <a href="/patient/login" target="_top" style="display:inline-flex; align-items:center; gap:6px; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); color: #38bdf8; padding: 7px 14px; border-radius: 8px; font-size: 0.825rem; font-weight: 600; text-decoration: none;">
                            Patient Sign In
                        </a>
                        <a href="/patient/signup" target="_top" style="display:inline-flex; align-items:center; gap:6px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); color: #34d399; padding: 7px 14px; border-radius: 8px; font-size: 0.825rem; font-weight: 600; text-decoration: none;">
                            Register Account
                        </a>
                    </div>
                </div>
            `;
            resultsSection.insertBefore(guestBanner, resultsSection.firstChild);
            return;
        }

        // Export fundus scan canvas if available
        let fundusDataUrl = null;
        try {
            const previewCanvas = document.getElementById('preview-canvas');
            if (previewCanvas && previewCanvas.width > 0) {
                fundusDataUrl = previewCanvas.toDataURL('image/jpeg', 0.8);
            }
        } catch (e) {
            console.warn('Fundus canvas export skipped:', e);
        }

        // Extract detected lesions from pipeline
        const rs = (result && result.grading && result.grading.ruleScore) || {};
        const detectedLesions = {
            microaneurysm: (rs.maCount > 0) || (finalGrade >= 1),
            hemorrhage: (rs.dotCount > 0 || rs.blotCount > 0) || (finalGrade >= 2),
            hard_exudate: (rs.exudateFrac > 0.001) || (finalGrade >= 2),
            soft_exudate: finalGrade >= 3
        };

        const ref = referralResult || window.currentReferralResult || {};

        const screeningPayload = {
            imageUrl: fundusDataUrl || 'fundus_scan.jpg',
            imageQuality: iqaResult ? (iqaResult.predicted_class || iqaResult.quality_status || 'Accepted') : 'Accepted',
            modelUsed: dlResult ? 'EfficientNet-B0 (PyTorch)' : 'Client-Side Rule Pipeline',
            drClass: finalGrade !== undefined ? finalGrade : 0,
            drLabel: finalLabel || 'No Diabetic Retinopathy',
            confidence: typeof finalConfidence === 'number' ? finalConfidence : 0.95,
            riskLevel: ref.risk_level || (finalGrade >= 3 ? 'High Risk' : (finalGrade >= 1 ? 'Moderate Risk' : 'Low Risk')),
            recommendation: ref.recommendation || (finalGrade >= 3 ? 'Priority ophthalmologist referral required within 24-48 hours' : (finalGrade >= 1 ? 'Comprehensive eye examination recommended within 2-4 weeks' : 'Routine follow-up in 12 months')),
            gradCamUrl: window.currentGradcamData?.overlay || null,
            lesionMaskUrl: window.currentGradcamData?.unet_findings?.overlay || null,
            detectedLesions: (window.currentGradcamData?.unet_findings?.detections) || detectedLesions
        };

        try {
            const saved = await window.PatientScreening.saveScreening(screeningPayload);
            if (saved) {
                window.currentSavedScreeningId = saved.screeningId;
                const patientBanner = document.createElement('div');
                patientBanner.id = 'patient-portal-banner';
                patientBanner.innerHTML = `
                    <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(56, 189, 248, 0.08)); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 14px 20px; margin-bottom: 1.5rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                        <div style="display:flex; align-items:center; gap: 12px;">
                            <span style="font-size: 1.4rem;">📋</span>
                            <div>
                                <div style="font-weight: 600; color: #10b981; font-size: 0.95rem;">Screening Saved to Patient Medical Record</div>
                                <div style="font-size: 0.8rem; color: #94a3b8;">
                                    Patient: <span style="color:#e2e8f0; font-weight:600;">${currentUser.fullName || currentUser.email}</span> 
                                    (<span style="color:#38bdf8;">${currentUser.patientId || 'ID'}</span>) &bull; Record ID: <span style="font-family:monospace; color:#cbd5e1;">${saved.screeningId}</span>
                                </div>
                            </div>
                        </div>
                        <div style="display:flex; gap: 8px;">
                            <a href="/patient/screening/${saved.screeningId}" target="_top" style="display:inline-flex; align-items:center; gap:6px; background: rgba(56, 189, 248, 0.2); border: 1px solid rgba(56, 189, 248, 0.4); color: #38bdf8; padding: 7px 14px; border-radius: 8px; font-size: 0.825rem; font-weight: 600; text-decoration: none;">
                                View Report Details
                            </a>
                            <a href="/patient/dashboard" target="_top" style="display:inline-flex; align-items:center; gap:6px; background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.4); color: #34d399; padding: 7px 14px; border-radius: 8px; font-size: 0.825rem; font-weight: 600; text-decoration: none;">
                                Patient Dashboard →
                            </a>
                        </div>
                    </div>
                `;
                resultsSection.insertBefore(patientBanner, resultsSection.firstChild);
            }
        } catch (err) {
            console.error('Failed to auto-save screening to patient portal:', err);
        }
    }

    // ===================== Clinical Override Handler =====================
    window.overrideIqaAndScreen = async function() {
        if (window.currentGateResult) {
            window.currentGateResult.screening_allowed = true;
        }
        const blockedCard = document.getElementById('screening-blocked-card');
        const gradeCard = document.getElementById('grade-card');
        const dlCard = document.getElementById('efficientnet-dl-card');
        const gradcamCard = document.getElementById('gradcam-explainability-card');
        const shapCard = document.getElementById('shap-explainability-card');
        const otherCards = document.querySelectorAll('.results-section > .glass-card:not(#iqa-card):not(#screening-blocked-card)');

        if (blockedCard) blockedCard.style.display = 'none';
        if (gradeCard) gradeCard.style.display = 'flex';
        if (dlCard) dlCard.style.display = 'block';
        if (gradcamCard) gradcamCard.style.display = 'block';
        if (shapCard) shapCard.style.display = 'block';
        otherCards.forEach(c => { c.style.display = 'block'; });

        const statusBadge = document.getElementById('iqa-status-badge');
        if (statusBadge) {
            statusBadge.textContent = 'OVERRIDDEN BY OPERATOR: SCREENING UNLOCKED';
            statusBadge.style.background = 'rgba(245,158,11,0.2)';
            statusBadge.style.color = '#f59e0b';
            statusBadge.style.border = '1px solid rgba(245,158,11,0.4)';
        }

        // If deep learning prediction was not yet loaded, fetch it now via /api/predict
        if (currentFile && !window.currentDlResult) {
            try {
                const formData = new FormData();
                formData.append('image', currentFile);
                const resp = await fetch('/api/predict', { method: 'POST', body: formData });
                if (resp.ok) {
                    const dlData = await resp.json();
                    window.currentDlResult = dlData;
                    renderResults(window.currentPipelineResult, dlData, window.currentIqaResult, { screening_allowed: true });
                }
            } catch (e) {
                console.warn('Clinical override DR fetch failed:', e);
            }
        }

        if (currentFile) {
            fetchGradcamExplanation(currentFile);
        }
    };

    // ===================== Grad-CAM Explainability =====================
    window.currentGradcamData = null;
    let currentGradcamView = 'overlay';

    async function fetchGradcamExplanation(imageFileOrBlob, targetClass = null) {
        if (!imageFileOrBlob) return;
        const card = document.getElementById('gradcam-explainability-card');
        if (card) card.style.opacity = '0.7';

        try {
            const formData = new FormData();
            formData.append('image', imageFileOrBlob);
            let url = '/api/gradcam?with_unet=true';
            if (targetClass !== null && targetClass !== undefined) {
                url += `&target_class=${targetClass}`;
            }
            const slider = document.getElementById('gradcam-opacity-slider');
            const alphaVal = slider ? (parseFloat(slider.value) / 100).toFixed(2) : '0.40';
            url += `&alpha=${alphaVal}`;

            const resp = await fetch(url, { method: 'POST', body: formData });
            if (resp.ok) {
                const data = await resp.json();
                window.currentGradcamData = data;
                renderGradcamExplanation(data);

                // Asynchronously augment saved screening record with Grad-CAM & U-Net overlays
                if (window.currentSavedScreeningId && window.PatientScreening) {
                    window.PatientScreening.updateScreeningExplanations(window.currentSavedScreeningId, {
                        gradCamUrl: data.overlay || null,
                        lesionMaskUrl: data.unet_findings?.overlay || null,
                        detectedLesions: data.unet_findings?.detections || null
                    }).catch(e => console.warn('Screening overlay update error:', e));
                }
            }
        } catch (err) {
            console.warn('Grad-CAM explanation fetch failed:', err);
        } finally {
            if (card) card.style.opacity = '1.0';
        }
    }

    function renderGradcamExplanation(data) {
        if (!data) return;

        // Base & Overlay Images
        const baseImg = document.getElementById('gradcam-img-base');
        const overlayImg = document.getElementById('gradcam-img-overlay');
        const sideOrig = document.getElementById('gradcam-side-orig');
        const sideOver = document.getElementById('gradcam-side-over');
        const sideUnet = document.getElementById('gradcam-side-unet');

        if (sideOrig && data.original) sideOrig.src = data.original;
        if (sideOver && data.overlay) sideOver.src = data.overlay;
        if (sideUnet && data.unet_findings?.overlay) sideUnet.src = data.unet_findings.overlay;

        // Apply current view
        switchGradcamView(currentGradcamView);

        // Target badge & explanation
        const tgtBadge = document.getElementById('gradcam-target-class-badge');
        if (tgtBadge) tgtBadge.textContent = `Target: ${data.target_severity || 'Auto'}`;

        const explText = document.getElementById('gradcam-explanation-text');
        if (explText) explText.textContent = data.explanation || 'Highlighted regions indicate areas that contributed strongly to the model prediction.';

        const sanityStats = document.getElementById('gradcam-sanity-stats');
        if (sanityStats && data.sanity_stats) {
            sanityStats.textContent = `Heatmap Min: ${data.sanity_stats.min} | Max: ${data.sanity_stats.max} | Mean: ${data.sanity_stats.mean} | Shape: [${data.sanity_stats.dimensions.join('×')}]`;
        }

        // U-Net tags
        if (data.unet_findings?.detections) {
            const d = data.unet_findings.detections;
            const setTag = (id, name, detected) => {
                const el = document.getElementById(id);
                if (el) {
                    el.textContent = `${name}: ${detected ? 'Detected' : 'None'}`;
                    el.style.background = detected ? 'rgba(239,68,68,0.2)' : 'rgba(34,197,94,0.15)';
                    el.style.color = detected ? '#f87171' : '#4ade80';
                    el.style.border = `1px solid ${detected ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.3)'}`;
                }
            };
            setTag('unet-tag-ma', 'Microaneurysms', d.microaneurysm);
            setTag('unet-tag-he', 'Hemorrhages', d.hemorrhage);
            setTag('unet-tag-ex', 'Hard Exudates', d.hard_exudate);
            setTag('unet-tag-se', 'Soft Exudates', d.soft_exudate);
        }
    }

    window.switchGradcamView = function(viewName) {
        currentGradcamView = viewName;
        const data = window.currentGradcamData;
        if (!data) return;

        // Update tabs active state
        document.querySelectorAll('.btn-gradcam-tab').forEach(b => b.classList.remove('active'));
        const activeBtn = document.getElementById(`tab-view-${viewName}`);
        if (activeBtn) activeBtn.classList.add('active');

        const singleView = document.getElementById('gradcam-single-view');
        const combinedView = document.getElementById('gradcam-combined-view');
        const baseImg = document.getElementById('gradcam-img-base');
        const overlayImg = document.getElementById('gradcam-img-overlay');

        if (viewName === 'combined') {
            if (singleView) singleView.style.display = 'none';
            if (combinedView) combinedView.style.display = 'flex';
        } else {
            if (singleView) singleView.style.display = 'flex';
            if (combinedView) combinedView.style.display = 'none';

            if (viewName === 'overlay') {
                if (baseImg) baseImg.src = data.original;
                if (overlayImg) {
                    overlayImg.src = data.overlay;
                    overlayImg.style.display = 'block';
                }
            } else if (viewName === 'heatmap') {
                if (baseImg) baseImg.src = data.heatmap;
                if (overlayImg) overlayImg.style.display = 'none';
            } else if (viewName === 'original') {
                if (baseImg) baseImg.src = data.original;
                if (overlayImg) overlayImg.style.display = 'none';
            } else if (viewName === 'unet') {
                if (baseImg) baseImg.src = data.unet_findings?.overlay || data.original;
                if (overlayImg) overlayImg.style.display = 'none';
            }
        }
    };

    window.updateGradcamOpacity = function(val) {
        const overlayImg = document.getElementById('gradcam-img-overlay');
        const valText = document.getElementById('gradcam-opacity-val');
        const alpha = (val / 100).toFixed(2);
        if (overlayImg) overlayImg.style.opacity = alpha;
        if (valText) valText.textContent = `${val}%`;
    };

    window.onGradcamTargetClassChange = function() {
        if (window.currentGateResult && !window.currentGateResult.screening_allowed) {
            return;
        }
        const sel = document.getElementById('gradcam-target-select');
        const val = sel.value === 'auto' ? null : parseInt(sel.value);
        if (currentFile) {
            fetchGradcamExplanation(currentFile, val);
        }
    };

    // ===================== SHAP Feature Explainability =====================
    window.currentShapData = null;
    let currentShapView = 'overlay';

    window.triggerShapExplanation = async function() {
        if (!currentFile) {
            alert('Please select or upload a fundus photograph first.');
            return;
        }
        if (window.currentGateResult && !window.currentGateResult.screening_allowed) {
            alert('DR classification and SHAP explainability are blocked: image quality is insufficient for screening. Recapture required.');
            return;
        }
        const sel = document.getElementById('shap-target-select');
        const targetClass = sel && sel.value !== 'auto' ? parseInt(sel.value) : null;
        await fetchShapExplanation(currentFile, targetClass);
    };

    async function fetchShapExplanation(imageFileOrBlob, targetClass = null) {
        if (!imageFileOrBlob) return;
        const spinner = document.getElementById('btn-shap-spinner');
        const btnText = document.getElementById('btn-shap-text');
        const btnRun = document.getElementById('btn-run-shap');
        const contentArea = document.getElementById('shap-content-area');

        if (spinner) spinner.style.display = 'inline-block';
        if (btnText) btnText.textContent = 'Computing SHAP Values (10-15s)...';
        if (btnRun) btnRun.disabled = true;

        try {
            const formData = new FormData();
            formData.append('image', imageFileOrBlob);
            let url = '/api/explain/shap?with_unet=true&nsamples=30';
            if (targetClass !== null && targetClass !== undefined) {
                url += `&target_class=${targetClass}`;
            }

            const resp = await fetch(url, { method: 'POST', body: formData });
            if (resp.ok) {
                const data = await resp.json();
                window.currentShapData = data;
                if (contentArea) contentArea.style.display = 'block';
                renderShapExplanation(data);
            } else {
                const err = await resp.json();
                alert(`SHAP computation error: ${err.error || 'Server error'}`);
            }
        } catch (err) {
            console.warn('SHAP explanation fetch failed:', err);
            alert(`SHAP execution request failed: ${err.message}`);
        } finally {
            if (spinner) spinner.style.display = 'none';
            if (btnText) btnText.textContent = '🔄 Re-calculate SHAP';
            if (btnRun) btnRun.disabled = false;
        }
    }

    function renderShapExplanation(data) {
        if (!data || !data.visualizations) return;

        // Target badge & narratives
        const tgtBadge = document.getElementById('shap-target-badge');
        if (tgtBadge) tgtBadge.textContent = `Target: ${data.target_label || 'Auto'}`;

        const posNarrative = document.getElementById('shap-pos-narrative');
        if (posNarrative) posNarrative.textContent = data.explanation?.positive_regions || 'Warm regions pushed toward prediction.';

        const negNarrative = document.getElementById('shap-neg-narrative');
        if (negNarrative) negNarrative.textContent = data.explanation?.negative_regions || 'Cool regions opposed prediction.';

        const statsBar = document.getElementById('shap-stats-bar');
        if (statsBar && data.explanation?.statistics) {
            const s = data.explanation.statistics;
            statsBar.textContent = `Sparsity: ${(s.sparsity * 100).toFixed(1)}% | Min: ${s.min.toFixed(4)} | Max: ${s.max.toFixed(4)} | Time: ${s.computation_time_seconds}s`;
        }

        // Multi-modal side panels
        const sideOrig = document.getElementById('shap-side-orig');
        const sideGradcam = document.getElementById('shap-side-gradcam');
        const sideShap = document.getElementById('shap-side-shap');
        const sideUnet = document.getElementById('shap-side-unet');

        if (sideOrig) sideOrig.src = data.visualizations.original;
        if (sideGradcam && window.currentGradcamData?.overlay) sideGradcam.src = window.currentGradcamData.overlay;
        else if (sideGradcam) sideGradcam.src = data.visualizations.overlay;
        if (sideShap) sideShap.src = data.visualizations.overlay;
        if (sideUnet && data.unet_findings?.overlay) sideUnet.src = data.unet_findings.overlay;
        else if (sideUnet && window.currentGradcamData?.unet_findings?.overlay) sideUnet.src = window.currentGradcamData.unet_findings.overlay;

        // Lesion-level regional stats
        if (data.lesion_contributions) {
            const lc = data.lesion_contributions;
            const setLesion = (id, name, item) => {
                const el = document.getElementById(id);
                if (el) {
                    if (item && item.available) {
                        el.textContent = `• ${name}: Mean = ${item.mean_contribution > 0 ? '+' : ''}${item.mean_contribution.toFixed(6)} (${item.status})`;
                        el.style.color = item.mean_contribution > 0 ? '#f87171' : '#60a5fa';
                    } else {
                        el.textContent = `• ${name}: Not available (no segmented lesions)`;
                        el.style.color = '#94a3b8';
                    }
                }
            };
            setLesion('shap-lesion-ma', 'Microaneurysms', lc.microaneurysm);
            setLesion('shap-lesion-he', 'Hemorrhages', lc.hemorrhage);
            setLesion('shap-lesion-ex', 'Hard Exudates', lc.hard_exudate);
            setLesion('shap-lesion-se', 'Soft Exudates', lc.soft_exudate);
        }

        const overlapBadge = document.getElementById('shap-unet-overlap-badge');
        if (overlapBadge && data.shap_unet_overlap?.overlap_available) {
            overlapBadge.textContent = `SHAP/U-Net Overlap (Dice): ${data.shap_unet_overlap.overlap_percentage}% (Spatial intersection metric)`;
            overlapBadge.style.color = '#34d399';
        }

        switchShapView(currentShapView);
    }

    window.switchShapView = function(viewName) {
        currentShapView = viewName;
        const data = window.currentShapData;
        if (!data || !data.visualizations) return;

        document.querySelectorAll('.btn-shap-tab').forEach(b => b.classList.remove('active'));
        const activeBtn = document.getElementById(`tab-shap-${viewName}`);
        if (activeBtn) activeBtn.classList.add('active');

        const singleView = document.getElementById('shap-single-view');
        const sidebysideView = document.getElementById('shap-sidebyside-view');
        const baseImg = document.getElementById('shap-img-base');
        const overlayImg = document.getElementById('shap-img-overlay');

        if (viewName === 'sidebyside') {
            if (singleView) singleView.style.display = 'none';
            if (sidebysideView) sidebysideView.style.display = 'flex';
        } else {
            if (singleView) singleView.style.display = 'flex';
            if (sidebysideView) sidebysideView.style.display = 'none';

            if (viewName === 'overlay') {
                if (baseImg) baseImg.src = data.visualizations.original;
                if (overlayImg) {
                    overlayImg.src = data.visualizations.overlay;
                    overlayImg.style.display = 'block';
                }
            } else if (viewName === 'positive') {
                if (baseImg) baseImg.src = data.visualizations.positive;
                if (overlayImg) overlayImg.style.display = 'none';
            } else if (viewName === 'negative') {
                if (baseImg) baseImg.src = data.visualizations.negative;
                if (overlayImg) overlayImg.style.display = 'none';
            } else if (viewName === 'combined') {
                if (baseImg) baseImg.src = data.visualizations.combined;
                if (overlayImg) overlayImg.style.display = 'none';
            }
        }
    };

    window.updateShapOpacity = function(val) {
        const overlayImg = document.getElementById('shap-img-overlay');
        const valText = document.getElementById('shap-opacity-val');
        const alpha = (val / 100).toFixed(2);
        if (overlayImg) overlayImg.style.opacity = alpha;
        if (valText) valText.textContent = `${val}%`;
    };

    window.onShapTargetClassChange = function() {
        const sel = document.getElementById('shap-target-select');
        const val = sel.value === 'auto' ? null : parseInt(sel.value);
        if (currentFile) {
            fetchShapExplanation(currentFile, val);
        }
    };

    // ===================== Report Download =====================
    function downloadReport(result) {
        const canvas = document.createElement('canvas');
        canvas.width = 1200;
        canvas.height = 800;
        const ctx = canvas.getContext('2d');

        // Background
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, 1200, 800);

        // Header bar
        const gc = ['#22c55e', '#84cc16', '#f59e0b', '#f97316', '#ef4444'][result.grading.icdrGrade];
        ctx.fillStyle = gc;
        ctx.fillRect(0, 0, 1200, 6);

        // Title
        ctx.fillStyle = '#f1f5f9';
        ctx.font = 'bold 22px Inter, sans-serif';
        ctx.fillText('Diabetic Retinopathy Screening Report', 30, 40);
        ctx.font = '13px Inter, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText('For Ophthalmologist Review — Decision Support Only', 30, 60);

        // Grade box
        ctx.fillStyle = gc;
        ctx.beginPath();
        ctx.roundRect(30, 80, 200, 120, 12);
        ctx.fill();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 48px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`Grade ${result.grading.icdrGrade}`, 130, 135);
        ctx.font = '16px Inter, sans-serif';
        ctx.fillText(result.grading.gradeLabel, 130, 165);
        ctx.textAlign = 'left';

        // Info
        ctx.fillStyle = '#f1f5f9';
        ctx.font = '14px Inter, sans-serif';
        let y = 100;
        ctx.fillText(`Referable: ${result.grading.referable ? 'YES' : 'No'}`, 260, y);
        ctx.fillText(`Confidence: ${(result.grading.confidence * 100).toFixed(0)}%`, 260, y + 24);
        ctx.fillText(`Manual Review: ${result.grading.needsManualReview ? 'FLAGGED' : 'Standard'}`, 260, y + 48);
        ctx.fillText(`Grading Mode: Rule-based (no CNN)`, 260, y + 72);

        // Images (draw from existing canvases)
        const origCanvas = document.getElementById('result-original-canvas');
        const heatCanvas = document.getElementById('result-heatmap-canvas');
        if (origCanvas.width > 0) {
            ctx.drawImage(origCanvas, 30, 220, 280, 280);
            ctx.fillStyle = '#94a3b8';
            ctx.font = '11px Inter, sans-serif';
            ctx.fillText('Original (Enhanced)', 30, 516);
        }
        if (heatCanvas.width > 0) {
            ctx.drawImage(heatCanvas, 330, 220, 280, 280);
            ctx.fillText('Attention Heatmap', 330, 516);
        }

        // Lesion table
        ctx.fillStyle = '#f1f5f9';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.fillText('Lesion Evidence Summary', 650, 235);
        ctx.font = '13px Inter, sans-serif';
        const rs = result.grading.ruleScore;
        const tableData = [
            ['Microaneurysms', rs.maCount],
            ['Dot hemorrhages', rs.dotCount],
            ['Blot hemorrhages', rs.blotCount],
            ['Exudate area', `${(rs.exudateFrac * 100).toFixed(3)}%`],
            ['NV flag', rs.nvSuspected ? 'SUSPECTED' : 'Not detected']
        ];
        y = 260;
        for (const [label, val] of tableData) {
            ctx.fillStyle = '#94a3b8';
            ctx.fillText(label, 660, y);
            ctx.fillStyle = '#6ee7b7';
            ctx.fillText(String(val), 860, y);
            y += 28;
        }

        // QC
        ctx.fillStyle = '#f1f5f9';
        ctx.font = 'bold 14px Inter, sans-serif';
        ctx.fillText('Image Quality', 650, y + 20);
        ctx.font = '13px Inter, sans-serif';
        y += 44;
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`Overall: ${result.qc.score.toFixed(2)}  |  Sharp: ${result.qc.sharpness.toFixed(2)}  |  Illum: ${result.qc.illumination.toFixed(2)}`, 660, y);

        // Timestamp
        ctx.fillStyle = '#64748b';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText(`Generated: ${new Date().toLocaleString()}`, 30, 770);
        ctx.fillText('Not an autonomous diagnostic device. Decision-support only.', 30, 788);

        // Download
        const link = document.createElement('a');
        link.download = 'DR_Screening_Report.png';
        link.href = canvas.toDataURL('image/png');
        link.click();
    }

    // ===================== Capacity Planning =====================
    document.getElementById('btn-run-capacity').addEventListener('click', () => {
        const params = {
            annualPatients: parseFloat(document.getElementById('cap-annual').value),
            screeningDaysPerYear: parseFloat(document.getElementById('cap-days').value),
            hoursPerDay: parseFloat(document.getElementById('cap-hours').value),
            imageSizeMB: parseFloat(document.getElementById('cap-imgsize').value),
            uplinkMbps: parseFloat(document.getElementById('cap-uplink').value),
            aiProcessingSecPerImage: parseFloat(document.getElementById('cap-aitime').value),
            reviewMinPerCase: parseFloat(document.getElementById('cap-reviewtime').value),
            referableFraction: parseFloat(document.getElementById('cap-referable').value),
            targetUtilization: parseFloat(document.getElementById('cap-util').value)
        };

        const planner = new CapacityPlanner(params);
        const results = planner.analyze();
        renderCapacityResults(results);

        document.getElementById('capacity-results').scrollIntoView({ behavior: 'smooth' });
    });

    // ===================== Initialize Docs =====================
    initDocs();
});
