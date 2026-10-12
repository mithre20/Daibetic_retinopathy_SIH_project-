/**
 * AI DR SCREEN — Patient Details & Clinical Intelligence Controller
 * Provides interactive medical dossier, AI Grad-CAM and lesion overlay rendering,
 * longitudinal progression tracking, and care pathway referral management.
 */

(function() {
    'use strict';

    function syncTheme(theme) {
        const active = theme || localStorage.getItem('xairis_theme') || 'light';
        document.documentElement.setAttribute('data-theme', active);
        document.body.setAttribute('data-theme', active);
    }
    syncTheme();
    window.addEventListener('message', (e) => {
        if (e.data && e.data.action === 'themeChange') syncTheme(e.data.theme);
    });

    // Comprehensive Patient Database
    const PATIENT_DATABASE = {
        'DR25072026': {
            id: 'DR25072026',
            abhaId: '91-4458-3921-0029',
            name: 'Ramesh Kumar',
            ageGender: '58 / Male',
            dob: '14-Aug-1968',
            phone: '+91 94432 87102',
            village: 'Thirunallur, Tamil Nadu',
            subDistrict: 'Gingee Block, Villupuram',
            phc: 'Primary Health Centre — Zone 4',
            ashaWorker: 'Kamala Devi (ID: ASHA-TN-402)',
            duration: '8 Years',
            dmType: 'Type 2 Diabetes Mellitus',
            severity: 'Moderate',
            severityClass: 'moderate',
            icdrText: 'Moderate NPDR (Grade 2)',
            confidence: 92.4,
            hba1c: '8.6%',
            hba1cStatus: 'danger',
            hba1cLabel: 'Elevated (>8.0%)',
            bp: '138/86 mmHg',
            bpStatus: 'warning',
            bpLabel: 'Pre-Hypertension',
            fbs: '178 mg/dL',
            fbsStatus: 'warning',
            ppbs: '245 mg/dL',
            bmi: '26.4 kg/m²',
            visualAcuityOD: '6/12',
            visualAcuityOS: '6/9',
            medications: [
                { name: 'Metformin Hydrochloride', dose: '1000 mg', freq: 'BID (Twice Daily, Post-meal)' },
                { name: 'Glimepiride', dose: '2 mg', freq: 'OD (Once Daily, Pre-breakfast)' },
                { name: 'Telmisartan', dose: '40 mg', freq: 'OD (Once Daily, Morning)' },
                { name: 'Atorvastatin', dose: '10 mg', freq: 'HS (Bedtime)' }
            ],
            allergies: 'No Known Drug Allergies (NKDA)',
            comorbidities: 'Stage 1 Essential Hypertension, Mild Distal Neuropathy',
            lesions: {
                microaneurysms: 14,
                exudates: 6,
                hemorrhages: 4,
                csmeSuspicion: 'Moderate CSME risk (temporal macular ring)'
            },
            history: [
                { date: '25-Jul-2026', title: 'Current Screening (Zone 4 PHC)', grade: 'Grade 2: Moderate NPDR', gradeClass: 'grade-2', nodeClass: 'danger', hba1c: '8.6%', lesions: '14 MAs, 6 Exudates, 4 Dot Hemorrhages', action: 'Referral generated for District Hospital' },
                { date: '22-Aug-2025', title: 'Annual Surveillance (Thirunallur)', grade: 'Grade 1: Mild NPDR', gradeClass: 'grade-1', nodeClass: 'warning', hba1c: '7.9%', lesions: '3 Microaneurysms, No Exudates', action: 'Dietary counseling, 12-month re-screen' },
                { date: '14-Nov-2024', title: 'Baseline T2DM Screening', grade: 'Grade 0: No Apparent DR', gradeClass: 'grade-0', nodeClass: 'normal', hba1c: '7.4%', lesions: 'Clean retina, clear foveal avascular zone', action: 'Routine annual eye checkup recommended' }
            ],
            referralNotes: 'Patient exhibits rapid escalation from Mild to Moderate NPDR over 11 months accompanied by glycemic deterioration (HbA1c 7.9% → 8.6%). Discrete hard exudate ring in temporal macula indicates early DME. Urgently refer to District Hospital Ophthalmology for dilated stereoscopic biomicroscopy & Macular OCT.'
        },
        'DR25072027': {
            id: 'DR25072027',
            abhaId: '91-3829-1940-0081',
            name: 'Lakshmi Ammal',
            ageGender: '62 / Female',
            dob: '02-Mar-1964',
            phone: '+91 98412 34918',
            village: 'Kottakuppam, Tamil Nadu',
            subDistrict: 'Vanur Block, Villupuram',
            phc: 'Primary Health Centre — Kottakuppam',
            ashaWorker: 'Meena Sundaram (ID: ASHA-TN-219)',
            duration: '12 Years',
            dmType: 'Type 2 Diabetes Mellitus',
            severity: 'Severe',
            severityClass: 'severe',
            icdrText: 'Severe NPDR (Grade 3 — 4-2-1 Rule Met)',
            confidence: 96.2,
            hba1c: '9.4%',
            hba1cStatus: 'danger',
            hba1cLabel: 'Critical (>9.0%)',
            bp: '152/92 mmHg',
            bpStatus: 'danger',
            bpLabel: 'Stage 2 HTN',
            fbs: '210 mg/dL',
            fbsStatus: 'danger',
            ppbs: '298 mg/dL',
            bmi: '28.1 kg/m²',
            visualAcuityOD: '6/18',
            visualAcuityOS: '6/18',
            medications: [
                { name: 'Insulin Glargine (Basal)', dose: '18 Units', freq: 'QPM (Nightly)' },
                { name: 'Metformin', dose: '1000 mg', freq: 'BID' },
                { name: 'Amlodipine', dose: '5 mg', freq: 'OD' }
            ],
            allergies: 'Penicillin (Skin rash)',
            comorbidities: 'Chronic Kidney Disease (Stage 2), Peripheral Arterial Disease',
            lesions: {
                microaneurysms: 28,
                exudates: 16,
                hemorrhages: 12,
                csmeSuspicion: 'High CSME likelihood (Fovea involving)'
            },
            history: [
                { date: '25-Jul-2026', title: 'Current Screening (Kottakuppam)', grade: 'Grade 3: Severe NPDR', gradeClass: 'grade-2', nodeClass: 'danger', hba1c: '9.4%', lesions: 'Extensive intraretinal microvascular abnormalities (IRMA)', action: 'Fast-track tertiary referral within 1 week' },
                { date: '10-Oct-2025', title: 'Routine Review', grade: 'Grade 2: Moderate NPDR', gradeClass: 'grade-1', nodeClass: 'warning', hba1c: '8.8%', lesions: 'Cotton wool spots and venous beading', action: 'Review insulin dosage' }
            ],
            referralNotes: 'Severe NPDR meeting International 4-2-1 criteria. Immediate vitreoretinal evaluation required. High risk of near-term neovascular conversion (PDR).'
        },
        'DR25072028': {
            id: 'DR25072028',
            abhaId: '91-7712-4409-0114',
            name: 'Murugan P.',
            ageGender: '47 / Male',
            dob: '19-May-1979',
            phone: '+91 97890 12093',
            village: 'Pondicherry Rural',
            subDistrict: 'Oulgaret Block',
            phc: 'Rural Community Health Center',
            ashaWorker: 'Revathi S. (ID: ASHA-PY-108)',
            duration: '4 Years',
            dmType: 'Type 2 Diabetes Mellitus',
            severity: 'Mild',
            severityClass: 'mild',
            icdrText: 'Mild NPDR (Grade 1)',
            confidence: 88.7,
            hba1c: '7.3%',
            hba1cStatus: 'normal',
            hba1cLabel: 'Fair Control',
            bp: '124/80 mmHg',
            bpStatus: 'normal',
            bpLabel: 'Optimal',
            fbs: '132 mg/dL',
            fbsStatus: 'normal',
            ppbs: '168 mg/dL',
            bmi: '24.2 kg/m²',
            visualAcuityOD: '6/6',
            visualAcuityOS: '6/6',
            medications: [
                { name: 'Metformin', dose: '500 mg', freq: 'BID (Post-meal)' }
            ],
            allergies: 'None',
            comorbidities: 'None',
            lesions: {
                microaneurysms: 3,
                exudates: 0,
                hemorrhages: 0,
                csmeSuspicion: 'CSME Absent'
            },
            history: [
                { date: '25-Jul-2026', title: 'Current Screening (Pondicherry)', grade: 'Grade 1: Mild NPDR', gradeClass: 'grade-0', nodeClass: 'normal', hba1c: '7.3%', lesions: 'Isolated microaneurysms in temporal periphery', action: 'Annual follow-up, lifestyle adherence' }
            ],
            referralNotes: 'Early microvascular changes consistent with Mild NPDR. Vision is intact (6/6 bilaterally). Re-screening indicated in 12 months with glycemic optimization.'
        }
    };

    // State Variables
    let currentPatientId = 'DR25072026';
    let currentEye = 'OD';
    let activeOverlays = {
        gradcam: true,
        microaneurysms: true,
        exudates: true,
        landmarks: true
    };

    // DOM Elements
    let scanCanvas, scanCtx, scanImg;

    // Initialization
    document.addEventListener('DOMContentLoaded', () => {
        // Read URL query parameter
        const urlParams = new URLSearchParams(window.location.search);
        const queryId = urlParams.get('id');
        if (queryId && PATIENT_DATABASE[queryId]) {
            currentPatientId = queryId;
        }

        scanCanvas = document.getElementById('scan-canvas');
        if (scanCanvas) {
            scanCtx = scanCanvas.getContext('2d');
        }
        scanImg = document.getElementById('scan-img');

        initPatientSwitcher();
        initTabs();
        initEyeToggle();
        initOverlayToggles();
        initNotesAutosave();
        renderPatient(currentPatientId);

        // Resize handler for canvas
        window.addEventListener('resize', () => {
            if (activeOverlays.gradcam || activeOverlays.microaneurysms || activeOverlays.exudates || activeOverlays.landmarks) {
                renderScanOverlays();
            }
        });
    });

    // 1. Patient Switcher
    function initPatientSwitcher() {
        const switcher = document.getElementById('patient-select-dropdown');
        if (switcher) {
            switcher.value = currentPatientId;
            switcher.addEventListener('change', (e) => {
                currentPatientId = e.target.value;
                // Update URL without reload
                const newUrl = new URL(window.location);
                newUrl.searchParams.set('id', currentPatientId);
                window.history.pushState({}, '', newUrl);
                renderPatient(currentPatientId);
                showToast(`Switched patient record to ${PATIENT_DATABASE[currentPatientId].name}`, 'info');
            });
        }
    }

    // 2. Render Full Patient Dossier
    function renderPatient(id) {
        const p = PATIENT_DATABASE[id];
        if (!p) return;

        // Populate Header & Identity
        safeText('p-name-large', p.name);
        safeText('p-id-val', p.id);
        safeText('p-abha-val', p.abhaId);
        safeText('p-age-gender', p.ageGender);
        safeText('p-village', p.village);
        safeText('p-duration', p.duration);
        safeText('p-dm-type', p.dmType);
        safeText('p-dob', p.dob);
        safeText('p-phone', p.phone);
        safeText('p-phc', p.phc);
        safeText('p-asha', p.ashaWorker);
        safeText('p-comorbidities', p.comorbidities);
        safeText('p-allergies', p.allergies);

        // Severity Badges
        const severityBadge = document.getElementById('p-severity-badge');
        if (severityBadge) {
            severityBadge.className = `badge-severity ${p.severityClass}`;
            severityBadge.textContent = p.severity + ' NPDR';
        }
        safeText('p-icdr-text', p.icdrText);
        safeText('p-confidence-text', `${p.confidence}% AI Confidence`);

        // Vitals Cards
        safeText('vital-hba1c', p.hba1c);
        const vitalHba1cStatus = document.getElementById('vital-hba1c-status');
        if (vitalHba1cStatus) {
            vitalHba1cStatus.className = `vital-status ${p.hba1cStatus}`;
            vitalHba1cStatus.textContent = p.hba1cLabel;
        }

        safeText('vital-bp', p.bp);
        const vitalBpStatus = document.getElementById('vital-bp-status');
        if (vitalBpStatus) {
            vitalBpStatus.className = `vital-status ${p.bpStatus}`;
            vitalBpStatus.textContent = p.bpLabel;
        }

        safeText('vital-fbs', p.fbs);
        safeText('vital-ppbs', p.ppbs);
        safeText('vital-bmi', p.bmi);
        safeText('vital-va-od', p.visualAcuityOD);
        safeText('vital-va-os', p.visualAcuityOS);

        // Biomarker Summary
        safeText('lesion-ma-count', p.lesions.microaneurysms);
        safeText('lesion-exudate-count', p.lesions.exudates);
        safeText('lesion-hem-count', p.lesions.hemorrhages);
        safeText('lesion-csme', p.lesions.csmeSuspicion);

        // Medications List
        const medContainer = document.getElementById('medications-list');
        if (medContainer) {
            medContainer.innerHTML = p.medications.map(m => `
                <div class="med-item">
                    <div class="med-info">
                        <strong>${m.name}</strong>
                        <span>${m.dose}</span>
                    </div>
                    <span class="med-freq">${m.freq}</span>
                </div>
            `).join('');
        }

        // Screening Timeline
        const timelineContainer = document.getElementById('timeline-container');
        if (timelineContainer) {
            timelineContainer.innerHTML = p.history.map((h, index) => `
                <div class="timeline-item">
                    <div class="timeline-node ${h.nodeClass}">${index + 1}</div>
                    <div class="timeline-content">
                        <div class="timeline-header">
                            <span class="timeline-date">${h.date} — ${h.title}</span>
                            <span class="timeline-badge ${h.gradeClass}">${h.grade}</span>
                        </div>
                        <div style="font-size:12px; color:var(--text-secondary); margin-bottom:4px;">
                            <strong>Biomarkers:</strong> ${h.lesions} • <strong>HbA1c:</strong> ${h.hba1c}
                        </div>
                        <div style="font-size:12px; color:var(--accent);">
                            <strong>Action:</strong> ${h.action}
                        </div>
                    </div>
                </div>
            `).join('');
        }

        // Referral Slip Preview
        safeText('slip-patient-name', p.name);
        safeText('slip-patient-id', p.id);
        safeText('slip-patient-abha', p.abhaId);
        safeText('slip-patient-age-gender', p.ageGender);
        safeText('slip-patient-village', p.village);
        safeText('slip-icdr', p.icdrText);
        safeText('slip-hba1c', p.hba1c);
        safeText('slip-notes', p.referralNotes);

        // Load saved doctor notes or default
        const notesArea = document.getElementById('doctor-notes-textarea');
        if (notesArea) {
            const saved = localStorage.getItem(`aidr_notes_${p.id}`);
            notesArea.value = saved || `Clinical assessment confirms ${p.icdrText} with ${p.lesions.microaneurysms} microaneurysms. Recommended for vitreoretinal screening at District Hospital. Tele-review queued.`;
        }

        // Render Retinal Scan & Overlays
        renderScanOverlays();
    }

    function safeText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }

    // 3. Clinical Tabs Switching
    function initTabs() {
        const tabBtns = document.querySelectorAll('.tab-btn');
        const tabPanels = document.querySelectorAll('.tab-panel');

        tabBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                tabBtns.forEach(b => b.classList.remove('active'));
                tabPanels.forEach(p => p.classList.remove('active'));

                btn.classList.add('active');
                const targetId = btn.getAttribute('data-tab');
                const targetPanel = document.getElementById(targetId);
                if (targetPanel) {
                    targetPanel.classList.add('active');
                }

                if (targetId === 'tab-retinal') {
                    setTimeout(renderScanOverlays, 50);
                }
            });
        });
    }

    // 4. Eye Selector (OD vs OS)
    function initEyeToggle() {
        const eyeBtns = document.querySelectorAll('.btn-eye-select');
        const eyeBadge = document.getElementById('scan-eye-badge');

        eyeBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                eyeBtns.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                currentEye = btn.getAttribute('data-eye');
                
                if (eyeBadge) {
                    eyeBadge.textContent = currentEye === 'OD' ? 'OD — Right Eye' : 'OS — Left Eye';
                }

                if (scanImg) {
                    scanImg.style.transform = currentEye === 'OS' ? 'scaleX(-1)' : 'scaleX(1)';
                }

                renderScanOverlays();
                showToast(`Retinal viewport changed to ${currentEye}`, 'info');
            });
        });
    }

    // 5. Layer Overlay Toggles (Grad-CAM, MAs, Exudates, Landmarks)
    function initOverlayToggles() {
        const toggles = {
            'toggle-gradcam': 'gradcam',
            'toggle-ma': 'microaneurysms',
            'toggle-exudates': 'exudates',
            'toggle-landmarks': 'landmarks'
        };

        Object.entries(toggles).forEach(([elemId, key]) => {
            const input = document.getElementById(elemId);
            if (input) {
                input.checked = activeOverlays[key];
                input.addEventListener('change', (e) => {
                    activeOverlays[key] = e.target.checked;
                    renderScanOverlays();
                });
            }
        });
    }

    // 6. Retinal Canvas Rendering (Simulated Explainable AI Layers)
    function renderScanOverlays() {
        if (!scanCanvas || !scanImg) return;
        const rect = scanImg.getBoundingClientRect();
        const w = (scanCanvas.width = scanImg.clientWidth || 600);
        const h = (scanCanvas.height = scanImg.clientHeight || 450);

        if (w === 0 || h === 0) return;
        scanCtx.clearRect(0, 0, w, h);

        const p = PATIENT_DATABASE[currentPatientId];
        const isOS = currentEye === 'OS';

        // 1. Grad-CAM Attention Heatmap
        if (activeOverlays.gradcam) {
            const heatSpots = isOS ? [
                { x: w * 0.48, y: h * 0.42, r: 50, color: 'rgba(239, 68, 68, 0.45)' },
                { x: w * 0.35, y: h * 0.52, r: 40, color: 'rgba(245, 158, 11, 0.40)' }
            ] : [
                { x: w * 0.52, y: h * 0.44, r: 55, color: 'rgba(239, 68, 68, 0.48)' },
                { x: w * 0.65, y: h * 0.38, r: 42, color: 'rgba(245, 158, 11, 0.40)' },
                { x: w * 0.38, y: h * 0.62, r: 35, color: 'rgba(16, 185, 129, 0.30)' }
            ];

            heatSpots.forEach(s => {
                const grad = scanCtx.createRadialGradient(s.x, s.y, 4, s.x, s.y, s.r);
                grad.addColorStop(0, s.color);
                grad.addColorStop(0.7, s.color.replace(/[\d\.]+\)$/, '0.12)'));
                grad.addColorStop(1, 'rgba(0,0,0,0)');
                scanCtx.fillStyle = grad;
                scanCtx.beginPath();
                scanCtx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
                scanCtx.fill();
            });
        }

        // 2. Microaneurysm Detection Rings (Red target dots)
        if (activeOverlays.microaneurysms && p.lesions.microaneurysms > 0) {
            const count = Math.min(p.lesions.microaneurysms, 12);
            scanCtx.strokeStyle = '#ef4444';
            scanCtx.lineWidth = 2;
            scanCtx.fillStyle = 'rgba(239, 68, 68, 0.6)';

            for (let i = 0; i < count; i++) {
                const angle = (i / count) * Math.PI * 2;
                const dist = 35 + (i * 7);
                const cx = (w * 0.52) + Math.cos(angle) * dist * (isOS ? -1 : 1);
                const cy = (h * 0.44) + Math.sin(angle) * dist;

                scanCtx.beginPath();
                scanCtx.arc(cx, cy, 6, 0, Math.PI * 2);
                scanCtx.stroke();

                scanCtx.beginPath();
                scanCtx.arc(cx, cy, 2, 0, Math.PI * 2);
                scanCtx.fill();
            }
        }

        // 3. Hard Exudates (Bright yellow clusters)
        if (activeOverlays.exudates && p.lesions.exudates > 0) {
            scanCtx.strokeStyle = '#fbbf24';
            scanCtx.lineWidth = 1.8;
            scanCtx.fillStyle = 'rgba(251, 191, 36, 0.3)';

            const exudateClusters = isOS ? [
                { x: w * 0.40, y: h * 0.35, rx: 18, ry: 10 }
            ] : [
                { x: w * 0.60, y: h * 0.35, rx: 22, ry: 14 },
                { x: w * 0.64, y: h * 0.52, rx: 16, ry: 12 }
            ];

            exudateClusters.forEach(c => {
                scanCtx.beginPath();
                scanCtx.ellipse(c.x, c.y, c.rx, c.ry, 0.2, 0, Math.PI * 2);
                scanCtx.stroke();
                scanCtx.fill();
            });
        }

        // 4. Optic Disc & Fovea Landmarks
        if (activeOverlays.landmarks) {
            scanCtx.strokeStyle = 'rgba(6, 214, 160, 0.85)';
            scanCtx.setLineDash([4, 4]);
            scanCtx.lineWidth = 1.5;

            // Optic disc circle
            const odX = isOS ? w * 0.68 : w * 0.32;
            const odY = h * 0.46;
            scanCtx.beginPath();
            scanCtx.arc(odX, odY, 34, 0, Math.PI * 2);
            scanCtx.stroke();

            // Macula target crosshair
            const macX = isOS ? w * 0.36 : w * 0.64;
            const macY = h * 0.46;
            scanCtx.beginPath();
            scanCtx.arc(macX, macY, 18, 0, Math.PI * 2);
            scanCtx.stroke();
            scanCtx.setLineDash([]);

            scanCtx.fillStyle = '#06d6a0';
            scanCtx.font = '10px JetBrains Mono, sans-serif';
            scanCtx.fillText('DISC', odX - 12, odY - 38);
            scanCtx.fillText('FOVEA', macX - 14, macY - 22);
        }
    }

    // 7. Notes Autosave
    function initNotesAutosave() {
        const notesArea = document.getElementById('doctor-notes-textarea');
        const saveBtn = document.getElementById('btn-save-notes');

        if (saveBtn && notesArea) {
            saveBtn.addEventListener('click', () => {
                localStorage.setItem(`aidr_notes_${currentPatientId}`, notesArea.value);
                showToast('Doctor clinical notes saved to local dossier', 'success');
            });
        }
    }

    // 8. Action Functions (Exposed to Global Window for HTML buttons)
    window.printPatientReport = function() {
        window.print();
    };

    window.triggerSmsAlert = function() {
        const p = PATIENT_DATABASE[currentPatientId];
        showToast(`SMS referral alert sent to ${p.phone} (${p.name}) and ASHA worker ${p.ashaWorker.split(' ')[0]}`, 'success');
    };

    window.sendTeleconsultQueue = function() {
        const p = PATIENT_DATABASE[currentPatientId];
        showToast(`Case ${p.id} synced to District Tele-consultation Queue (Dr. S. Ramanathan)`, 'info');
    };

    // Toast Utility
    function showToast(message, type = 'info') {
        const toast = document.getElementById('patient-toast');
        if (!toast) return;

        toast.textContent = message;
        toast.style.borderColor = type === 'success' ? '#10b981' : (type === 'error' ? '#ef4444' : '#06d6a0');
        toast.style.display = 'block';

        setTimeout(() => {
            toast.style.display = 'none';
        }, 4000);
    }

})();
