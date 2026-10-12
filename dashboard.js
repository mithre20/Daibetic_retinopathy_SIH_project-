/**
 * AI DR Screen — Interactive Dashboard Application Logic
 * Matches the 2026 Rural India Vision UI Reference Design
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // 1. DATA REPOSITORY & STATE
    // -------------------------------------------------------------
    const appState = {
        currentEye: 'OD',
        zoomActive: false,
        heatmapVisible: false,
        language: 'en',
        isOnline: true,
        period: 'This Month',
        statsData: {
            'This Month': { total: 243, referable: 67, normal: 176, followups: 34, growth: '+18% from last month' },
            'Last Month': { total: 206, referable: 58, normal: 148, followups: 29, growth: '+12% from previous' },
            'This Year': { total: 1842, referable: 512, normal: 1330, followups: 265, growth: '+42% year-to-date' }
        },
        patients: [
            {
                id: 'DR25072026',
                name: 'Ramesh Kumar',
                ageGender: '58 / Male',
                duration: '8 Years',
                village: 'Thirunallur',
                severity: 'Moderate',
                confidence: 92,
                image: 'fundus_scan.jpg'
            },
            {
                id: 'DR25072027',
                name: 'Lakshmi Ammal',
                ageGender: '62 / Female',
                duration: '12 Years',
                village: 'Kottakuppam',
                severity: 'Severe',
                confidence: 96,
                image: 'fundus_scan.jpg'
            },
            {
                id: 'DR25072028',
                name: 'Murugan P.',
                ageGender: '47 / Male',
                duration: '4 Years',
                village: 'Pondicherry Rural',
                severity: 'Mild',
                confidence: 88,
                image: 'fundus_scan.jpg'
            }
        ]
    };

    // -------------------------------------------------------------
    // 2. RETINAL VIEW & THUMBNAIL SWITCHER
    // -------------------------------------------------------------
    const retinalImg = document.getElementById('retinal-img');
    const eyeBadge = document.querySelector('.eye-badge');
    const thumbs = document.querySelectorAll('.retinal-thumbs .thumb');
    const canvas = document.getElementById('retinal-heatmap-overlay');

    thumbs.forEach(thumb => {
        thumb.addEventListener('click', () => {
            thumbs.forEach(t => t.classList.remove('active'));
            thumb.classList.add('active');

            const label = thumb.querySelector('span').textContent.trim();
            if (eyeBadge) {
                eyeBadge.textContent = label.includes('OD') ? 'OD' : 'OS';
            }

            // Visual scan adjustment for selected thumbnail
            if (label.includes('OS')) {
                retinalImg.style.transform = 'scaleX(-1)';
            } else {
                retinalImg.style.transform = 'scaleX(1)';
            }

            if (label.includes('Zoom')) {
                retinalImg.style.transform += ' scale(1.35)';
                retinalImg.style.transition = 'transform 0.4s ease';
                showHeatmapOverlay();
            } else {
                retinalImg.style.transform = label.includes('OS') ? 'scaleX(-1) scale(1)' : 'scale(1)';
                hideHeatmapOverlay();
            }

            showToast(`Switched view to ${label}`, 'info');
        });
    });

    // -------------------------------------------------------------
    // 3. HEATMAP OVERLAY VISUALIZATION (CANVAS)
    // -------------------------------------------------------------
    function showHeatmapOverlay() {
        if (!canvas) return;
        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        const w = (canvas.width = canvas.offsetWidth || 340);
        const h = (canvas.height = canvas.offsetHeight || 340);

        ctx.clearRect(0, 0, w, h);

        // Simulated AI Grad-CAM heat signatures for microaneurysms
        const lesions = [
            { x: w * 0.42, y: h * 0.48, r: 42, color: 'rgba(239, 68, 68, 0.45)' },
            { x: w * 0.62, y: h * 0.38, r: 35, color: 'rgba(245, 158, 11, 0.4)' },
            { x: w * 0.35, y: h * 0.65, r: 28, color: 'rgba(239, 68, 68, 0.35)' },
            { x: w * 0.52, y: h * 0.58, r: 50, color: 'rgba(16, 185, 129, 0.25)' }
        ];

        lesions.forEach(item => {
            const radGrad = ctx.createRadialGradient(item.x, item.y, 4, item.x, item.y, item.r);
            radGrad.addColorStop(0, item.color);
            radGrad.addColorStop(0.7, item.color.replace(/[\d\.]+\)$/, '0.15)'));
            radGrad.addColorStop(1, 'rgba(0,0,0,0)');
            ctx.fillStyle = radGrad;
            ctx.beginPath();
            ctx.arc(item.x, item.y, item.r, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    function hideHeatmapOverlay() {
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.style.display = 'none';
    }

    // -------------------------------------------------------------
    // 4. PERIOD FILTER & ANIMATED STAT COUNTERS
    // -------------------------------------------------------------
    const periodSelect = document.getElementById('period-select');
    if (periodSelect) {
        periodSelect.addEventListener('change', (e) => {
            const selected = e.target.value;
            const data = appState.statsData[selected];
            if (!data) return;

            animateStatNumber('.stat-tile.blue .stat-num', data.total);
            animateStatNumber('.stat-tile.red .stat-num', data.referable);
            animateStatNumber('.stat-tile.green .stat-num', data.normal);
            animateStatNumber('.stat-tile.purple .stat-num', data.followups);

            const trendEl = document.querySelector('.stat-tile.blue .stat-trend');
            if (trendEl) trendEl.textContent = data.growth;

            showToast(`Metrics updated for ${selected}`, 'info');
        });
    }

    function animateStatNumber(selector, targetVal) {
        const el = document.querySelector(selector);
        if (!el) return;
        const currentVal = parseInt(el.textContent.replace(/[^\d]/g, ''), 10) || 0;
        const duration = 500;
        const steps = 20;
        const inc = (targetVal - currentVal) / steps;
        let step = 0;

        const timer = setInterval(() => {
            step++;
            const val = Math.round(currentVal + inc * step);
            el.textContent = val;
            if (step >= steps) {
                el.textContent = targetVal;
                clearInterval(timer);
            }
        }, duration / steps);
    }

    // -------------------------------------------------------------
    // 5. STATUS TOGGLE (ONLINE / OFFLINE)
    // -------------------------------------------------------------
    const statusBadge = document.querySelector('.status-badge');
    if (statusBadge) {
        statusBadge.style.cursor = 'pointer';
        statusBadge.setAttribute('title', 'Click to toggle Online/Offline Edge Mode');
        statusBadge.addEventListener('click', () => {
            appState.isOnline = !appState.isOnline;
            if (appState.isOnline) {
                statusBadge.className = 'status-badge online';
                statusBadge.innerHTML = '<span class="status-dot"></span> Online';
                showToast('Connected to Central Health Cloud', 'success');
            } else {
                statusBadge.className = 'status-badge offline';
                statusBadge.style.borderColor = 'rgba(245, 158, 11, 0.4)';
                statusBadge.innerHTML = '<span class="status-dot" style="background:#f59e0b; box-shadow:0 0 8px #f59e0b"></span> Offline Edge';
                showToast('Offline Mode: Using on-device AI inference', 'warning');
            }
        });
    }

    // -------------------------------------------------------------
    // 6. LANGUAGE SELECTOR (ENGLISH / TAMIL / HINDI)
    // -------------------------------------------------------------
    const langBtn = document.getElementById('lang-btn');
    if (langBtn) {
        langBtn.addEventListener('click', () => {
            const languages = ['English', 'தமிழ் (Tamil)', 'हिंदी (Hindi)'];
            const currentIdx = languages.findIndex(l => l.includes(langBtn.textContent.trim().split(' ')[0]));
            const nextIdx = (currentIdx + 1) % languages.length;
            const nextLang = languages[nextIdx];

            langBtn.innerHTML = `
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6.5" stroke="currentColor" stroke-width="1.3"/><path d="M1.5 8h13M8 1.5c-2 2.5-2 10.5 0 13M8 1.5c2 2.5 2 10.5 0 13" stroke="currentColor" stroke-width="1.1"/></svg>
                ${nextLang.split(' ')[0]}
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M3 5l3 3 3-3" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
            `;

            showToast(`Language switched to ${nextLang}`, 'info');
        });
    }

    // -------------------------------------------------------------
    // 7. SIDEBAR NAVIGATION ACTIVE STATES
    // -------------------------------------------------------------
    const navLinks = document.querySelectorAll('.sidebar-nav .nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            const page = link.getAttribute('data-page');
            showToast(`Navigated to ${link.textContent.trim()}`, 'info');
        });
    });

    // -------------------------------------------------------------
    // 8. MODAL DIALOGS (NEW SCREENING, PATIENT PROFILE, REPORTS)
    // -------------------------------------------------------------
    createModals();

    const startBtn = document.getElementById('btn-start-screening');
    if (startBtn) {
        startBtn.addEventListener('click', () => openModal('modal-screening'));
    }

    const newScreeningNav = document.querySelector('[data-page="new-screening"]');
    if (newScreeningNav) {
        newScreeningNav.addEventListener('click', () => openModal('modal-screening'));
    }

    const profileBtn = document.getElementById('btn-view-profile');
    if (profileBtn) {
        profileBtn.addEventListener('click', () => {
            window.location.href = 'patient-details.html?id=DR25072026';
        });
    }

    const patientRecordsNav = document.querySelector('[data-page="patient-records"]');
    if (patientRecordsNav) {
        patientRecordsNav.addEventListener('click', (e) => {
            e.preventDefault();
            window.location.href = 'patient-details.html?id=DR25072026';
        });
    }

    const reportBtn = document.getElementById('btn-detailed-report');
    if (reportBtn) {
        reportBtn.addEventListener('click', () => openModal('modal-report'));
    }

    // -------------------------------------------------------------
    // 9. TOAST NOTIFICATION UTILITY
    // -------------------------------------------------------------
    function showToast(message, type = 'info') {
        let container = document.getElementById('toast-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toast-container';
            container.style.cssText = `
                position: fixed;
                bottom: 24px;
                right: 24px;
                display: flex;
                flex-direction: column;
                gap: 10px;
                z-index: 9999;
                pointer-events: none;
            `;
            document.body.appendChild(container);
        }

        const toast = document.createElement('div');
        const colors = {
            info: { bg: '#151d32', border: '#3b82f6', text: '#93c5fd' },
            success: { bg: '#151d32', border: '#10b981', text: '#6ee7b7' },
            warning: { bg: '#151d32', border: '#f59e0b', text: '#fcd34d' },
            error: { bg: '#151d32', border: '#ef4444', text: '#fca5a5' }
        };
        const c = colors[type] || colors.info;

        toast.style.cssText = `
            background: ${c.bg};
            border: 1px solid ${c.border};
            color: #ffffff;
            padding: 12px 18px;
            border-radius: 12px;
            font-size: 13px;
            font-weight: 500;
            box-shadow: 0 8px 30px rgba(0,0,0,0.4);
            display: flex;
            align-items: center;
            gap: 10px;
            opacity: 0;
            transform: translateY(12px);
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            backdrop-filter: blur(12px);
            pointer-events: auto;
        `;
        toast.innerHTML = `<span style="color:${c.border}; font-size:16px;">●</span> ${message}`;

        container.appendChild(toast);
        requestAnimationFrame(() => {
            toast.style.opacity = '1';
            toast.style.transform = 'translateY(0)';
        });

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(8px)';
            setTimeout(() => toast.remove(), 300);
        }, 3200);
    }

    // -------------------------------------------------------------
    // 10. MODAL CREATION FUNCTION
    // -------------------------------------------------------------
    function createModals() {
        const modalContainer = document.createElement('div');
        modalContainer.id = 'modals-root';
        modalContainer.innerHTML = `
            <!-- Screening Modal -->
            <div id="modal-screening" class="app-modal" style="display:none;">
                <div class="modal-backdrop"></div>
                <div class="modal-dialog">
                    <div class="modal-header">
                        <h3>New Retinal AI Screening</h3>
                        <button class="modal-close">&times;</button>
                    </div>
                    <div class="modal-body">
                        <div class="upload-dropzone">
                            <svg width="48" height="48" viewBox="0 0 48 48" fill="none"><circle cx="24" cy="24" r="22" stroke="#06d6a0" stroke-width="2" stroke-dasharray="4 4"/><path d="M24 16v16M16 24l8-8 8 8" stroke="#06d6a0" stroke-width="2.5" stroke-linecap="round"/></svg>
                            <p><strong>Upload Retinal Fundus Scan</strong></p>
                            <p style="font-size:12px; color:#8b97b5;">Drop image here, take camera photo, or browse JPG/PNG</p>
                            <input type="file" id="modal-file-input" style="display:none;" accept="image/*">
                            <button class="btn-upload-browse" onclick="document.getElementById('modal-file-input').click()">Browse Image</button>
                        </div>
                        <div class="patient-quick-form" style="margin-top:20px; display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                            <div>
                                <label style="font-size:11px; color:#8b97b5;">Patient Name</label>
                                <input type="text" value="New Rural Patient" style="width:100%; background:#121a2e; border:1px solid rgba(255,255,255,0.1); color:#fff; padding:8px 12px; border-radius:8px;">
                            </div>
                            <div>
                                <label style="font-size:11px; color:#8b97b5;">Eye Tested</label>
                                <select style="width:100%; background:#121a2e; border:1px solid rgba(255,255,255,0.1); color:#fff; padding:8px 12px; border-radius:8px;">
                                    <option>OD (Right Eye)</option>
                                    <option>OS (Left Eye)</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer" style="padding:16px 24px; display:flex; justify-content:flex-end; gap:12px; border-top:1px solid rgba(255,255,255,0.06);">
                        <button class="btn-cancel" onclick="closeAllModals()">Cancel</button>
                        <button class="btn-run-ai" style="background:linear-gradient(135deg, #06d6a0, #3b82f6); color:#fff; border:none; padding:8px 20px; border-radius:10px; font-weight:600; cursor:pointer;">Run AI Diagnostic</button>
                    </div>
                </div>
            </div>

            <!-- Profile Modal -->
            <div id="modal-profile" class="app-modal" style="display:none;">
                <div class="modal-backdrop"></div>
                <div class="modal-dialog">
                    <div class="modal-header">
                        <h3>Patient Clinical Profile — Ramesh Kumar</h3>
                        <button class="modal-close">&times;</button>
                    </div>
                    <div class="modal-body" style="font-size:13px; line-height:1.7;">
                        <p><strong>Patient ID:</strong> DR25072026</p>
                        <p><strong>Village:</strong> Thirunallur, Tamil Nadu</p>
                        <p><strong>Primary Care Unit:</strong> Primary Health Center - Zone 4</p>
                        <p><strong>HbA1c Level:</strong> 8.6% (Uncontrolled)</p>
                        <p><strong>Systolic BP:</strong> 138/86 mmHg</p>
                        <p><strong>Prior Screenings:</strong> 2 prior scans (Aug 2025, Nov 2024)</p>
                    </div>
                    <div class="modal-footer" style="padding:16px 24px; display:flex; justify-content:flex-end;">
                        <button class="btn-cancel" onclick="closeAllModals()">Close</button>
                    </div>
                </div>
            </div>

            <!-- Report Modal -->
            <div id="modal-report" class="app-modal" style="display:none;">
                <div class="modal-backdrop"></div>
                <div class="modal-dialog">
                    <div class="modal-header">
                        <h3>Clinical Diagnostic Report</h3>
                        <button class="modal-close">&times;</button>
                    </div>
                    <div class="modal-body" style="font-size:13px; line-height:1.7;">
                        <div style="background:#1a2340; padding:16px; border-radius:12px; margin-bottom:12px;">
                            <h4 style="color:#06d6a0; margin-bottom:6px;">ICDR Severity: Moderate NPDR</h4>
                            <p>Confidence: 92% (Deep Neural Ensemble)</p>
                            <p>Key Biomarkers: Multiple Microaneurysms in macula, early hard exudate rings, clear optic disc boundary.</p>
                        </div>
                        <p><strong>Actionable Pathway:</strong> Referral slip generated for District Hospital Ophthalmology wing within 2 to 4 weeks. Automated SMS reminder queued for patient Ramesh Kumar.</p>
                    </div>
                    <div class="modal-footer" style="padding:16px 24px; display:flex; justify-content:flex-end; gap:12px;">
                        <button class="btn-cancel" onclick="closeAllModals()">Close</button>
                        <button class="btn-run-ai" onclick="window.print()" style="background:#8b5cf6; color:#fff; border:none; padding:8px 20px; border-radius:10px; cursor:pointer;">Print / Export PDF</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modalContainer);

        // Add modal styles dynamically
        const modalStyle = document.createElement('style');
        modalStyle.textContent = `
            .app-modal {
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .modal-backdrop {
                position: absolute;
                top: 0; left: 0; width: 100%; height: 100%;
                background: rgba(10, 14, 26, 0.75);
                backdrop-filter: blur(8px);
            }
            .modal-dialog {
                position: relative;
                width: 90%;
                max-width: 520px;
                background: #151d32;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 18px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
                overflow: hidden;
                animation: modalIn 0.25s ease-out;
            }
            @keyframes modalIn {
                from { opacity: 0; transform: scale(0.94); }
                to { opacity: 1; transform: scale(1); }
            }
            .modal-header {
                padding: 18px 24px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.06);
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
            .modal-header h3 { font-size: 16px; color: #fff; font-weight: 600; }
            .modal-close {
                background: none; border: none; font-size: 24px; color: #8b97b5; cursor: pointer;
            }
            .modal-close:hover { color: #fff; }
            .modal-body { padding: 20px 24px; color: #d1d5db; }
            .upload-dropzone {
                border: 2px dashed rgba(6, 214, 160, 0.35);
                border-radius: 14px;
                padding: 30px 20px;
                text-align: center;
                background: rgba(6, 214, 160, 0.03);
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 8px;
                cursor: pointer;
            }
            .btn-upload-browse {
                margin-top: 8px;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.15);
                color: #fff;
                padding: 6px 16px;
                border-radius: 8px;
                cursor: pointer;
            }
            .btn-cancel {
                background: rgba(255, 255, 255, 0.06);
                border: 1px solid rgba(255, 255, 255, 0.1);
                color: #8b97b5;
                padding: 8px 16px;
                border-radius: 10px;
                cursor: pointer;
            }
            .btn-cancel:hover { color: #fff; }
        `;
        document.head.appendChild(modalStyle);

        // Bind close buttons
        document.querySelectorAll('.modal-close').forEach(btn => {
            btn.addEventListener('click', closeAllModals);
        });
        document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
            backdrop.addEventListener('click', closeAllModals);
        });

        // File upload simulation
        const fileIn = document.getElementById('modal-file-input');
        if (fileIn) {
            fileIn.addEventListener('change', (e) => {
                if (e.target.files && e.target.files[0]) {
                    const reader = new FileReader();
                    reader.onload = (re) => {
                        if (retinalImg) retinalImg.src = re.target.result;
                        closeAllModals();
                        showToast('New retinal scan uploaded and analyzed!', 'success');
                    };
                    reader.readAsDataURL(e.target.files[0]);
                }
            });
        }
    }

    window.openModal = function(id) {
        const modal = document.getElementById(id);
        if (modal) modal.style.display = 'flex';
    };

    window.closeAllModals = function() {
        document.querySelectorAll('.app-modal').forEach(m => m.style.display = 'none');
    };

    // -------------------------------------------------------------
    // 11. USER SESSION & SIGN-OUT DROPDOWN
    // -------------------------------------------------------------
    initUserSession();
    initUserDropdown();

    function initUserSession() {
        try {
            const raw = localStorage.getItem('aidr_user_session');
            const nameEl = document.querySelector('.doctor-name');
            const centerEl = document.querySelector('.doctor-center');
            const menuTitleEl = document.getElementById('doctor-menu-user-name');

            if (raw) {
                const session = JSON.parse(raw);
                if (session && session.user) {
                    if (nameEl) nameEl.textContent = session.user.name || 'Medical Officer';
                    if (centerEl) centerEl.textContent = session.user.center || session.user.roleLabel || 'Primary Health Center';
                    if (menuTitleEl) menuTitleEl.textContent = session.user.name || 'Medical Officer';
                    return;
                }
            }
            if (nameEl) nameEl.textContent = 'Dr. Anitha';
            if (centerEl) centerEl.textContent = 'Primary Health Center';
            if (menuTitleEl) menuTitleEl.textContent = 'Dr. Anitha';
        } catch (e) {}
    }

    window.addEventListener('message', (e) => {
        if (e.data && e.data.action === 'sessionUpdated') {
            initUserSession();
        }
    });

    window.addEventListener('storage', (e) => {
        if (e.key === 'aidr_user_session') {
            initUserSession();
        }
    });

    function initUserDropdown() {
        const docProfile = document.querySelector('.doctor-profile');
        if (!docProfile || document.getElementById('doctor-dropdown-menu')) return;

        docProfile.style.cursor = 'pointer';
        docProfile.style.position = 'relative';

        const menu = document.createElement('div');
        menu.id = 'doctor-dropdown-menu';
        menu.style.cssText = `
            display: none;
            position: absolute;
            top: 100%;
            right: 0;
            margin-top: 10px;
            width: 220px;
            background: #151d32;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 14px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
            padding: 8px;
            z-index: 9999;
            backdrop-filter: blur(16px);
        `;

        menu.innerHTML = `
            <div style="padding: 10px 12px; border-bottom: 1px solid rgba(255,255,255,0.06); font-size: 11.5px; color: #8b97b5;">
                Signed in as <strong id="doctor-menu-user-name" style="color:#fff; display:block; margin-top:2px;">Dr. Anitha</strong>
            </div>
            <a href="#" id="btn-switch-account" style="display:flex; align-items:center; gap:8px; padding:10px 12px; color:#e2e8f0; text-decoration:none; font-size:12px; border-radius:8px; transition:0.2s;">
                <svg width="14" height="14" viewBox="0 0 20 20" fill="none"><circle cx="10" cy="10" r="7" stroke="currentColor" stroke-width="1.5"/><path d="M10 6v4l3 2" stroke="currentColor" stroke-width="1.5"/></svg>
                Switch Account / Login
            </a>
            <button id="btn-signout" style="width:100%; display:flex; align-items:center; gap:8px; padding:10px 12px; color:#ef4444; background:none; border:none; font-size:12px; border-radius:8px; cursor:pointer; text-align:left; transition:0.2s;">
                <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M3 3a1 1 0 011 1v12a1 1 0 11-2 0V4a1 1 0 011-1zm7.707 3.293a1 1 0 010 1.414L9.414 9H17a1 1 0 110 2H9.414l1.293 1.293a1 1 0 01-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0z" clip-rule="evenodd"/></svg>
                Sign Out Terminal
            </button>
        `;

        docProfile.appendChild(menu);

        docProfile.addEventListener('click', (e) => {
            e.stopPropagation();
            menu.style.display = menu.style.display === 'block' ? 'none' : 'block';
        });

        document.addEventListener('click', () => {
            menu.style.display = 'none';
        });

        const switchBtn = menu.querySelector('#btn-switch-account');
        if (switchBtn) {
            switchBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (window.parent && typeof window.parent.switchPortalTab === 'function') {
                    window.parent.switchPortalTab('login');
                } else {
                    window.location.href = 'login.html';
                }
            });
        }

        const signOutBtn = menu.querySelector('#btn-signout');
        if (signOutBtn) {
            signOutBtn.addEventListener('click', () => {
                localStorage.removeItem('aidr_user_session');
                showToast('Logged out successfully', 'info');
                initUserSession();
                setTimeout(() => {
                    if (window.parent && typeof window.parent.switchPortalTab === 'function') {
                        if (typeof window.parent.updatePortalUser === 'function') {
                            window.parent.updatePortalUser();
                        }
                        window.parent.switchPortalTab('login');
                    } else {
                        window.location.href = 'login.html';
                    }
                }, 400);
            });
        }
    }

    // -------------------------------------------------------------
    // THEME SYNCHRONIZATION WITH PARENT PORTAL
    // -------------------------------------------------------------
    function applyDashboardTheme(theme) {
        const active = theme || localStorage.getItem('xairis_theme') || 'light';
        document.documentElement.setAttribute('data-theme', active);
        document.body.setAttribute('data-theme', active);
    }
    applyDashboardTheme();

    window.addEventListener('message', (e) => {
        if (e.data && e.data.action === 'themeChange') {
            applyDashboardTheme(e.data.theme);
        }
    });

    // -------------------------------------------------------------
    // 3D PATIENT RETINAL TWIN MODAL (THREE.JS)
    // -------------------------------------------------------------
    let modal3DScene, modal3DCamera, modal3DRenderer, modal3DGroup;
    let modal3DInitialized = false;
    let modal3DAnimId = null;

    window.open3DRetinalModal = function() {
        const modal = document.getElementById('modal-3d-retina');
        if (!modal) return;
        modal.style.display = 'flex';

        if (!modal3DInitialized) {
            setTimeout(initModal3D, 50);
        } else {
            onModalResize();
            if (!modal3DAnimId) runModalRenderLoop();
        }
    };

    window.close3DRetinalModal = function() {
        const modal = document.getElementById('modal-3d-retina');
        if (modal) modal.style.display = 'none';
        if (modal3DAnimId) {
            cancelAnimationFrame(modal3DAnimId);
            modal3DAnimId = null;
        }
    };

    function initModal3D() {
        const wrap = document.getElementById('dashboard-3d-wrap');
        const canvas = document.getElementById('dashboard-3d-canvas');
        if (!wrap || !canvas || typeof THREE === 'undefined') return;

        modal3DScene = new THREE.Scene();
        modal3DCamera = new THREE.PerspectiveCamera(45, wrap.clientWidth / wrap.clientHeight, 0.1, 100);
        modal3DCamera.position.set(0, 0, 7.5);

        modal3DRenderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
        modal3DRenderer.setSize(wrap.clientWidth, wrap.clientHeight);
        modal3DRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Lights
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const amb = new THREE.AmbientLight(isDark ? 0x223344 : 0xffffff, isDark ? 1.2 : 1.6);
        modal3DScene.add(amb);

        const dir = new THREE.DirectionalLight(0xffffff, 1.4);
        dir.position.set(5, 8, 7);
        modal3DScene.add(dir);

        modal3DGroup = new THREE.Group();
        modal3DScene.add(modal3DGroup);

        // 3D Hemispherical Retinal Fundus Bowl
        const fundusGeo = new THREE.SphereGeometry(2.6, 48, 36, 0, Math.PI * 2, 0, Math.PI * 0.58);
        const fundusMat = new THREE.MeshStandardMaterial({
            color: 0xc2410c,
            roughness: 0.5,
            side: THREE.BackSide
        });
        const fundusMesh = new THREE.Mesh(fundusGeo, fundusMat);
        fundusMesh.rotation.x = Math.PI;
        modal3DGroup.add(fundusMesh);

        // Optic Disc
        const odGeo = new THREE.CircleGeometry(0.35, 32);
        const odMat = new THREE.MeshBasicMaterial({ color: 0xfde047, side: THREE.DoubleSide });
        const odMesh = new THREE.Mesh(odGeo, odMat);
        odMesh.position.set(0.9, 0.15, -2.4);
        modal3DGroup.add(odMesh);

        // Macula & Fovea
        const macGeo = new THREE.CircleGeometry(0.5, 32);
        const macMat = new THREE.MeshBasicMaterial({ color: 0x78350f, side: THREE.DoubleSide });
        const macMesh = new THREE.Mesh(macGeo, macMat);
        macMesh.position.set(-0.5, -0.1, -2.45);
        modal3DGroup.add(macMesh);

        // Patient Specific Moderate NPDR Lesions (Grade 2)
        const patientLesions = [
            { name: 'Microaneurysm 1', x: -0.3, y: 0.3, z: -2.35, color: 0xef4444, size: 0.08 },
            { name: 'Microaneurysm 2', x: 0.2, y: -0.4, z: -2.4, color: 0xef4444, size: 0.08 },
            { name: 'Blot Hemorrhage 1', x: -0.8, y: -0.5, z: -2.2, color: 0xdc2626, size: 0.13 },
            { name: 'Blot Hemorrhage 2', x: 0.6, y: 0.7, z: -2.15, color: 0xdc2626, size: 0.12 },
            { name: 'Hard Exudate', x: -0.1, y: 0.6, z: -2.3, color: 0xfbbf24, size: 0.1 }
        ];

        patientLesions.forEach(les => {
            const bGeo = new THREE.SphereGeometry(les.size, 16, 16);
            const bMat = new THREE.MeshBasicMaterial({ color: les.color });
            const bMesh = new THREE.Mesh(bGeo, bMat);
            bMesh.position.set(les.x, les.y, les.z);
            modal3DGroup.add(bMesh);

            const rGeo = new THREE.RingGeometry(les.size * 1.4, les.size * 1.8, 16);
            const rMat = new THREE.MeshBasicMaterial({ color: les.color, side: THREE.DoubleSide });
            const rMesh = new THREE.Mesh(rGeo, rMat);
            rMesh.position.set(les.x, les.y, les.z);
            modal3DGroup.add(rMesh);
        });

        // Orbit Drag Interaction on modal
        let isDrag = false, pX = 0, pY = 0, rX = 0.1, rY = 0;
        wrap.addEventListener('mousedown', (e) => { isDrag = true; pX = e.clientX; pY = e.clientY; });
        window.addEventListener('mousemove', (e) => {
            if (!isDrag) return;
            rY += (e.clientX - pX) * 0.01;
            rX += (e.clientY - pY) * 0.01;
            pX = e.clientX; pY = e.clientY;
            modal3DGroup.rotation.x = rX;
            modal3DGroup.rotation.y = rY;
        });
        window.addEventListener('mouseup', () => { isDrag = false; });

        runModalRenderLoop();
        modal3DInitialized = true;
    }

    function runModalRenderLoop() {
        function render() {
            modal3DAnimId = requestAnimationFrame(render);
            if (modal3DGroup) modal3DGroup.rotation.y += 0.003;
            if (modal3DRenderer && modal3DScene && modal3DCamera) {
                modal3DRenderer.render(modal3DScene, modal3DCamera);
            }
        }
        render();
    }

    function onModalResize() {
        const wrap = document.getElementById('dashboard-3d-wrap');
        if (wrap && modal3DCamera && modal3DRenderer) {
            modal3DCamera.aspect = wrap.clientWidth / wrap.clientHeight;
            modal3DCamera.updateProjectionMatrix();
            modal3DRenderer.setSize(wrap.clientWidth, wrap.clientHeight);
        }
    }
});

