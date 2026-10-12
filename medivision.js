/* ============================================================
   MediVision AI — Application Logic
   Dashboard interactions, charts, data, heatmap, i18n
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

    // ===================== Date Display =====================
    const dateEl = document.getElementById('date-display');
    const now = new Date();
    dateEl.textContent = now.toLocaleDateString('en-IN', {
        weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
    });

    // ===================== Sidebar Toggle =====================
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebar-toggle');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');

    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('collapsed');
    });

    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.toggle('mobile-open');
    });

    // Close mobile sidebar on outside click
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && sidebar.classList.contains('mobile-open')
            && !sidebar.contains(e.target) && e.target !== mobileMenuBtn) {
            sidebar.classList.remove('mobile-open');
        }
    });

    // ===================== Page Navigation =====================
    const navItems = document.querySelectorAll('.nav-item');
    const pages = document.querySelectorAll('.page');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const target = item.dataset.page;
            if (target === 'settings') {
                if (window.parent && window.parent.switchPortalTab) {
                    window.parent.switchPortalTab('settings');
                    return;
                } else {
                    window.location.href = 'settings.html';
                    return;
                }
            }
            navItems.forEach(n => n.classList.remove('active'));
            item.classList.add('active');
            pages.forEach(p => p.classList.toggle('active', p.id === `page-${target}`));
            // Mobile: close sidebar
            if (window.innerWidth <= 768) sidebar.classList.remove('mobile-open');
        });
    });

    // New Screening button → AI Screening page
    document.getElementById('new-screening-btn').addEventListener('click', () => {
        document.querySelector('[data-page="screening"]').click();
    });

    // ===================== Language Toggle =====================
    const langBtn = document.getElementById('lang-btn');
    const langDropdown = document.getElementById('lang-dropdown');
    const langLabel = document.getElementById('lang-label');
    let currentLang = 'en';

    langBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        langDropdown.classList.toggle('open');
    });

    document.addEventListener('click', () => langDropdown.classList.remove('open'));

    document.querySelectorAll('.lang-option').forEach(opt => {
        opt.addEventListener('click', () => {
            currentLang = opt.dataset.lang;
            langLabel.textContent = currentLang === 'en' ? 'EN' : 'தமிழ்';
            document.querySelectorAll('.lang-option').forEach(o => o.classList.remove('active'));
            opt.classList.add('active');
            langDropdown.classList.remove('open');
            applyLanguage(currentLang);
            showToast(`Language changed to ${currentLang === 'en' ? 'English' : 'தமிழ்'}`, 'info');
        });
    });

    function applyLanguage(lang) {
        document.querySelectorAll('[data-en]').forEach(el => {
            const text = el.getAttribute(`data-${lang}`);
            if (text) el.textContent = text;
        });
    }

    // ===================== Voice Assistant =====================
    const voiceModal = document.getElementById('voice-modal');
    const voiceBtn = document.getElementById('voice-btn');
    const voiceClose = document.getElementById('voice-close');

    voiceBtn.addEventListener('click', () => {
        voiceModal.classList.add('open');
    });

    voiceClose.addEventListener('click', () => {
        voiceModal.classList.remove('open');
    });

    // ===================== Notification =====================
    document.getElementById('notification-btn').addEventListener('click', () => {
        showToast('3 new screening results awaiting review', 'info');
    });

    // ===================== Toast System =====================
    function showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        const icons = { success: '✓', error: '✕', info: 'ℹ' };
        toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span> ${message}`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(8px)';
            toast.style.transition = '0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    // ===================== Doctor Actions =====================
    document.getElementById('btn-confirm').addEventListener('click', () => {
        showToast('Diagnosis confirmed and saved to patient record', 'success');
    });

    document.getElementById('btn-second-opinion').addEventListener('click', () => {
        showToast('Request for second opinion sent to Dr. Venkatesh', 'info');
    });

    document.getElementById('btn-generate-report').addEventListener('click', () => {
        showToast('Clinical report generated successfully', 'success');
    });

    // ===================== Recent Patients =====================
    const patients = [
        { name: 'Rajesh Kumar', age: 58, id: 'MV-2847', type: 'DR Screening', status: 'moderate', statusText: 'Moderate', time: '10 min ago', color: '#2563eb' },
        { name: 'Lakshmi Devi', age: 45, id: 'MV-2846', type: 'Retinal Scan', status: 'normal', statusText: 'Normal', time: '25 min ago', color: '#059669' },
        { name: 'Mohan Rajan', age: 62, id: 'MV-2845', type: 'DR Screening', status: 'high', statusText: 'High Risk', time: '1 hr ago', color: '#dc2626' },
        { name: 'Anitha Selvam', age: 51, id: 'MV-2844', type: 'Follow-up', status: 'normal', statusText: 'Normal', time: '2 hrs ago', color: '#7c3aed' },
        { name: 'Suresh Babu', age: 67, id: 'MV-2843', type: 'DR Screening', status: 'pending', statusText: 'Pending', time: '3 hrs ago', color: '#0d9488' },
        { name: 'Kavitha Raman', age: 43, id: 'MV-2842', type: 'Retinal Scan', status: 'normal', statusText: 'Normal', time: '4 hrs ago', color: '#ea580c' },
    ];

    const patientsList = document.getElementById('patients-list');
    patients.forEach(p => {
        const initials = p.name.split(' ').map(w => w[0]).join('');
        patientsList.innerHTML += `
            <div class="patient-row">
                <div class="patient-avatar" style="background: ${p.color};">${initials}</div>
                <div class="patient-info">
                    <span class="patient-name">${p.name}</span>
                    <span class="patient-detail">Age ${p.age} • ${p.id} • ${p.type}</span>
                </div>
                <span class="patient-status ${p.status}">${p.statusText}</span>
                <span class="patient-time">${p.time}</span>
            </div>
        `;
    });

    // ===================== Patient Timeline =====================
    const timeline = document.getElementById('timeline');
    const timelineData = [
        { date: 'Sep 5, 2026', title: 'AI Screening — Moderate NPDR', desc: 'ICDR Grade 2 detected. 12 microaneurysms, hard exudates present near macula.', dot: 'amber' },
        { date: 'Jun 12, 2026', title: 'Follow-up — Mild NPDR', desc: 'Slight progression from previous visit. 5 microaneurysms identified.', dot: '' },
        { date: 'Mar 3, 2026', title: 'Initial Screening — Mild NPDR', desc: 'First AI-assisted screening. 2 microaneurysms detected in temporal retina.', dot: 'green' },
        { date: 'Dec 15, 2025', title: 'Baseline Eye Exam', desc: 'No diabetic retinopathy detected. Normal fundus appearance.', dot: 'green' },
    ];

    timelineData.forEach(t => {
        timeline.innerHTML += `
            <div class="timeline-item">
                <div class="timeline-dot ${t.dot}"></div>
                <div class="timeline-date">${t.date}</div>
                <div class="timeline-title">${t.title}</div>
                <div class="timeline-desc">${t.desc}</div>
            </div>
        `;
    });

    // ===================== Screening Trends Chart =====================
    function drawTrendsChart() {
        const canvas = document.getElementById('trends-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = 220 * dpr;
        ctx.scale(dpr, dpr);
        const W = rect.width, H = 220;

        const data = [120, 145, 132, 168, 155, 178, 192, 185, 210, 198, 225, 240, 218, 255];
        const labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

        const padL = 44, padR = 16, padT = 16, padB = 32;
        const plotW = W - padL - padR;
        const plotH = H - padT - padB;
        const maxVal = Math.max(...data) * 1.1;

        // Grid lines
        ctx.strokeStyle = '#e8ecf1';
        ctx.lineWidth = 0.8;
        ctx.font = '10px Inter, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const y = padT + plotH * (1 - i / 4);
            ctx.beginPath();
            ctx.moveTo(padL, y);
            ctx.lineTo(W - padR, y);
            ctx.stroke();
            ctx.fillText(Math.round(maxVal * i / 4), padL - 8, y + 4);
        }

        // X labels
        ctx.textAlign = 'center';
        data.forEach((_, i) => {
            const x = padL + (i / (data.length - 1)) * plotW;
            if (i % 2 === 0) ctx.fillText(labels[i], x, H - 8);
        });

        // Gradient fill
        const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
        grad.addColorStop(0, 'rgba(13, 148, 136, 0.12)');
        grad.addColorStop(1, 'rgba(13, 148, 136, 0.01)');

        ctx.beginPath();
        data.forEach((v, i) => {
            const x = padL + (i / (data.length - 1)) * plotW;
            const y = padT + plotH * (1 - v / maxVal);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.lineTo(padL + plotW, padT + plotH);
        ctx.lineTo(padL, padT + plotH);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Line
        ctx.beginPath();
        data.forEach((v, i) => {
            const x = padL + (i / (data.length - 1)) * plotW;
            const y = padT + plotH * (1 - v / maxVal);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.strokeStyle = '#0d9488';
        ctx.lineWidth = 2.5;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        ctx.stroke();

        // Dots
        data.forEach((v, i) => {
            const x = padL + (i / (data.length - 1)) * plotW;
            const y = padT + plotH * (1 - v / maxVal);
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, Math.PI * 2);
            ctx.fillStyle = '#0d9488';
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x, y, 2, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
        });
    }

    // ===================== Donut Chart =====================
    function drawDonutChart() {
        const canvas = document.getElementById('donut-chart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = 220 * dpr;
        ctx.scale(dpr, dpr);
        const W = rect.width, H = 220;

        const data = [
            { label: 'No DR (Grade 0)', value: 5842, color: '#10b981' },
            { label: 'Mild (Grade 1)', value: 1654, color: '#3b82f6' },
            { label: 'Moderate (Grade 2)', value: 823, color: '#f59e0b' },
            { label: 'Severe (Grade 3)', value: 312, color: '#f97316' },
            { label: 'Proliferative (Grade 4)', value: 147, color: '#ef4444' },
        ];

        const total = data.reduce((s, d) => s + d.value, 0);
        const cx = W / 2, cy = H / 2;
        const outerR = Math.min(cx, cy) - 20;
        const innerR = outerR * 0.62;

        let startAngle = -Math.PI / 2;
        data.forEach(d => {
            const sliceAngle = (d.value / total) * Math.PI * 2;
            ctx.beginPath();
            ctx.arc(cx, cy, outerR, startAngle, startAngle + sliceAngle);
            ctx.arc(cx, cy, innerR, startAngle + sliceAngle, startAngle, true);
            ctx.closePath();
            ctx.fillStyle = d.color;
            ctx.fill();
            startAngle += sliceAngle;
        });

        // Center text
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 1.4rem Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(total.toLocaleString(), cx, cy - 8);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '0.7rem Inter, sans-serif';
        ctx.fillText('Total Cases', cx, cy + 14);

        // Legend
        const legend = document.getElementById('donut-legend');
        legend.innerHTML = data.map(d => `
            <div class="legend-item">
                <span class="legend-dot" style="background: ${d.color};"></span>
                ${d.label} (${((d.value / total) * 100).toFixed(1)}%)
            </div>
        `).join('');
    }

    // ===================== Heatmap Overlay =====================
    function drawHeatmap() {
        const scanImg = document.getElementById('scan-image');
        const canvas2 = document.getElementById('heatmap-canvas-2');
        if (!canvas2 || !scanImg) return;

        // Wait for image load
        function render() {
            const wrap = canvas2.parentElement.parentElement;
            const w = wrap.offsetWidth || 300;
            const h = wrap.offsetHeight || 300;
            canvas2.width = w;
            canvas2.height = h;
            const ctx = canvas2.getContext('2d');

            // Create a heatmap with simulated lesion hotspots
            const hotspots = [
                { x: 0.55, y: 0.45, r: 0.08, intensity: 0.9 },  // Near macula
                { x: 0.6, y: 0.38, r: 0.05, intensity: 0.7 },
                { x: 0.48, y: 0.52, r: 0.06, intensity: 0.6 },
                { x: 0.65, y: 0.55, r: 0.04, intensity: 0.8 },
                { x: 0.42, y: 0.42, r: 0.03, intensity: 0.5 },
                { x: 0.58, y: 0.6, r: 0.04, intensity: 0.65 },
                { x: 0.35, y: 0.48, r: 0.07, intensity: 0.4 },
                { x: 0.7, y: 0.42, r: 0.05, intensity: 0.55 },
            ];

            // Build heatmap data
            const imgData = ctx.createImageData(w, h);
            for (let py = 0; py < h; py++) {
                for (let px = 0; px < w; px++) {
                    let heat = 0;
                    for (const hs of hotspots) {
                        const dx = (px / w - hs.x);
                        const dy = (py / h - hs.y);
                        const dist2 = dx * dx + dy * dy;
                        heat += hs.intensity * Math.exp(-dist2 / (2 * hs.r * hs.r));
                    }
                    heat = Math.min(1, heat);

                    // Jet colormap
                    let r, g, b;
                    if (heat < 0.25) { r = 0; g = heat * 4; b = 1; }
                    else if (heat < 0.5) { r = 0; g = 1; b = 1 - (heat - 0.25) * 4; }
                    else if (heat < 0.75) { r = (heat - 0.5) * 4; g = 1; b = 0; }
                    else { r = 1; g = 1 - (heat - 0.75) * 4; b = 0; }

                    const alpha = heat * 0.55;
                    const i = (py * w + px) * 4;
                    imgData.data[i] = r * 255;
                    imgData.data[i + 1] = g * 255;
                    imgData.data[i + 2] = b * 255;
                    imgData.data[i + 3] = alpha * 255;
                }
            }
            ctx.putImageData(imgData, 0, 0);
        }

        if (scanImg.complete) {
            render();
        } else {
            scanImg.addEventListener('load', render);
        }
        // Also handle resize
        window.addEventListener('resize', render);
    }

    // ===================== Animated Counters =====================
    function animateCounter(el, target) {
        const isPercent = target.includes('%');
        const isComma = target.includes(',');
        const numericStr = target.replace(/[,%]/g, '');
        const numTarget = parseFloat(numericStr);
        const duration = 1500;
        const start = performance.now();

        function tick(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
            const current = numTarget * eased;

            if (isPercent) {
                el.textContent = current.toFixed(1) + '%';
            } else if (isComma) {
                el.textContent = Math.round(current).toLocaleString();
            } else {
                el.textContent = Math.round(current);
            }

            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    }

    // Animate stats on load
    setTimeout(() => {
        animateCounter(document.getElementById('stat-patients'), '12,847');
        animateCounter(document.getElementById('stat-screenings'), '8,234');
        animateCounter(document.getElementById('stat-highrisk'), '1,482');
        animateCounter(document.getElementById('stat-accuracy'), '96.7%');
    }, 300);

    // ===================== Chart Tab Interaction =====================
    document.querySelectorAll('.chart-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            drawTrendsChart(); // Redraw with same data for now
        });
    });

    // ===================== File Upload (Screening Page) =====================
    const uploadZone = document.getElementById('screening-upload-zone');
    const uploadInput = document.getElementById('screening-file-input');

    if (uploadZone) {
        uploadZone.addEventListener('click', () => uploadInput.click());
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '#0d9488';
            uploadZone.style.background = '#f0fdfa';
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.style.borderColor = '';
            uploadZone.style.background = '';
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '';
            uploadZone.style.background = '';
            if (e.dataTransfer.files[0]) {
                showToast('Image uploaded — starting AI analysis...', 'success');
                // Navigate to dashboard to show screening
                document.querySelector('[data-page="dashboard"]').click();
            }
        });
        uploadInput.addEventListener('change', () => {
            if (uploadInput.files[0]) {
                showToast('Image uploaded — starting AI analysis...', 'success');
                document.querySelector('[data-page="dashboard"]').click();
            }
        });
    }

    // ===================== Initialize Charts =====================
    // Use setTimeout to ensure layout is settled
    setTimeout(() => {
        drawTrendsChart();
        drawDonutChart();
        drawHeatmap();
    }, 100);

    window.addEventListener('resize', () => {
        drawTrendsChart();
        drawDonutChart();
    });

    // ===================== Confidence Animation =====================
    setTimeout(() => {
        document.querySelectorAll('.confidence-fill').forEach(el => {
            const width = el.style.width;
            el.style.width = '0%';
            setTimeout(() => { el.style.width = width; }, 50);
        });
    }, 500);

});
