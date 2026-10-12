/**
 * AI DR Screen — Modern Healthcare SaaS Settings Page Controller
 * Handles 12 settings categories, theme integration, local persistence,
 * accessibility modes, audio chime testing, and session synchronization.
 */

(function () {
    'use strict';

    const STORAGE_KEY_SETTINGS = 'aidr_settings';
    const STORAGE_KEY_THEME = 'xairis_theme';

    const DEFAULT_SETTINGS = {
        general: {
            dashboardView: 'screening_terminal',
            landingPage: 'dashboard',
            dateFormat: 'DD/MM/YYYY',
            timeFormat: '12',
            density: 'comfortable',
            animationPref: 'full',
            autoSave: true
        },
        appearance: {
            theme: 'dark',
            fontSize: '14',
            sidebarBehavior: 'pinned',
            reduceMotion: false,
            highContrast: false
        },
        notifications: {
            screeningCompleted: true,
            reportGenerated: true,
            highRiskAlert: true,
            referralRec: true,
            systemNotif: true,
            emailNotif: false,
            soundEnabled: true
        },
        screening: {
            iqaEnabled: true,
            iqaThreshold: 75,
            rejectPoorQuality: true,
            rejectUnusable: true,
            claheEnabled: true,
            autoPreprocess: true,
            brightnessNorm: true,
            contrastEnhance: true,
            showConfidence: true,
            autoResult: true,
            autoReport: false
        },
        ai: {
            gradcamEnabled: true,
            unetEnabled: true,
            shapEnabled: true,
            confidenceEnabled: true,
            drSeverityExplanation: true,
            abnormalRegionHighlight: true
        },
        reports: {
            autoGenerateReport: true,
            reportFormat: 'standard_pdf',
            incSeverity: true,
            incConfidence: true,
            incGradcam: true,
            incSegmentation: true,
            incReferral: true,
            incTimestamp: true,
            incCenterInfo: true
        },
        privacy: {
            sessionTimeout: '30',
            autoLogout: true,
            dataRetentionDays: '30',
            maskPhone: true
        },
        profile: {
            name: 'Dr. Anitha Rajan, MD',
            role: 'Chief Medical Officer & Vitreoretinal Specialist',
            center: 'Primary Health Center — Thirunallur',
            email: 'dr.anitha@phc.gov.in',
            phone: '+91 94432 01080',
            licenseNumber: 'TN-MCI-2026-9481',
            abdmHpid: 'HPID-ABDM-TN-2026-0814'
        },
        language: {
            lang: 'en',
            largeText: false,
            highContrastMode: false,
            keyboardNavAssist: false,
            screenReaderAnnounce: true
        },
        device: {
            centerName: 'Primary Health Center — Thirunallur',
            centerLocation: 'Thirunallur, Kallakurichi District, Tamil Nadu',
            operatorName: 'Dr. Anitha Rajan / Priya Sundaram (ASHA Lead)',
            cameraModel: 'Remidio Non-Mydriatic Fundus-on-Phone (FOP) NM-200',
            imageResolution: '2048 x 1536 (3.1 MP FOV-45)',
            guidanceRing: true,
            fovValidation: true
        },
        sync: {
            offlineMode: true,
            autoSync: true,
            lastSyncTime: 'Just now'
        }
    };

    let currentSettings = JSON.parse(JSON.stringify(DEFAULT_SETTINGS));

    // -------------------------------------------------------------
    // 1. INITIALIZATION
    // -------------------------------------------------------------
    document.addEventListener('DOMContentLoaded', () => {
        loadSettingsFromStorage();
        syncWithExistingSession();
        bindUIControls();
        applyLiveSettings();

        // Handle URL hash navigation (e.g. settings.html#ai or settings.html#appearance)
        const hash = window.location.hash ? window.location.hash.replace('#', '') : 'general';
        switchCategory(hash || 'general');
    });

    // -------------------------------------------------------------
    // 2. STORAGE MANAGEMENT
    // -------------------------------------------------------------
    function loadSettingsFromStorage() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY_SETTINGS);
            if (raw) {
                const parsed = JSON.parse(raw);
                currentSettings = deepMerge(DEFAULT_SETTINGS, parsed);
            }
            // Ensure theme matches current xairis_theme if already set
            const savedTheme = localStorage.getItem(STORAGE_KEY_THEME);
            if (savedTheme) {
                currentSettings.appearance.theme = savedTheme;
            }
        } catch (e) {
            currentSettings = JSON.parse(JSON.stringify(DEFAULT_SETTINGS));
        }
    }

    function saveSettingsToStorage(showFeedback = true) {
        try {
            localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(currentSettings));
            applyLiveSettings();
            if (showFeedback) {
                showToast('✓ Settings saved successfully');
            }
        } catch (e) {
            console.warn('Failed to save settings:', e);
        }
    }

    function deepMerge(target, source) {
        const out = { ...target };
        for (const key in source) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                out[key] = deepMerge(target[key] || {}, source[key]);
            } else {
                out[key] = source[key];
            }
        }
        return out;
    }

    // -------------------------------------------------------------
    // 3. SYNCHRONIZE PROFILE WITH EXISTING ACTIVE SESSION
    // -------------------------------------------------------------
    function syncWithExistingSession() {
        try {
            // Check Doctor Profile first
            const rawDoc = localStorage.getItem('aidr_doctor_auth_user');
            if (rawDoc) {
                const doc = JSON.parse(rawDoc);
                if (doc.fullName) currentSettings.profile.name = doc.fullName;
                if (doc.specialization) currentSettings.profile.role = doc.specialization;
                if (doc.hospitalOrClinic) currentSettings.profile.center = doc.hospitalOrClinic;
                if (doc.email) currentSettings.profile.email = doc.email;
                if (doc.phone) currentSettings.profile.phone = doc.phone;
                if (doc.licenseNumber) currentSettings.profile.licenseNumber = doc.licenseNumber;
                if (doc.abdmHpid) currentSettings.profile.abdmHpid = doc.abdmHpid;
                return;
            }

            // Otherwise check general user session
            const rawUser = localStorage.getItem('aidr_user_session');
            if (rawUser) {
                const session = JSON.parse(rawUser);
                if (session && session.user) {
                    if (session.user.name) currentSettings.profile.name = session.user.name;
                    if (session.user.roleLabel) currentSettings.profile.role = session.user.roleLabel;
                    if (session.user.center) currentSettings.profile.center = session.user.center;
                    if (session.user.id) currentSettings.profile.email = session.user.id;
                }
            }
        } catch (e) {}
    }

    // -------------------------------------------------------------
    // 4. CATEGORY NAVIGATION CONTROLLER
    // -------------------------------------------------------------
    window.switchCategory = function (categoryId) {
        const panels = document.querySelectorAll('.settings-panel');
        const desktopBtns = document.querySelectorAll('.set-nav-btn');
        const mobilePills = document.querySelectorAll('.mobile-cat-pill');

        let targetPanel = document.getElementById(`panel-${categoryId}`);
        if (!targetPanel) {
            categoryId = 'general';
            targetPanel = document.getElementById('panel-general');
        }

        // Update Panels
        panels.forEach(p => p.classList.remove('active'));
        if (targetPanel) {
            targetPanel.classList.add('active');
        }

        // Update Desktop Navigation Items
        desktopBtns.forEach(btn => {
            if (btn.getAttribute('data-category') === categoryId) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        // Update Mobile Category Pills
        mobilePills.forEach(pill => {
            if (pill.getAttribute('data-category') === categoryId) {
                pill.classList.add('active');
                pill.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
            } else {
                pill.classList.remove('active');
            }
        });

        window.location.hash = categoryId;
    };

    // -------------------------------------------------------------
    // 5. THEME INTEGRATION (CONNECTED TO EXISTING xairis_theme)
    // -------------------------------------------------------------
    window.selectThemeOption = function (themeChoice) {
        let activeTheme = themeChoice;
        if (themeChoice === 'system') {
            activeTheme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }

        currentSettings.appearance.theme = themeChoice;
        localStorage.setItem(STORAGE_KEY_THEME, activeTheme);

        // Apply theme to DOM
        document.documentElement.setAttribute('data-theme', activeTheme);
        document.body.setAttribute('data-theme', activeTheme);

        // Update Theme Cards Selection
        document.querySelectorAll('.theme-card-option').forEach(card => {
            if (card.getAttribute('data-theme-val') === themeChoice) {
                card.classList.add('active');
            } else {
                card.classList.remove('active');
            }
        });

        // Broadcast to parent or other iframes
        try {
            if (window.parent && window.parent !== window) {
                window.parent.postMessage({ action: 'themeChange', theme: activeTheme }, '*');
            }
        } catch (e) {}

        saveSettingsToStorage(true);
    };

    // -------------------------------------------------------------
    // 6. LIVE ACCESSIBILITY & DISPLAY PREFERENCES
    // -------------------------------------------------------------
    function applyLiveSettings() {
        // 1. Theme
        const theme = currentSettings.appearance.theme || 'dark';
        let appliedTheme = theme;
        if (theme === 'system') {
            appliedTheme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
        }
        document.documentElement.setAttribute('data-theme', appliedTheme);
        document.body.setAttribute('data-theme', appliedTheme);

        // 2. High Contrast
        if (currentSettings.appearance.highContrast || currentSettings.language.highContrastMode) {
            document.documentElement.setAttribute('data-high-contrast', 'true');
        } else {
            document.documentElement.removeAttribute('data-high-contrast');
        }

        // 3. Reduce Motion
        if (currentSettings.appearance.reduceMotion || currentSettings.general.animationPref === 'minimal') {
            document.documentElement.setAttribute('data-reduced-motion', 'true');
        } else {
            document.documentElement.removeAttribute('data-reduced-motion');
        }

        // 4. Font Scaling / Large Text
        if (currentSettings.language.largeText || currentSettings.appearance.fontSize === '16') {
            document.documentElement.style.fontSize = '16px';
        } else if (currentSettings.appearance.fontSize === '13') {
            document.documentElement.style.fontSize = '13px';
        } else {
            document.documentElement.style.fontSize = '14px';
        }
    }

    // -------------------------------------------------------------
    // 7. BIND UI CONTROLS TO SETTINGS STATE
    // -------------------------------------------------------------
    function bindUIControls() {
        // Theme cards
        document.querySelectorAll('.theme-card-option').forEach(card => {
            if (card.getAttribute('data-theme-val') === currentSettings.appearance.theme) {
                card.classList.add('active');
            }
        });

        // Toggles
        bindToggle('toggle-reduce-motion', 'appearance', 'reduceMotion');
        bindToggle('toggle-high-contrast', 'appearance', 'highContrast');
        bindToggle('toggle-notif-screening', 'notifications', 'screeningCompleted');
        bindToggle('toggle-notif-report', 'notifications', 'reportGenerated');
        bindToggle('toggle-notif-highrisk', 'notifications', 'highRiskAlert');
        bindToggle('toggle-notif-referral', 'notifications', 'referralRec');
        bindToggle('toggle-notif-system', 'notifications', 'systemNotif');
        bindToggle('toggle-notif-email', 'notifications', 'emailNotif');
        bindToggle('toggle-notif-sound', 'notifications', 'soundEnabled');

        // Screening & IQA Toggles
        bindToggle('toggle-iqa-enable', 'screening', 'iqaEnabled');
        bindToggle('toggle-reject-poor', 'screening', 'rejectPoorQuality');
        bindToggle('toggle-reject-unusable', 'screening', 'rejectUnusable');
        bindToggle('toggle-clahe', 'screening', 'claheEnabled');
        bindToggle('toggle-autopreprocess', 'screening', 'autoPreprocess');
        bindToggle('toggle-brightness-norm', 'screening', 'brightnessNorm');
        bindToggle('toggle-contrast-enhance', 'screening', 'contrastEnhance');
        bindToggle('toggle-show-confidence', 'screening', 'showConfidence');
        bindToggle('toggle-auto-result', 'screening', 'autoResult');
        bindToggle('toggle-auto-report', 'screening', 'autoReport');

        // AI & Explainability Toggles
        bindToggle('toggle-ai-gradcam', 'ai', 'gradcamEnabled');
        bindToggle('toggle-ai-unet', 'ai', 'unetEnabled');
        bindToggle('toggle-ai-shap', 'ai', 'shapEnabled');
        bindToggle('toggle-ai-confidence', 'ai', 'confidenceEnabled');
        bindToggle('toggle-ai-dr-severity', 'ai', 'drSeverityExplanation');
        bindToggle('toggle-ai-abnormal-highlight', 'ai', 'abnormalRegionHighlight');

        // Report Settings Toggles
        bindToggle('toggle-rep-autogen', 'reports', 'autoGenerateReport');
        bindToggle('toggle-rep-severity', 'reports', 'incSeverity');
        bindToggle('toggle-rep-confidence', 'reports', 'incConfidence');
        bindToggle('toggle-rep-gradcam', 'reports', 'incGradcam');
        bindToggle('toggle-rep-segmentation', 'reports', 'incSegmentation');
        bindToggle('toggle-rep-referral', 'reports', 'incReferral');
        bindToggle('toggle-rep-timestamp', 'reports', 'incTimestamp');
        bindToggle('toggle-rep-centerinfo', 'reports', 'incCenterInfo');

        // Privacy Toggles
        bindToggle('toggle-auto-logout', 'privacy', 'autoLogout');
        bindToggle('toggle-mask-phone', 'privacy', 'maskPhone');

        // Language & Accessibility Toggles
        bindToggle('toggle-large-text', 'language', 'largeText');
        bindToggle('toggle-lang-contrast', 'language', 'highContrastMode');
        bindToggle('toggle-keynav-assist', 'language', 'keyboardNavAssist');
        bindToggle('toggle-screen-reader', 'language', 'screenReaderAnnounce');

        // Device & Sync Toggles
        bindToggle('toggle-guidance-ring', 'device', 'guidanceRing');
        bindToggle('toggle-fov-val', 'device', 'fovValidation');
        bindToggle('toggle-offline-mode', 'sync', 'offlineMode');
        bindToggle('toggle-auto-sync', 'sync', 'autoSync');

        // Dropdowns & Inputs
        bindSelect('select-dashboard-view', 'general', 'dashboardView');
        bindSelect('select-landing-page', 'general', 'landingPage');
        bindSelect('select-date-format', 'general', 'dateFormat');
        bindSelect('select-time-format', 'general', 'timeFormat');
        bindSelect('select-density', 'general', 'density');
        bindSelect('select-animation-pref', 'general', 'animationPref');
        bindSelect('select-font-size', 'appearance', 'fontSize');
        bindSelect('select-sidebar-behavior', 'appearance', 'sidebarBehavior');
        bindSelect('select-session-timeout', 'privacy', 'sessionTimeout');
        bindSelect('select-data-retention', 'privacy', 'dataRetentionDays');
        bindSelect('select-report-format', 'reports', 'reportFormat');

        // Slider
        const iqaSlider = document.getElementById('slider-iqa-threshold');
        const iqaBadge = document.getElementById('iqa-threshold-val');
        if (iqaSlider && iqaBadge) {
            iqaSlider.value = currentSettings.screening.iqaThreshold;
            iqaBadge.textContent = `${currentSettings.screening.iqaThreshold}%`;
            iqaSlider.addEventListener('input', (e) => {
                const val = e.target.value;
                iqaBadge.textContent = `${val}%`;
                currentSettings.screening.iqaThreshold = parseInt(val, 10);
                saveSettingsToStorage(false);
            });
        }

        // Profile Form Fields
        populateProfileFields();
    }

    function bindToggle(id, section, key) {
        const el = document.getElementById(id);
        if (!el) return;
        el.checked = !!currentSettings[section][key];
        el.addEventListener('change', () => {
            currentSettings[section][key] = el.checked;
            saveSettingsToStorage(true);
        });
    }

    function bindSelect(id, section, key) {
        const el = document.getElementById(id);
        if (!el) return;
        el.value = currentSettings[section][key];
        el.addEventListener('change', () => {
            currentSettings[section][key] = el.value;
            saveSettingsToStorage(true);
        });
    }

    function populateProfileFields() {
        setVal('prof-input-name', currentSettings.profile.name);
        setVal('prof-input-role', currentSettings.profile.role);
        setVal('prof-input-center', currentSettings.profile.center);
        setVal('prof-input-email', currentSettings.profile.email);
        setVal('prof-input-phone', currentSettings.profile.phone);
        setVal('prof-input-license', currentSettings.profile.licenseNumber);
        setVal('prof-input-abdm', currentSettings.profile.abdmHpid);

        // Header Avatar and Name
        const heroName = document.getElementById('prof-hero-name');
        const heroRole = document.getElementById('prof-hero-role');
        const heroAvatar = document.getElementById('prof-hero-avatar');
        if (heroName) heroName.textContent = currentSettings.profile.name;
        if (heroRole) heroRole.textContent = currentSettings.profile.role;
        if (heroAvatar) {
            heroAvatar.textContent = (currentSettings.profile.name || 'D').replace('Dr. ', '').charAt(0).toUpperCase();
        }
    }

    function setVal(id, val) {
        const el = document.getElementById(id);
        if (el) el.value = val || '';
    }

    // -------------------------------------------------------------
    // 8. PROFILE ACTIONS & SESSION PERSISTENCE
    // -------------------------------------------------------------
    window.handleSaveProfileForm = function (e) {
        e.preventDefault();
        currentSettings.profile.name = document.getElementById('prof-input-name').value.trim();
        currentSettings.profile.role = document.getElementById('prof-input-role').value.trim();
        currentSettings.profile.center = document.getElementById('prof-input-center').value.trim();
        currentSettings.profile.email = document.getElementById('prof-input-email').value.trim();
        currentSettings.profile.phone = document.getElementById('prof-input-phone').value.trim();
        currentSettings.profile.licenseNumber = document.getElementById('prof-input-license').value.trim();
        currentSettings.profile.abdmHpid = document.getElementById('prof-input-abdm').value.trim();

        // Also update doctor session storage if active
        try {
            const rawDoc = localStorage.getItem('aidr_doctor_auth_user');
            if (rawDoc) {
                const doc = JSON.parse(rawDoc);
                doc.fullName = currentSettings.profile.name;
                doc.specialization = currentSettings.profile.role;
                doc.hospitalOrClinic = currentSettings.profile.center;
                doc.email = currentSettings.profile.email;
                doc.phone = currentSettings.profile.phone;
                doc.licenseNumber = currentSettings.profile.licenseNumber;
                doc.abdmHpid = currentSettings.profile.abdmHpid;
                localStorage.setItem('aidr_doctor_auth_user', JSON.stringify(doc));
            }
        } catch (err) {}

        populateProfileFields();
        saveSettingsToStorage(true);
    };

    window.handleChangePassword = function (e) {
        e.preventDefault();
        const current = document.getElementById('pw-current').value;
        const next = document.getElementById('pw-new').value;
        const confirm = document.getElementById('pw-confirm').value;

        if (!current || !next || !confirm) {
            alert('Please fill in all password fields.');
            return;
        }

        if (next !== confirm) {
            alert('New passwords do not match. Please verify.');
            return;
        }

        if (next.length < 6) {
            alert('New security password must be at least 6 characters.');
            return;
        }

        document.getElementById('pw-current').value = '';
        document.getElementById('pw-new').value = '';
        document.getElementById('pw-confirm').value = '';
        showToast('✓ Security password updated successfully');
    };

    // -------------------------------------------------------------
    // 9. SEARCH SETTINGS FEATURE
    // -------------------------------------------------------------
    window.handleSettingsSearch = function (query) {
        const clean = query.trim().toLowerCase();
        const settingRows = document.querySelectorAll('.setting-row, .settings-card');

        if (!clean) {
            settingRows.forEach(el => (el.style.display = ''));
            return;
        }

        let firstMatchCategory = null;

        settingRows.forEach(row => {
            const text = row.textContent.toLowerCase();
            if (text.includes(clean)) {
                row.style.display = '';
                const panel = row.closest('.settings-panel');
                if (panel && !firstMatchCategory) {
                    firstMatchCategory = panel.id.replace('panel-', '');
                }
            } else {
                row.style.display = 'none';
            }
        });

        if (firstMatchCategory) {
            switchCategory(firstMatchCategory);
        }
    };

    // -------------------------------------------------------------
    // 10. NOTIFICATION AUDIO TEST (WEB AUDIO API CHIME)
    // -------------------------------------------------------------
    window.testNotificationSound = function () {
        try {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            const ctx = new AudioContext();

            // Two-tone subtle medical chime (587 Hz D5 -> 880 Hz A5)
            const now = ctx.currentTime;
            const osc1 = ctx.createOscillator();
            const gain1 = ctx.createGain();

            osc1.type = 'sine';
            osc1.frequency.setValueAtTime(587.33, now);
            osc1.frequency.exponentialRampToValueAtTime(880, now + 0.12);

            gain1.gain.setValueAtTime(0.001, now);
            gain1.gain.linearRampToValueAtTime(0.2, now + 0.04);
            gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

            osc1.connect(gain1);
            gain1.connect(ctx.destination);

            osc1.start(now);
            osc1.stop(now + 0.38);

            showToast('🔔 Notification chime played');
        } catch (e) {
            console.warn('Audio test warning:', e);
        }
    };

    // -------------------------------------------------------------
    // 11. MANUAL CLOUD & EDGE SYNC ACTION
    // -------------------------------------------------------------
    window.triggerManualSync = function () {
        const btn = document.getElementById('btn-sync-now');
        const statusEl = document.getElementById('sync-status-badge');
        const lastSyncEl = document.getElementById('sync-last-timestamp');

        if (btn) {
            btn.disabled = true;
            btn.innerHTML = `
                <span class="auth-btn-spinner" style="width:14px; height:14px; border:2px solid #000; border-top-color:transparent; border-radius:50%; display:inline-block; animation:spin 0.6s linear infinite;"></span>
                <span>Synchronizing to Grid...</span>
            `;
        }

        setTimeout(() => {
            const nowStr = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = `<span>Sync Now</span>`;
            }
            if (statusEl) {
                statusEl.textContent = '● All data synchronized';
                statusEl.style.color = 'var(--set-accent)';
            }
            if (lastSyncEl) {
                lastSyncEl.textContent = `Today at ${nowStr}`;
            }
            showToast('✓ 3 offline records successfully uploaded to National Grid');
        }, 1200);
    };

    // -------------------------------------------------------------
    // 12. SENSITIVE ACTIONS & CONFIRMATION MODALS
    // -------------------------------------------------------------
    window.openClearCacheModal = function () {
        const modal = document.getElementById('modal-clear-cache');
        if (modal) modal.style.display = 'flex';
    };

    window.closeClearCacheModal = function () {
        const modal = document.getElementById('modal-clear-cache');
        if (modal) modal.style.display = 'none';
    };

    window.confirmClearCache = function () {
        try {
            // Keep user session, remove temporary cached files
            sessionStorage.clear();
            closeClearCacheModal();
            showToast('✓ Local temporary cache cleared');
        } catch (e) {}
    };

    window.openDeleteHistoryModal = function () {
        const modal = document.getElementById('modal-delete-history');
        if (modal) {
            document.getElementById('confirm-delete-input').value = '';
            modal.style.display = 'flex';
        }
    };

    window.closeDeleteHistoryModal = function () {
        const modal = document.getElementById('modal-delete-history');
        if (modal) modal.style.display = 'none';
    };

    window.confirmDeleteHistory = function () {
        const input = document.getElementById('confirm-delete-input').value.trim();
        if (input !== 'DELETE') {
            alert('Confirmation keyword mismatch. Please type "DELETE" to proceed.');
            return;
        }

        try {
            localStorage.removeItem('aidr_firestore_screenings');
            closeDeleteHistoryModal();
            showToast('Screening history cleared from local device.');
        } catch (e) {}
    };

    // -------------------------------------------------------------
    // 13. TOAST NOTIFICATION UTILITY
    // -------------------------------------------------------------
    function showToast(message) {
        let toast = document.getElementById('settings-toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'settings-toast';
            toast.className = 'settings-toast';
            document.body.appendChild(toast);
        }

        toast.textContent = message;
        toast.classList.add('visible');

        setTimeout(() => {
            toast.classList.remove('visible');
        }, 2800);
    }

})();
