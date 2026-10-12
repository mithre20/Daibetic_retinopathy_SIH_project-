/**
 * XAIRIS — Clinical Staff Authentication & Edge Terminal Logic
 * Handles medical authentication, 4-digit PIN auto-validation,
 * session persistence, and real-time theme synchronization.
 */

// -------------------------------------------------------------
// 1. THEME SYNCHRONIZATION (STANDALONE & IFRAME INTEGRATION)
// -------------------------------------------------------------
function applyLoginTheme(theme) {
    const active = theme || localStorage.getItem('xairis_theme') || 'light';
    document.documentElement.setAttribute('data-theme', active);
    document.body.setAttribute('data-theme', active);
}

// Listen for theme changes from parent portal
window.addEventListener('message', (e) => {
    if (e.data && e.data.action === 'themeChange') {
        applyLoginTheme(e.data.theme);
    }
});

// -------------------------------------------------------------
// 2. DEMO ACCOUNTS REPOSITORY
// -------------------------------------------------------------
const DEMO_ACCOUNTS = {
    doctor: {
        id: 'dr.anitha@phc.gov.in',
        password: 'vision2026',
        name: 'Dr. Anitha Rajan, MD',
        role: 'doctor',
        roleLabel: 'Chief Medical Officer',
        center: 'Primary Health Center — Thirunallur',
        pin: '2026'
    },
    asha: {
        id: 'priya.asha@ruralhealth.org',
        password: 'fieldworker2026',
        name: 'Priya Sundaram',
        role: 'phc',
        roleLabel: 'ASHA Field Screening Lead',
        center: 'Kallakurichi Rural Mobile Unit',
        pin: '1080'
    }
};

// -------------------------------------------------------------
// 3. INITIALIZATION
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    applyLoginTheme();
    initPinInputs();
    checkExistingSession();

    // Check query params (e.g. login.html?role=doctor)
    try {
        const params = new URLSearchParams(window.location.search);
        const role = params.get('role');
        if (role === 'doctor') {
            quickFill('doctor');
        } else if (role === 'phc' || role === 'asha') {
            quickFill('asha');
        }
    } catch (e) {}
});

// -------------------------------------------------------------
// 4. QUICK DEMO FILL WITH VISUAL FEEDBACK
// -------------------------------------------------------------
window.quickFill = function(type) {
    const account = DEMO_ACCOUNTS[type];
    if (!account) return;

    const idInput = document.getElementById('login-id');
    const pwInput = document.getElementById('login-password');

    if (idInput && pwInput) {
        idInput.value = account.id;
        pwInput.value = account.password;

        // Visual flash highlight
        [idInput, pwInput].forEach(el => {
            el.classList.add('input-highlight-pulse');
            setTimeout(() => el.classList.remove('input-highlight-pulse'), 800);
        });
    }

    // Set role toggle chip
    const chips = document.querySelectorAll('.role-toggle-chip, .role-chip');
    chips.forEach(chip => {
        const radio = chip.querySelector('input[type="radio"]');
        if (radio && radio.value === account.role) {
            chips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            radio.checked = true;
        }
    });

    // Populate PIN digits
    const digits = document.querySelectorAll('.pin-digit');
    if (account.pin && digits.length === 4) {
        account.pin.split('').forEach((d, i) => {
            if (digits[i]) {
                digits[i].value = d;
                digits[i].classList.add('input-highlight-pulse');
                setTimeout(() => digits[i].classList.remove('input-highlight-pulse'), 800);
            }
        });
    }

    // Update submit button text based on role
    const btnLabel = document.getElementById('btn-login-label');
    if (btnLabel) {
        btnLabel.textContent = account.role === 'doctor'
            ? 'Authenticate & Open Doctor Profile'
            : 'Authenticate & Open Screening Terminal';
    }

    showToast(`Loaded demo profile: ${account.name} (${account.roleLabel})`, 'info');
};

// -------------------------------------------------------------
// 6. ROLE SELECTION
// -------------------------------------------------------------
window.selectRole = function(element, role) {
    const chips = document.querySelectorAll('.role-toggle-chip, .role-chip');
    chips.forEach(c => c.classList.remove('active'));
    element.classList.add('active');

    const radio = element.querySelector('input[type="radio"]');
    if (radio) {
        radio.checked = true;
    }

    const btnLabel = document.getElementById('btn-login-label');
    if (btnLabel) {
        btnLabel.textContent = role === 'doctor'
            ? 'Authenticate & Open Doctor Profile'
            : 'Authenticate & Open Screening Terminal';
    }
};

// -------------------------------------------------------------
// 7. PASSWORD VISIBILITY TOGGLER WITH SVG SWITCH
// -------------------------------------------------------------
window.togglePasswordVisibility = function(inputId) {
    const input = document.getElementById(inputId);
    const btn = document.getElementById('btn-toggle-pw');
    if (!input) return;

    const isPassword = input.type === 'password';
    input.type = isPassword ? 'text' : 'password';

    if (btn) {
        const eyeOpen = btn.querySelector('.eye-open-icon');
        const eyeClosed = btn.querySelector('.eye-closed-icon');
        if (eyeOpen && eyeClosed) {
            eyeOpen.style.display = isPassword ? 'none' : 'block';
            eyeClosed.style.display = isPassword ? 'block' : 'none';
        }
    }
};

// -------------------------------------------------------------
// 8. 4-DIGIT PIN AUTO-FOCUS, PASTE & VALIDATION
// -------------------------------------------------------------
function initPinInputs() {
    const pinDigits = document.querySelectorAll('.pin-digit');
    if (!pinDigits.length) return;

    pinDigits.forEach((input, index) => {
        // Only allow numbers 0-9
        input.addEventListener('input', (e) => {
            const val = e.target.value.replace(/[^0-9]/g, '');
            e.target.value = val ? val.slice(-1) : '';

            if (e.target.value && index < pinDigits.length - 1) {
                pinDigits[index + 1].focus();
            }

            // Check if all 4 digits are filled
            const pin = Array.from(pinDigits).map(d => d.value).join('');
            if (pin.length === 4) {
                handlePinAuthentication(pin);
            }
        });

        // Keydown handling for Backspace, Arrow keys
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Backspace') {
                if (!input.value && index > 0) {
                    pinDigits[index - 1].focus();
                }
            } else if (e.key === 'ArrowLeft' && index > 0) {
                pinDigits[index - 1].focus();
            } else if (e.key === 'ArrowRight' && index < pinDigits.length - 1) {
                pinDigits[index + 1].focus();
            }
        });

        // Paste support: paste 4 digits at once
        input.addEventListener('paste', (e) => {
            e.preventDefault();
            const pasted = (e.clipboardData || window.clipboardData).getData('text');
            const cleanDigits = pasted.replace(/[^0-9]/g, '').slice(0, 4);

            if (cleanDigits.length > 0) {
                cleanDigits.split('').forEach((digit, i) => {
                    if (pinDigits[i]) {
                        pinDigits[i].value = digit;
                    }
                });

                const focusIndex = Math.min(cleanDigits.length, pinDigits.length - 1);
                pinDigits[focusIndex].focus();

                if (cleanDigits.length === 4) {
                    handlePinAuthentication(cleanDigits);
                }
            }
        });
    });
}

window.clearPinInputs = function() {
    const pinDigits = document.querySelectorAll('.pin-digit');
    pinDigits.forEach(d => {
        d.value = '';
        d.classList.remove('pin-error-shake');
    });
    if (pinDigits[0]) pinDigits[0].focus();
};

function handlePinAuthentication(pin) {
    const matched = Object.values(DEMO_ACCOUNTS).find(acc => acc.pin === pin);
    const pinDigits = document.querySelectorAll('.pin-digit');

    if (matched) {
        pinDigits.forEach(d => d.classList.add('pin-success-pulse'));
        showToast(`PIN Verified: Welcome back, ${matched.name}`, 'success');
        setTimeout(() => completeLogin(matched), 350);
    } else {
        showToast('Invalid 4-digit PIN. (Demo Doctor PIN: "2026", ASHA PIN: "1080")', 'error');
        pinDigits.forEach(d => {
            d.classList.add('pin-error-shake');
        });

        setTimeout(() => {
            pinDigits.forEach(d => {
                d.value = '';
                d.classList.remove('pin-error-shake');
            });
            if (pinDigits[0]) pinDigits[0].focus();
        }, 1100);
    }
}

// -------------------------------------------------------------
// 9. FORM LOGIN HANDLER
// -------------------------------------------------------------
window.handleLogin = function(event) {
    event.preventDefault();
    const btn = document.getElementById('btn-login-submit');
    const idInput = document.getElementById('login-id').value.trim();
    const pwInput = document.getElementById('login-password').value.trim();

    if (!btn) return;

    btn.disabled = true;
    btn.innerHTML = `
        <span class="auth-btn-spinner" aria-hidden="true"></span>
        <span>Authenticating ABDM Token...</span>
    `;

    setTimeout(() => {
        // Match demo or build authenticated session
        let account = Object.values(DEMO_ACCOUNTS).find(acc => acc.id.toLowerCase() === idInput.toLowerCase());
        if (!account) {
            const roleSelected = document.querySelector('input[name="login-role"]:checked')?.value || 'doctor';
            account = {
                id: idInput,
                name: idInput.includes('@') ? idInput.split('@')[0].replace('.', ' ').toUpperCase() : idInput,
                role: roleSelected,
                roleLabel: roleSelected === 'doctor' ? 'Medical Officer' : 'Field Worker',
                center: 'Primary Health Center',
                pin: '2026'
            };
        }

        showToast(`Access Verified: Welcome, ${account.name}`, 'success');
        completeLogin(account);
    }, 600);
};

// -------------------------------------------------------------
// 9. FORGOT PASSWORD HANDLER
// -------------------------------------------------------------
window.handleForgot = function(event) {
    event.preventDefault();
    showToast('Offline Recovery: Demo password is "vision2026" or PIN "2026"', 'info');
};

// -------------------------------------------------------------
// 12. CROSS-PAGE NAVIGATION UTILITY
// -------------------------------------------------------------
window.navigateTo = function(target) {
    try {
        if (window.parent && window.parent !== window) {
            if (typeof window.parent.switchPortalTab === 'function') {
                window.parent.switchPortalTab(target, true);
                return;
            }
            window.parent.postMessage({ action: 'switchView', target: target }, '*');
            return;
        }
    } catch (e) {}

    // Standalone fallback
    if (target === 'hero') {
        window.location.href = 'index.html#hero';
    } else if (target === 'dashboard') {
        window.location.href = 'index.html#dashboard';
    } else if (target === 'medivision') {
        window.location.href = 'index.html#medivision';
    } else if (target === 'pipeline') {
        window.location.href = 'index.html#pipeline';
    } else {
        window.location.href = 'index.html';
    }
};

window.logoutSession = function() {
    localStorage.removeItem('aidr_user_session');
    localStorage.removeItem('aidr_doctor_auth_user');
    try {
        if (window.parent && window.parent !== window) {
            window.parent.postMessage({ action: 'sessionUpdated', session: null }, '*');
            if (typeof window.parent.updatePortalUser === 'function') {
                window.parent.updatePortalUser();
            }
        }
    } catch (e) {}

    const note = document.getElementById('active-session-banner');
    if (note) note.remove();
    showToast('Signed out of current session.', 'info');
};

// -------------------------------------------------------------
// 13. COMPLETE LOGIN & REDIRECT
// -------------------------------------------------------------
function completeLogin(account) {
    const isDoctor = account.role === 'doctor' || 
                     account.role === 'CLINICAL_DOCTOR' || 
                     (account.id && (account.id.toLowerCase().includes('dr.') || account.id.toLowerCase().includes('doctor')));

    const sessionData = {
        authenticated: true,
        user: account,
        role: isDoctor ? 'doctor' : 'phc',
        loginTime: new Date().toISOString()
    };

    localStorage.setItem('aidr_user_session', JSON.stringify(sessionData));

    if (isDoctor) {
        // Clinical Doctor Profile & Auth State
        const doctorProfile = {
            uid: account.uid || 'doc_anitha',
            doctorId: account.doctorId || 'DOC-TN-2026-9481',
            fullName: account.name || 'Dr. Anitha Rajan, MD',
            email: account.id || 'dr.anitha@phc.gov.in',
            specialization: account.specialization || 'Chief Medical Officer & Vitreoretinal Specialist',
            licenseNumber: account.regNo || account.licenseNumber || 'TN-MCI-2026-9481',
            abdmHpid: 'HPID-ABDM-TN-2026-0814',
            hospitalOrClinic: account.center || 'Primary Health Center — Thirunallur',
            assignedCamps: ['Thirunallur PHC', 'Kallakurichi Mobile Screening Unit', 'Thiruvarur Satellite Camp'],
            phone: '+91 94432 01080',
            role: 'CLINICAL_DOCTOR',
            status: 'ACTIVE_CONSULTANT',
            createdAt: '2026-01-15T08:00:00.000Z',
            updatedAt: new Date().toISOString()
        };
        localStorage.setItem('aidr_doctor_auth_user', JSON.stringify(doctorProfile));
    }

    // Broadcast session update to parent portal & all child frames
    try {
        if (window.parent && window.parent !== window) {
            window.parent.postMessage({
                action: 'sessionUpdated',
                session: sessionData
            }, '*');

            if (typeof window.parent.updatePortalUser === 'function') {
                window.parent.updatePortalUser();
            }
        }
    } catch (e) {}

    // Direct routing: Doctor -> doctor-dashboard.html, Health Center Staff -> screening terminal
    setTimeout(() => {
        if (isDoctor) {
            window.location.href = 'doctor-dashboard.html';
        } else {
            navigateTo('dashboard');
        }
    }, 550);
}

// -------------------------------------------------------------
// 14. CHECK EXISTING SESSION
// -------------------------------------------------------------
function checkExistingSession() {
    const oldBanner = document.getElementById('active-session-banner');
    if (oldBanner) oldBanner.remove();

    try {
        const raw = localStorage.getItem('aidr_user_session');
        if (!raw) return;
        const session = JSON.parse(raw);
        if (session && session.authenticated && session.user) {
            const isDoctor = session.user.role === 'doctor' || session.role === 'doctor' ||
                (session.user.id && session.user.id.toLowerCase().includes('dr.'));
            const continueTarget = isDoctor ? "window.location.href='doctor-dashboard.html'" : "navigateTo('dashboard')";
            const continueLabel = isDoctor ? "Open Doctor Profile & Patients →" : "Open Screening Terminal →";

            const note = document.createElement('div');
            note.id = 'active-session-banner';
            note.className = 'active-session-pill';
            note.innerHTML = `
                <span class="session-status-dot"></span>
                <span class="session-user-name">Active: <strong>${session.user.name}</strong> (${isDoctor ? 'Doctor' : 'PHC Staff'})</span>
                <button type="button" class="btn-session-continue" onclick="${continueTarget}">${continueLabel}</button>
                <button type="button" class="btn-session-switch" onclick="logoutSession()">Switch</button>
            `;
            document.body.appendChild(note);
        }
    } catch (e) {}
}

window.addEventListener('storage', (e) => {
    if (e.key === 'aidr_user_session') {
        checkExistingSession();
    }
});

window.addEventListener('message', (e) => {
    if (e.data && e.data.action === 'sessionUpdated') {
        checkExistingSession();
    }
});

// -------------------------------------------------------------
// 15. TOAST NOTIFICATION UTILITY
// -------------------------------------------------------------
function showToast(message, type = 'info') {
    let container = document.getElementById('toast-root');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `custom-toast toast-${type}`;

    const icons = {
        info: 'ℹ️',
        success: '✓',
        warning: '⚠️',
        error: '✕'
    };

    toast.innerHTML = `
        <span class="toast-badge-icon">${icons[type] || '●'}</span>
        <span class="toast-msg-text">${message}</span>
    `;

    container.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.add('toast-visible');
    });

    setTimeout(() => {
        toast.classList.remove('toast-visible');
        setTimeout(() => toast.remove(), 320);
    }, 3600);
}
