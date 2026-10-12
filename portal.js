/**
 * XAIRIS — Explainable AI for Diabetic Retinopathy Screening
 * AI That Sees. AI That Explains.
 * Workflow Sequence:
 * - 01 Hero (First Page)
 *   -> Accessible only via: "EXPLORE XAIRIS", "SEE HOW IT WORKS", "REQUEST DEMO"
 * - 02 Staff Login
 * - 03 Main Screening Dashboard
 * - 04 MediVision Clinical Station
 * - 05 5-Stage Explainable AI Pipeline
 */

(function() {
    'use strict';

    // 6 Sequential Pages in exact user-requested order
    const VIEWS = ['hero', 'login', 'dashboard', 'medivision', 'pipeline', 'settings'];
    const PAGE_TITLES = {
        hero: '01 Hero Overview',
        login: '02 Staff Login',
        dashboard: '03 Main Screening Dashboard',
        medivision: '04 MediVision Clinical Station',
        pipeline: '05 5-Stage Explainable AI Pipeline',
        settings: '06 Settings & Preferences'
    };

    let currentView = 'hero';
    let isTransitioning = false;

    // DOM References
    let heroSection = null;
    let portalApp = null;
    let tabs = {};
    let panels = {};
    let navGlider = null;
    let navCenter = null;
    let masterBar = null;
    let causticOrbs = [];
    let userProfileChip = null;
    let userNameEl = null;
    let userRoleEl = null;
    let dropdownMenu = null;
    let stepperPrevBtn = null;
    let stepperNextBtn = null;
    let stepperCurText = null;

    // Parallax & 3D tilt tracking state
    let targetMouse = { x: 0, y: 0, normX: 0, normY: 0 };
    let currentMouse = { x: 0, y: 0, normX: 0, normY: 0 };
    let rafId = null;

    document.addEventListener('DOMContentLoaded', () => {
        initTheme();
        initElements();
        initSession();
        initHashRouting();
        initKeyboardShortcuts();
        initMessageBridge();

        // Position initial glider
        requestAnimationFrame(() => {
            if (currentView !== 'hero') {
                updateGliderPosition(currentView, false);
            }
            updateStepperUI(currentView);
        });

        window.addEventListener('resize', () => {
            if (currentView !== 'hero') {
                updateGliderPosition(currentView, false);
            }
        });
    });

    function initElements() {
        heroSection = document.getElementById('hero-section');
        portalApp = document.getElementById('portal-app');
        navGlider = document.getElementById('portal-nav-glider');
        navCenter = document.getElementById('portal-nav-center');
        masterBar = document.getElementById('portal-master-bar');
        causticOrbs = Array.from(document.querySelectorAll('.caustic-orb'));
        stepperPrevBtn = document.getElementById('portal-step-prev');
        stepperNextBtn = document.getElementById('portal-step-next');
        stepperCurText = document.getElementById('portal-step-cur-text');

        VIEWS.forEach(id => {
            tabs[id] = document.getElementById(`portal-tab-${id}`);
            panels[id] = document.getElementById(`portal-panel-${id}`);

            if (tabs[id]) {
                tabs[id].addEventListener('click', (e) => {
                    switchPortalTab(id, true, e);
                });
            }
        });

        userProfileChip = document.getElementById('portal-user-profile');
        userNameEl = document.getElementById('portal-user-name');
        userRoleEl = document.getElementById('portal-user-role');
        dropdownMenu = document.getElementById('portal-dropdown-menu');

        if (userProfileChip && dropdownMenu) {
            userProfileChip.addEventListener('click', (e) => {
                e.stopPropagation();
                dropdownMenu.classList.toggle('show');
            });

            document.addEventListener('click', () => {
                dropdownMenu.classList.remove('show');
            });
        }

        // Sign out button
        const signOutBtn = document.getElementById('portal-btn-signout');
        if (signOutBtn) {
            signOutBtn.addEventListener('click', () => {
                localStorage.removeItem('aidr_user_session');
                showPortalToast('Signed out of terminal', 'info');
                initSession();
                broadcastSessionToPanels(null);
                switchPortalTab('login', true);
            });
        }

        // Switch account button
        const switchAccBtn = document.getElementById('portal-btn-switch-account');
        if (switchAccBtn) {
            switchAccBtn.addEventListener('click', () => {
                switchPortalTab('login', true);
            });
        }
    }

    /**
     * Updates Stepper UI with Current Page Index & Tooltips
     */
    function updateStepperUI(viewId) {
        const idx = VIEWS.indexOf(viewId);
        if (idx === -1) return;

        if (stepperCurText) {
            stepperCurText.textContent = '0' + (idx + 1);
        }

        if (stepperPrevBtn) {
            const prevIdx = (idx - 1 + VIEWS.length) % VIEWS.length;
            stepperPrevBtn.title = `Previous Page: ${PAGE_TITLES[VIEWS[prevIdx]]} (ArrowLeft / Alt+Left)`;
        }

        if (stepperNextBtn) {
            const nextIdx = (idx + 1) % VIEWS.length;
            stepperNextBtn.title = `Next Page: ${PAGE_TITLES[VIEWS[nextIdx]]} (ArrowRight / Alt+Right)`;
        }
    }

    /**
     * Advances or rewinds by delta in exact corresponding order
     * Order: 01 Hero -> 02 Login -> 03 Dashboard -> 04 MediVision -> 05 Pipeline
     */
    window.stepPage = function(delta) {
        // Access from the hero page should go only to the login page
        if (currentView === 'hero') {
            switchPortalTab('login', true);
            return;
        }
        const currentIdx = VIEWS.indexOf(currentView);
        let nextIdx = currentIdx + delta;
        if (nextIdx < 0) nextIdx = VIEWS.length - 1; // wrap around
        if (nextIdx >= VIEWS.length) nextIdx = 0;   // wrap around
        switchPortalTab(VIEWS[nextIdx], true);
    };

    /**
     * Physically Connected Morphing Spring Glider
     */
    function updateGliderPosition(viewId, animate = true) {
        const targetTab = tabs[viewId];
        if (!targetTab || !navGlider || !navCenter) return;

        const navRect = navCenter.getBoundingClientRect();
        const tabRect = targetTab.getBoundingClientRect();

        const targetLeft = tabRect.left - navRect.left;
        const targetWidth = tabRect.width;

        if (!animate) {
            navGlider.style.transition = 'none';
            navGlider.style.transform = `translate3d(${targetLeft}px, 0, 0)`;
            navGlider.style.width = `${targetWidth}px`;
            navGlider.offsetHeight; // force reflow
            navGlider.style.transition = '';
            return;
        }

        const currentTransform = window.getComputedStyle(navGlider).transform;
        let currentX = targetLeft;
        if (currentTransform && currentTransform !== 'none') {
            const matrix = new DOMMatrix(currentTransform);
            currentX = matrix.m41;
        }

        const distance = Math.abs(targetLeft - currentX);
        const stretchRatio = Math.min(1.15, 1 + (distance / 600));

        navGlider.style.transform = `translate3d(${targetLeft}px, 0, 0) scaleX(${stretchRatio})`;
        navGlider.style.width = `${targetWidth}px`;

        setTimeout(() => {
            navGlider.style.transform = `translate3d(${targetLeft}px, 0, 0) scaleX(1)`;
        }, 180);
    }

    /**
     * Cinematic Sequential Page Switcher
     * Order: 01 Hero -> 02 Login -> 03 Dashboard -> 04 MediVision -> 05 Pipeline
     */
    window.switchPortalTab = function(newView, updateHash = true, clickEvent = null) {
        // Enforce rule: Access from the hero page should go only to the login page or settings
        if (currentView === 'hero' && newView !== 'hero' && newView !== 'login' && newView !== 'settings') {
            newView = 'login';
        }

        if (!VIEWS.includes(newView) || isTransitioning) {
            return;
        }

        if (newView === currentView) {
            if (newView !== 'hero') {
                updateGliderPosition(newView, false);
            }
            return;
        }

        const oldView = currentView;
        const oldIndex = VIEWS.indexOf(oldView);
        const newIndex = VIEWS.indexOf(newView);
        const isForward = newIndex > oldIndex;

        currentView = newView;
        isTransitioning = true;

        // 1. Update Tabs active classes
        VIEWS.forEach(id => {
            if (tabs[id]) {
                tabs[id].classList.toggle('active', id === newView);
            }
        });

        // 2. Move Morphing Glider & update Stepper
        if (newView !== 'hero') {
            updateGliderPosition(newView, true);
        }
        updateStepperUI(newView);

        // 3. Coordinate View Transitions:
        // CASE A: Returning to Page 01 (Hero Overview)
        if (newView === 'hero') {
            if (portalApp) {
                portalApp.classList.remove('portal-active', 'portal-exit-left', 'portal-exit-right');
                portalApp.classList.add('portal-exit-right');
            }
            if (heroSection) {
                heroSection.classList.remove('page-exit-left', 'page-exit-right');
                heroSection.style.transform = 'translate3d(0, 0, 0)';
                heroSection.style.opacity = '1';
                heroSection.style.visibility = 'visible';
                heroSection.style.pointerEvents = 'auto';
            }

            setTimeout(() => {
                if (portalApp) {
                    portalApp.style.visibility = 'hidden';
                }
                isTransitioning = false;
            }, 500);
        }
        // CASE B: Transitioning from Page 01 (Hero) into Portal (02–05)
        else if (oldView === 'hero') {
            if (heroSection) {
                heroSection.classList.remove('page-exit-left', 'page-exit-right');
                heroSection.classList.add(isForward ? 'page-exit-left' : 'page-exit-right');
            }
            if (portalApp) {
                portalApp.style.visibility = 'visible';
                portalApp.classList.remove('portal-exit-right', 'portal-exit-left');
                portalApp.classList.add('portal-active');
            }

            // Directly activate target panel inside portal
            VIEWS.forEach(id => {
                if (panels[id]) {
                    if (id === newView) {
                        panels[id].style.display = 'block';
                        panels[id].classList.remove('view-exit-left', 'view-exit-right', 'view-enter-from-right', 'view-enter-from-left');
                        panels[id].classList.add('active');
                        try {
                            if (panels[id].contentWindow && panels[id].contentWindow.focus) {
                                panels[id].contentWindow.focus();
                            }
                        } catch (e) {}
                    } else {
                        panels[id].classList.remove('active');
                        panels[id].style.display = 'none';
                    }
                }
            });

            setTimeout(() => {
                if (heroSection) {
                    heroSection.style.visibility = 'hidden';
                }
                isTransitioning = false;
            }, 500);
        }
        // CASE C: Transitioning BETWEEN Portal Pages (02, 03, 04, 05)
        else {
            if (portalApp) {
                portalApp.style.visibility = 'visible';
                portalApp.classList.add('portal-active');
            }

            let originPercent = 50;
            if (tabs[newView]) {
                const tabRect = tabs[newView].getBoundingClientRect();
                originPercent = ((tabRect.left + tabRect.width / 2) / window.innerWidth) * 100;
            }
            document.documentElement.style.setProperty('--clip-origin-x', `${originPercent}%`);

            const oldPanel = panels[oldView];
            const newPanel = panels[newView];

            if (oldPanel) {
                oldPanel.classList.remove('active', 'view-exit-left', 'view-exit-right');
                oldPanel.classList.add(isForward ? 'view-exit-left' : 'view-exit-right');
            }

            if (newPanel) {
                newPanel.style.display = 'block';
                newPanel.classList.remove('active', 'view-exit-left', 'view-exit-right');
                newPanel.classList.add(isForward ? 'view-enter-from-right' : 'view-enter-from-left');

                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        newPanel.classList.remove('view-enter-from-right', 'view-enter-from-left');
                        newPanel.classList.add('active');

                        try {
                            if (newPanel.contentWindow && newPanel.contentWindow.focus) {
                                newPanel.contentWindow.focus();
                            }
                        } catch (e) {}
                    });
                });
            }

            setTimeout(() => {
                if (oldPanel) {
                    oldPanel.classList.remove('view-exit-left', 'view-exit-right');
                    oldPanel.style.display = 'none';
                }
                isTransitioning = false;
            }, 500);
        }

        if (updateHash) {
            window.location.hash = newView;
        }

        showPortalToast(`Transferred to ${PAGE_TITLES[newView]}`, 'info');
    };

    /**
     * Calm Technology - Master bar is completely stationary without cursor tracking or 3D tilt
     */
    function initCinematicParallaxAnd3D() {
        // Disabled per 2026 Calm UI/UX healthcare standard
    }

    /**
     * Session Synchronization across all modules
     */
    window.updatePortalUser = function() {
        initSession();
        let session = null;
        try {
            const raw = localStorage.getItem('aidr_user_session');
            if (raw) session = JSON.parse(raw);
        } catch (e) {}
        broadcastSessionToPanels(session);
    };

    function broadcastSessionToPanels(sessionData) {
        Object.values(panels).forEach(panel => {
            if (panel && panel.contentWindow) {
                try {
                    panel.contentWindow.postMessage({
                        action: 'sessionUpdated',
                        session: sessionData
                    }, '*');
                } catch (e) {}
            }
        });
    }

    function initSession() {
        try {
            const raw = localStorage.getItem('aidr_user_session');
            if (raw) {
                const session = JSON.parse(raw);
                if (session && session.user) {
                    if (userNameEl) userNameEl.textContent = session.user.name || 'Medical Officer';
                    if (userRoleEl) userRoleEl.textContent = session.user.roleLabel || session.user.center || 'Primary Health Center';
                    return;
                }
            }
        } catch (e) {}

        if (userNameEl) userNameEl.textContent = 'Dr. Anitha Rajan';
        if (userRoleEl) userRoleEl.textContent = 'Primary Health Center';
    }

    /**
     * Hash Routing with Hero Default
     */
    function initHashRouting() {
        function checkHash() {
            const hash = window.location.hash.replace('#', '').trim().toLowerCase();
            if (VIEWS.includes(hash)) {
                switchPortalTab(hash, false);
            } else {
                // Default to Page 01: Hero Overview as the initial first page!
                switchPortalTab('hero', false);
            }
        }

        window.addEventListener('hashchange', checkHash);
        checkHash();
    }

    /**
     * Keyboard Shortcuts (ArrowLeft/Right, Alt+1..5, PageUp/PageDown)
     */
    function initKeyboardShortcuts() {
        window.addEventListener('keydown', (e) => {
            if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;

            // Sequential step with arrow keys or PageUp/PageDown
            if (e.key === 'ArrowRight' || e.key === 'PageDown') {
                e.preventDefault();
                stepPage(1);
            } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
                e.preventDefault();
                stepPage(-1);
            } else if (e.altKey && !e.ctrlKey && !e.metaKey) {
                if (e.key === '1') { e.preventDefault(); switchPortalTab('hero'); }
                if (e.key === '2') { e.preventDefault(); switchPortalTab('login'); }
                if (e.key === '3') { e.preventDefault(); switchPortalTab('dashboard'); }
                if (e.key === '4') { e.preventDefault(); switchPortalTab('medivision'); }
                if (e.key === '5') { e.preventDefault(); switchPortalTab('pipeline'); }
                if (e.key === '6') { e.preventDefault(); switchPortalTab('settings'); }
            }
        });
    }

    /**
     * Inter-frame Message Bridge
     */
    function initMessageBridge() {
        window.addEventListener('message', (event) => {
            if (!event.data) return;
            if (event.data.action === 'switchView' && event.data.target) {
                switchPortalTab(event.data.target, true);
            }
            if (event.data.action === 'stepPage' && typeof event.data.delta === 'number') {
                stepPage(event.data.delta);
            }
            if (event.data.action === 'sessionUpdated') {
                initSession();
                broadcastSessionToPanels(event.data.session);
            }
            if (event.data.action === 'toast' && event.data.message) {
                showPortalToast(event.data.message, event.data.type || 'info');
            }
            if (event.data.action === 'getTheme' && event.source) {
                try {
                    event.source.postMessage({ action: 'themeChange', theme: currentTheme }, '*');
                } catch (e) {}
            }
            if (event.data.action === 'setTheme' && event.data.theme) {
                setTheme(event.data.theme, true);
            }
        });

        window.addEventListener('storage', (e) => {
            if (e.key === 'aidr_user_session') {
                initSession();
            }
            if (e.key === 'xairis_theme' && e.newValue) {
                setTheme(e.newValue, false);
            }
        });
    }

    /**
     * Global Theme Management (Light by default)
     */
    let currentTheme = localStorage.getItem('xairis_theme') || 'light';

    function initTheme() {
        setTheme(currentTheme, false);

        const toggleBtn = document.getElementById('portal-theme-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
                setTheme(nextTheme, true);
            });
        }
    }

    function setTheme(theme, broadcast = true) {
        currentTheme = theme;
        localStorage.setItem('xairis_theme', theme);
        document.documentElement.setAttribute('data-theme', theme);
        document.body.setAttribute('data-theme', theme);

        const toggleBtn = document.getElementById('portal-theme-toggle');
        if (toggleBtn) {
            const label = toggleBtn.querySelector('.theme-label');
            if (label) {
                label.textContent = theme === 'light' ? 'Light' : 'Dark';
            }
        }

        // Notify 3D engine if loaded
        if (window.update3DTheme) {
            window.update3DTheme(theme);
        }

        // Broadcast to all child iframe panels
        Object.values(panels).forEach(panel => {
            if (panel && panel.contentWindow) {
                try {
                    panel.contentWindow.postMessage({ action: 'themeChange', theme: theme }, '*');
                } catch (e) {}
            }
        });

        if (broadcast) {
            showPortalToast(`Switched to ${theme === 'light' ? 'Light Theme' : 'Dark Theme'}`, 'info');
        }
    }
    window.setPortalTheme = setTheme;
    window.getPortalTheme = () => currentTheme;

    /**
     * Spring-Powered Cinematic Toast Notification
     */
    let toastTimer = null;
    window.showPortalToast = function(msg, type = 'info') {
        let toast = document.getElementById('portal-toast');
        if (!toast) return;

        toast.textContent = (type === 'success' ? '✓ ' : type === 'error' ? '⚠ ' : '✦ ') + msg;
        toast.classList.add('show');

        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            toast.classList.remove('show');
        }, 2800);
    };

})();
