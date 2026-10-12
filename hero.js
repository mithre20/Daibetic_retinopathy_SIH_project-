/**
 * AI DR SCREEN — Hero Section Controller
 * Handles:
 * - Timed cinematic page load entrance sequence
 * - Kinetic typography word-by-word reveal
 * - Magnetic primary CTA micro-interaction
 * - Scroll-driven depth transitions
 * - Smooth scroll into live diagnostic station
 */

(function() {
    'use strict';

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    document.addEventListener('DOMContentLoaded', () => {
        initCinematicEntranceSequence();
        initCTAClicks();
        initScrollTransition();
    });

    /**
     * 1. Cinematic Page Load Entrance Sequence
     * 0ms   - Dark canvas
     * 200ms - Atmospheric aurora emerges
     * 400ms - Top navigation and wordmark appear
     * 800ms - Eyebrow label reveals
     * 1000ms - Kinetic typography word-by-word reveal:
     *          "AI" -> "DR" -> "SCREEN"
     *          "INTELLIGENCE" -> "BEHIND" -> "EVERY" -> "IMAGE."
     * 1200ms - Supporting copy reveals
     * 1400ms - Primary & Secondary CTAs appear
     * 1600ms - Bottom bar & scroll indicator reveal
     */
    function initCinematicEntranceSequence() {
        if (prefersReducedMotion) {
            // Immediate reveal for reduced motion users
            document.querySelectorAll(
                '.hero-aurora-glow, .hero-navbar, .hero-eyebrow, ' +
                '.kinetic-word, .hero-supporting-copy, .hero-actions-row, .hero-bottom-bar, ' +
                '.ai-scan-marker, .scan-laser-hud'
            ).forEach(el => el.classList.add('revealed', 'active'));
            return;
        }

        // 200ms: Aurora glow
        setTimeout(() => {
            document.querySelectorAll('.hero-aurora-glow').forEach(el => el.classList.add('active'));
        }, 200);

        // 400ms: Top navigation
        setTimeout(() => {
            const nav = document.getElementById('hero-navbar');
            if (nav) nav.classList.add('revealed');
        }, 400);

        // 800ms: Eyebrow
        setTimeout(() => {
            const eyebrow = document.getElementById('hero-eyebrow');
            if (eyebrow) eyebrow.classList.add('revealed');
        }, 800);

        // 1000ms: Kinetic Typography Word-by-Word Reveal
        setTimeout(() => {
            const words = Array.from(document.querySelectorAll('.kinetic-word'));
            words.forEach((word, index) => {
                setTimeout(() => {
                    word.classList.add('revealed');
                }, index * 95); // 95ms cadence between words
            });
        }, 950);

        // 1300ms: Supporting Copy
        setTimeout(() => {
            const copy = document.getElementById('hero-supporting-copy');
            if (copy) copy.classList.add('revealed');
        }, 1300);

        // 1450ms: Primary & Secondary CTAs
        setTimeout(() => {
            const actions = document.getElementById('hero-actions-row');
            if (actions) actions.classList.add('revealed');
        }, 1450);

        // 1650ms: Bottom Spec Strip & Scroll Indicator
        setTimeout(() => {
            const bottomBar = document.getElementById('hero-bottom-bar');
            if (bottomBar) bottomBar.classList.add('revealed');
        }, 1650);
    }

    /**
     * 2. Primary & Secondary CTA Click Navigation
     * Stable, non-magnetic interaction compliant with calm design principles
     */
    function initCTAClicks() {
        const btn = document.getElementById('hero-primary-cta');
        if (btn) {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                if (window.switchPortalTab) {
                    window.switchPortalTab('login', true);
                }
            });
        }

        // Secondary CTA: opens Page 02: Staff Login
        const secondaryBtn = document.getElementById('hero-secondary-cta');
        if (secondaryBtn) {
            secondaryBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (window.switchPortalTab) {
                    window.switchPortalTab('login', true);
                }
            });
        }

        // Demo request button in navbar -> opens Page 02: Staff Login
        const demoBtn = document.getElementById('hero-nav-demo-btn');
        if (demoBtn) {
            demoBtn.addEventListener('click', (e) => {
                e.preventDefault();
                if (window.switchPortalTab) {
                    window.switchPortalTab('login', true);
                }
            });
        }
    }

    /**
     * 3. Sequential Page Navigation from Hero
     * Access from hero page routes exclusively to Page 02: Staff Login
     */
    window.scrollToPortal = function(targetView = 'login') {
        if (window.switchPortalTab) {
            window.switchPortalTab('login', true);
        }
    };

    /**
     * 4. Scroll-Driven Depth Transitions
     * As user scrolls down from hero:
     * - Headline glides upward
     * - 3D scan scales down into depth
     * - Background atmosphere transitions into the portal
     */
    function initScrollTransition() {
        const heroSection = document.getElementById('hero-section');
        const headline = document.querySelector('.hero-headline');
        const sceneCol = document.querySelector('.hero-scene-col');

        if (!heroSection || prefersReducedMotion) return;

        window.addEventListener('scroll', () => {
            const scrollY = window.scrollY;
            const heroHeight = heroSection.offsetHeight;

            if (scrollY < heroHeight) {
                const progress = scrollY / heroHeight;

                if (headline) {
                    headline.style.transform = `translate3d(0, ${-progress * 40}px, 0)`;
                    headline.style.opacity = `${1 - progress * 1.2}`;
                }

                if (sceneCol) {
                    sceneCol.style.transform = `translate3d(0, ${-progress * 25}px, -${progress * 150}px) scale(${1 - progress * 0.15})`;
                    sceneCol.style.opacity = `${1 - progress * 0.8}`;
                }
            }
        }, { passive: true });
    }

})();
