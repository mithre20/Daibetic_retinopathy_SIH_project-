/**
 * XAIRIS — 3D AI Diabetic Retinopathy (DR) Screening Model Suite (Three.js)
 * Interactive Models:
 * 1. 3D Anatomical Human Eye Globe (with Interactive Cutaway / Cross-Section)
 * 2. 3D Retinal Fundus & AI Lesion Detection Map (Microaneurysms, Hemorrhages, Hard Exudates, Cotton Wool Spots, Neovascularization)
 * 3. 3D Volumetric OCT (Optical Coherence Tomography) Layer Stratification with Diabetic Macular Edema (DME)
 * 4. 3D Digital AI Fundus Camera & Optical Diagnostic Sensor Rig
 *
 * Features:
 * - Orbit drag rotation & wheel zoom
 * - Real-time AI laser scan sweep
 * - Interactive 3D lesion beacons with raycaster tooltips
 * - Dynamic theme adaptation (Light / Dark mode lighting)
 * - Telemetry & HUD synchronizer
 */

(function() {
    'use strict';

    let scene, camera, renderer;
    let masterGroup;
    let modelEye, modelRetina, modelOCT, modelCamera;
    let activeModelId = 'eye';
    let animationFrameId = null;

    let dirLight, hemiLight, ambientLight, pointLight;
    let laserSweepMesh = null;
    let laserOCTMesh = null;
    let lesionBeacons = [];
    let showLesions = true;
    let showCutaway = false;
    let autoRotate = true;

    // Interaction & Camera tracking
    const mouse = { x: 0, y: 0 };
    const orbit = {
        isDragging: false,
        prevMouseX: 0,
        prevMouseY: 0,
        rotX: 0.15,
        rotY: -0.2,
        targetRotX: 0.15,
        targetRotY: -0.2,
        zoom: 8.5,
        targetZoom: 8.5
    };

    const container = document.getElementById('hero-3d-canvas-wrap');
    const canvas = document.getElementById('hero-3d-canvas');
    const tooltipEl = document.getElementById('hero-3d-tooltip');

    // Raycasting for interactive 3D lesion beacons
    const raycaster = new THREE.Raycaster();
    const mouseNormalized = new THREE.Vector2(-999, -999);
    let hoveredBeacon = null;

    // Model telemetry metadata
    const MODEL_TELEMETRY = {
        eye: {
            label1: 'ANATOMICAL GLOBE',
            val1: 'HUMAN EYE',
            sub1: 'CORNEA • IRIS • RETINA • OPTIC NERVE',
            label2: 'OPTICAL AXIS',
            val2: '24.2 mm AXIAL',
            sub2: 'ANTERIOR & POSTERIOR CHAMBERS',
            laser: 'OPTICAL ALIGNMENT [ACTIVE]'
        },
        retina: {
            label1: 'AI DR DETECTION',
            val1: '98.4% CONFIDENCE',
            sub1: 'MODERATE NPDR • 7 LESION HOTSPOTS',
            label2: 'POSTERIOR POLE',
            val2: 'MACULA + OPTIC DISC',
            sub2: '4-QUADRANT ARTERIOVENOUS ARCADE',
            laser: 'AI LESION SCANNER [SWEEPING]'
        },
        oct: {
            label1: 'OCT MACULAR EDEMA',
            val1: '420 µm THICKNESS',
            sub1: 'CYSTOID INTRARETINAL FLUID DETECTED',
            label2: 'LAYER STRATA',
            val2: '10 RETINAL LAYERS',
            sub2: 'ILM • GCL • IPL • INL • OPL • ONL • RPE',
            laser: 'B-SCAN CROSS-SECTION [420µm]'
        },
        camera: {
            label1: 'DIGITAL SENSOR RIG',
            val1: '50° FIELD OF VIEW',
            sub1: 'MULTI-SPECTRUM LED FLASH • TELECENTRIC',
            label2: 'TARGET ACQUISITION',
            val2: 'AUTO-FOCUS LOCK',
            sub2: 'OPHTHALMIC OBJECTIVE BARREL v2.6',
            laser: 'FUNDUS TARGET RETICLE [LOCKED]'
        }
    };

    document.addEventListener('DOMContentLoaded', () => {
        if (typeof THREE === 'undefined') {
            const checkThree = setInterval(() => {
                if (typeof THREE !== 'undefined') {
                    clearInterval(checkThree);
                    initThreeScene();
                }
            }, 50);
            setTimeout(() => clearInterval(checkThree), 6000);
        } else {
            initThreeScene();
        }
    });

    function initThreeScene() {
        if (!container || !canvas) return;

        try {
            // 1. Scene Setup
            scene = new THREE.Scene();

            // 2. Camera Setup
            const width = container.clientWidth || 560;
            const height = container.clientHeight || 560;
            camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1000);
            camera.position.set(0, 0, orbit.zoom);

            // 3. Renderer with high precision
            renderer = new THREE.WebGLRenderer({
                canvas: canvas,
                alpha: true,
                antialias: true,
                powerPreference: 'high-performance'
            });
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

            // 4. Lighting System (Adaptive to Light & Dark themes)
            setupLighting();

            // 5. Master 3D Group
            masterGroup = new THREE.Group();
            scene.add(masterGroup);

            // 6. Build the 4 Clinical 3D Models
            buildModelEye();
            buildModelRetina();
            buildModelOCT();
            buildModelCamera();

            // Activate initial model (3D Eye Globe)
            switchModel('eye', false);

            // 7. Setup Interaction & Event Listeners
            setupEventListeners();
            setupToolbarListeners();

            // 8. Start Rendering Loop
            animate(0);

            // Reveal 3D UI markers
            setTimeout(() => {
                const markers = document.querySelectorAll('.ai-scan-marker, .scan-laser-hud');
                markers.forEach(m => m.classList.add('revealed'));
            }, 600);

            // Expose global update hook
            window.update3DTheme = updateLightingForTheme;

        } catch (err) {
            console.warn('XAIRIS 3D Suite: WebGL initialization failed', err);
        }
    }

    /**
     * Lighting Setup calibrated for Light Theme default & Dark Theme toggle
     */
    function setupLighting() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';

        ambientLight = new THREE.AmbientLight(isDark ? 0x223344 : 0xffffff, isDark ? 1.2 : 1.6);
        scene.add(ambientLight);

        hemiLight = new THREE.HemisphereLight(
            isDark ? 0x0ea5e9 : 0xf0fdfa,
            isDark ? 0x050811 : 0xdbeafe,
            isDark ? 0.8 : 1.0
        );
        hemiLight.position.set(0, 20, 0);
        scene.add(hemiLight);

        dirLight = new THREE.DirectionalLight(0xffffff, isDark ? 1.4 : 1.6);
        dirLight.position.set(8, 12, 10);
        scene.add(dirLight);

        pointLight = new THREE.PointLight(0x0d9488, isDark ? 1.6 : 1.2, 25);
        pointLight.position.set(-6, -4, 6);
        scene.add(pointLight);
    }

    function updateLightingForTheme(theme) {
        if (!ambientLight || !hemiLight || !dirLight) return;
        const isDark = theme === 'dark';

        ambientLight.color.setHex(isDark ? 0x223344 : 0xffffff);
        ambientLight.intensity = isDark ? 1.2 : 1.6;

        hemiLight.color.setHex(isDark ? 0x0ea5e9 : 0xf0fdfa);
        hemiLight.groundColor.setHex(isDark ? 0x050811 : 0xdbeafe);
        hemiLight.intensity = isDark ? 0.8 : 1.0;

        dirLight.intensity = isDark ? 1.4 : 1.6;
        pointLight.color.setHex(isDark ? 0x00e5a3 : 0x0d9488);
    }

    /* ============================================================
       PROCEDURAL MEDICAL TEXTURES (CANVAS GENERATORS)
       ============================================================ */

    /**
     * Procedural Iris Texture (Golden Amber, Hazel & Sky Blue fibrillae matching reference image)
     */
    function createIrisTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');

        const cx = 256, cy = 256;

        // Base gradient from warm amber core to golden hazel and clinical cerulean limbus
        const grad = ctx.createRadialGradient(cx, cy, 25, cx, cy, 255);
        grad.addColorStop(0, '#1c1917');
        grad.addColorStop(0.18, '#451a03'); // deep warm amber core
        grad.addColorStop(0.42, '#b45309'); // golden hazel
        grad.addColorStop(0.68, '#0284c7'); // clinical cerulean transition
        grad.addColorStop(0.88, '#0369a1');
        grad.addColorStop(1, '#0b2540');    // deep dark limbal ring

        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, 512, 512);

        // Radial iris muscle striations (golden amber + precision sky blue fibers)
        ctx.save();
        ctx.translate(cx, cy);
        for (let i = 0; i < 360; i++) {
            const angle = (i * Math.PI) / 180;
            const len = 70 + Math.sin(i * 14) * 25 + Math.random() * 135;
            ctx.rotate(angle);
            ctx.beginPath();
            ctx.moveTo(35, 0);
            ctx.lineTo(len, 0);
            if (i % 3 === 0) {
                ctx.strokeStyle = 'rgba(251, 191, 36, 0.48)'; // golden amber
                ctx.lineWidth = 1.3;
            } else if (i % 3 === 1) {
                ctx.strokeStyle = 'rgba(56, 189, 248, 0.42)'; // clinical cyan
                ctx.lineWidth = 1.1;
            } else {
                ctx.strokeStyle = 'rgba(217, 119, 6, 0.38)';  // warm hazel
                ctx.lineWidth = 1.2;
            }
            ctx.stroke();
            ctx.rotate(-angle);
        }
        ctx.restore();

        // Inner pupillary collarette ring (golden glow)
        ctx.beginPath();
        ctx.arc(cx, cy, 78, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(252, 211, 77, 0.55)';
        ctx.lineWidth = 3.5;
        ctx.stroke();

        // Deep central pupil
        ctx.beginPath();
        ctx.arc(cx, cy, 48, 0, Math.PI * 2);
        ctx.fillStyle = '#020617';
        ctx.fill();

        return new THREE.CanvasTexture(canvas);
    }

    /**
     * Procedural Retinal Fundus Texture with Optic Disc & Vascular Tree
     */
    function createRetinaFundusTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 1024;
        const ctx = canvas.getContext('2d');

        const cx = 512, cy = 512;

        // Warm orange-red retinal fundus background gradient
        const bgGrad = ctx.createRadialGradient(cx, cy, 50, cx, cy, 500);
        bgGrad.addColorStop(0, '#ea580c');
        bgGrad.addColorStop(0.4, '#c2410c');
        bgGrad.addColorStop(0.75, '#9a3412');
        bgGrad.addColorStop(1, '#7c2d12');
        ctx.fillStyle = bgGrad;
        ctx.fillRect(0, 0, 1024, 1024);

        // Subtle choroidal vascular mottled pattern
        ctx.fillStyle = 'rgba(67, 20, 7, 0.15)';
        for (let i = 0; i < 600; i++) {
            const rx = Math.random() * 1024;
            const ry = Math.random() * 1024;
            ctx.beginPath();
            ctx.arc(rx, ry, 2 + Math.random() * 6, 0, Math.PI * 2);
            ctx.fill();
        }

        // 1. Optic Disc (Nasal side: x ~ 720, y ~ 512)
        const odX = 720, odY = 512;
        const odGrad = ctx.createRadialGradient(odX, odY, 15, odX, odY, 70);
        odGrad.addColorStop(0, '#fef08a'); // central physiological cup
        odGrad.addColorStop(0.6, '#fde047');
        odGrad.addColorStop(0.85, '#f59e0b');
        odGrad.addColorStop(1, '#d97706'); // neuroretinal rim
        ctx.fillStyle = odGrad;
        ctx.beginPath();
        ctx.arc(odX, odY, 68, 0, Math.PI * 2);
        ctx.fill();

        // 2. Macula & Fovea Centralis (Temporal: x ~ 430, y ~ 520)
        const macX = 430, macY = 520;
        const macGrad = ctx.createRadialGradient(macX, macY, 10, macX, macY, 120);
        macGrad.addColorStop(0, '#451a03'); // deep foveal avascular zone
        macGrad.addColorStop(0.4, '#78350f');
        macGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = macGrad;
        ctx.beginPath();
        ctx.arc(macX, macY, 120, 0, Math.PI * 2);
        ctx.fill();

        // Foveal reflex pinpoint
        ctx.fillStyle = 'rgba(254, 240, 138, 0.7)';
        ctx.beginPath();
        ctx.arc(macX, macY, 3, 0, Math.PI * 2);
        ctx.fill();

        // 3. Retinal Vascular Tree (Arterioles in bright crimson, Venules in dark wine)
        function drawVesselBranch(startX, startY, endX, endY, cp1X, cp1Y, cp2X, cp2Y, width, isArtery) {
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.bezierCurveTo(cp1X, cp1Y, cp2X, cp2Y, endX, endY);
            ctx.strokeStyle = isArtery ? 'rgba(239, 68, 68, 0.92)' : 'rgba(153, 27, 27, 0.9)';
            ctx.lineWidth = width;
            ctx.lineCap = 'round';
            ctx.stroke();
        }

        // Superior Temporal Arcade
        drawVesselBranch(odX - 10, odY - 30, 240, 140, 560, 240, 360, 160, 7.5, true);
        drawVesselBranch(odX, odY - 35, 200, 120, 580, 210, 340, 130, 9.5, false);

        // Inferior Temporal Arcade
        drawVesselBranch(odX - 10, odY + 30, 250, 880, 560, 780, 360, 860, 7.5, true);
        drawVesselBranch(odX, odY + 35, 210, 900, 580, 810, 340, 890, 9.5, false);

        // Nasal Arterioles & Venules
        drawVesselBranch(odX + 25, odY - 20, 950, 260, 820, 380, 900, 310, 6, true);
        drawVesselBranch(odX + 30, odY + 20, 960, 780, 830, 640, 910, 720, 7, false);

        // Macular fine twigs
        drawVesselBranch(520, 320, 440, 440, 480, 370, 450, 400, 2.5, true);
        drawVesselBranch(520, 720, 440, 600, 480, 670, 450, 640, 2.5, true);

        return new THREE.CanvasTexture(canvas);
    }

    /**
     * Procedural OCT Layer Stratification Texture
     */
    function createOCTTexture() {
        const canvas = document.createElement('canvas');
        canvas.width = 1024;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');

        // Dark background
        ctx.fillStyle = '#030712';
        ctx.fillRect(0, 0, 1024, 512);

        // Layer colors replicating clinical OCT pseudo-coloring
        const layers = [
            { y: 60, h: 25, color: '#fef08a', name: 'ILM / NFL' },
            { y: 90, h: 35, color: '#6ee7b7', name: 'GCL' },
            { y: 130, h: 40, color: '#38bdf8', name: 'IPL' },
            { y: 175, h: 35, color: '#1e293b', name: 'INL' },
            { y: 215, h: 45, color: '#93c5fd', name: 'OPL' },
            { y: 265, h: 45, color: '#0f172a', name: 'ONL' },
            { y: 315, h: 25, color: '#fcd34d', name: 'ELM / IS-OS' },
            { y: 345, h: 35, color: '#f97316', name: 'RPE' },
            { y: 385, h: 80, color: '#7c2d12', name: 'Choroid' }
        ];

        layers.forEach(layer => {
            ctx.fillStyle = layer.color;
            ctx.beginPath();
            ctx.moveTo(0, layer.y);
            for (let x = 0; x <= 1024; x += 16) {
                // Dip at fovea in center (x ~ 512)
                const distFovea = Math.abs(x - 512);
                const foveaDip = Math.exp(-(distFovea * distFovea) / 12000) * 45;
                const wave = Math.sin(x * 0.02) * 3;
                ctx.lineTo(x, layer.y + foveaDip + wave);
            }
            ctx.lineTo(1024, layer.y + layer.h);
            ctx.lineTo(0, layer.y + layer.h);
            ctx.closePath();
            ctx.globalAlpha = 0.85;
            ctx.fill();
        });

        // Diabetic Macular Edema (DME) Hyporeflective Fluid Cysts (Center)
        ctx.fillStyle = '#020617';
        ctx.globalAlpha = 0.95;
        const cystPositions = [
            { x: 470, y: 180, rx: 32, ry: 20 },
            { x: 540, y: 195, rx: 36, ry: 24 },
            { x: 505, y: 235, rx: 28, ry: 18 },
            { x: 440, y: 220, rx: 24, ry: 15 }
        ];
        cystPositions.forEach(c => {
            ctx.beginPath();
            ctx.ellipse(c.x, c.y, c.rx, c.ry, 0, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#38bdf8';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });

        ctx.globalAlpha = 1;
        return new THREE.CanvasTexture(canvas);
    }

    /* ============================================================
       MODEL 1: 3D ANATOMICAL HUMAN EYE GLOBE & RETINAL CAVITY
       ============================================================ */
    function buildModelEye() {
        modelEye = new THREE.Group();

        // 1.1 Outer Sclera Shell
        const scleraMat = new THREE.MeshStandardMaterial({
            color: 0xf8fafc,
            roughness: 0.25,
            metalness: 0.05,
            side: THREE.DoubleSide
        });

        // Full Eye Sclera (Sphere)
        const scleraGeoFull = new THREE.SphereGeometry(2.3, 36, 36);
        const scleraMeshFull = new THREE.Mesh(scleraGeoFull, scleraMat);
        modelEye.add(scleraMeshFull);

        // Cutaway Sclera (Quarter hemisphere window revealing interior)
        const scleraGeoCut = new THREE.SphereGeometry(2.3, 36, 36, 0, Math.PI * 1.5, 0, Math.PI);
        const scleraMeshCut = new THREE.Mesh(scleraGeoCut, scleraMat);
        scleraMeshCut.visible = false;
        modelEye.add(scleraMeshCut);
        modelEye.userData = { fullMesh: scleraMeshFull, cutMesh: scleraMeshCut };

        // 1.2 Clear Protruding Cornea (Front Optical Dome)
        const corneaGeo = new THREE.SphereGeometry(1.25, 32, 24, 0, Math.PI * 2, 0, Math.PI * 0.45);
        const corneaMat = new THREE.MeshPhysicalMaterial({
            color: 0xffffff,
            transmission: 0.95,
            opacity: 0.95,
            transparent: true,
            roughness: 0.05,
            ior: 1.376, // anatomical corneal refractive index
            thickness: 0.5,
            side: THREE.DoubleSide
        });
        const corneaMesh = new THREE.Mesh(corneaGeo, corneaMat);
        corneaMesh.position.set(0, 0, 1.4);
        corneaMesh.rotation.x = Math.PI;
        modelEye.add(corneaMesh);

        // 1.3 Detailed Iris with Procedural Striations
        const irisTex = createIrisTexture();
        const irisGeo = new THREE.RingGeometry(0.35, 1.2, 36);
        const irisMat = new THREE.MeshStandardMaterial({
            map: irisTex,
            roughness: 0.4,
            side: THREE.DoubleSide
        });
        const irisMesh = new THREE.Mesh(irisGeo, irisMat);
        irisMesh.position.set(0, 0, 1.48);
        modelEye.add(irisMesh);

        // 1.4 Deep Pupil Aperture
        const pupilGeo = new THREE.CircleGeometry(0.36, 32);
        const pupilMat = new THREE.MeshBasicMaterial({ color: 0x020617 });
        const pupilMesh = new THREE.Mesh(pupilGeo, pupilMat);
        pupilMesh.position.set(0, 0, 1.47);
        modelEye.add(pupilMesh);

        // 1.5 Crystalline Lens behind Pupil
        const lensGeo = new THREE.SphereGeometry(0.7, 24, 16);
        lensGeo.scale(1, 1, 0.45);
        const lensMat = new THREE.MeshPhysicalMaterial({
            color: 0xecfeff,
            transmission: 0.9,
            transparent: true,
            roughness: 0.1,
            ior: 1.42
        });
        const lensMesh = new THREE.Mesh(lensGeo, lensMat);
        lensMesh.position.set(0, 0, 1.15);
        modelEye.add(lensMesh);

        // 1.6 Optic Nerve Trunk (Posterior exit)
        const nerveGeo = new THREE.CylinderGeometry(0.38, 0.42, 1.6, 24);
        const nerveMat = new THREE.MeshStandardMaterial({ color: 0xfde68a, roughness: 0.6 });
        const nerveMesh = new THREE.Mesh(nerveGeo, nerveMat);
        nerveMesh.position.set(0.6, -0.2, -2.5);
        nerveMesh.rotation.x = Math.PI / 2.2;
        nerveMesh.rotation.z = -0.25;
        modelEye.add(nerveMesh);

        // Central Retinal Artery (Red) & Vein (Blue) entering nerve
        const artGeo = new THREE.CylinderGeometry(0.06, 0.06, 1.8, 12);
        const artMat = new THREE.MeshBasicMaterial({ color: 0xef4444 });
        const artMesh = new THREE.Mesh(artGeo, artMat);
        artMesh.position.set(0.55, -0.22, -2.5);
        artMesh.rotation.copy(nerveMesh.rotation);
        modelEye.add(artMesh);

        const veinGeo = new THREE.CylinderGeometry(0.07, 0.07, 1.8, 12);
        const veinMat = new THREE.MeshBasicMaterial({ color: 0x2563eb });
        const veinMesh = new THREE.Mesh(veinGeo, veinMat);
        veinMesh.position.set(0.66, -0.18, -2.5);
        veinMesh.rotation.copy(nerveMesh.rotation);
        modelEye.add(veinMesh);

        // 1.7 Internal Concave Retinal Fundus (visible inside cutaway)
        const innerRetinaGeo = new THREE.SphereGeometry(2.26, 32, 32);
        const innerRetinaMat = new THREE.MeshStandardMaterial({
            map: createRetinaFundusTexture(),
            roughness: 0.5,
            side: THREE.BackSide
        });
        const innerRetinaMesh = new THREE.Mesh(innerRetinaGeo, innerRetinaMat);
        modelEye.add(innerRetinaMesh);

        // 1.8 Anatomical Extraocular Rectus Muscle Bands (matching reference image)
        const muscleMat = new THREE.MeshStandardMaterial({
            color: 0xdb6565,
            roughness: 0.38,
            metalness: 0.08,
            side: THREE.DoubleSide
        });

        function createRectusMuscle(angleZ) {
            const group = new THREE.Group();
            // Smooth curve wrapping the sclera from anterior equator (z ~ 0.5) to posterior (z ~ -1.9)
            const curve = new THREE.CubicBezierCurve3(
                new THREE.Vector3(0, 2.33, 0.45),
                new THREE.Vector3(0, 2.42, -0.3),
                new THREE.Vector3(0, 2.28, -1.2),
                new THREE.Vector3(0, 1.35, -2.1)
            );
            const tubeGeo = new THREE.TubeGeometry(curve, 22, 0.28, 12, false);
            tubeGeo.scale(2.2, 0.52, 1); // Flatten into a ribbon muscle band
            const mesh = new THREE.Mesh(tubeGeo, muscleMat);
            group.add(mesh);
            group.rotation.z = angleZ;
            return group;
        }

        // 4 Principal Rectus Muscles: Superior, Inferior, Lateral, Medial
        const supRectus = createRectusMuscle(0);
        const infRectus = createRectusMuscle(Math.PI);
        const latRectus = createRectusMuscle(Math.PI * 0.5);
        const medRectus = createRectusMuscle(-Math.PI * 0.5);

        modelEye.add(supRectus);
        modelEye.add(infRectus);
        modelEye.add(latRectus);
        modelEye.add(medRectus);

        // 1.9 AI Neural Constellation Hologram Lattice (surrounding eye matching reference image)
        const latticeGeo = new THREE.IcosahedronGeometry(3.35, 1);
        const wireframeGeo = new THREE.WireframeGeometry(latticeGeo);
        const latticeLineMat = new THREE.LineBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.48
        });
        const latticeLines = new THREE.LineSegments(wireframeGeo, latticeLineMat);

        // Bright white/cyan neural nodes at vertices
        const posAttr = latticeGeo.attributes.position;
        const nodeGeo = new THREE.SphereGeometry(0.045, 10, 10);
        const nodeMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const nodeGroup = new THREE.Group();
        const recordedPts = [];

        for (let i = 0; i < posAttr.count; i++) {
            const pt = new THREE.Vector3(posAttr.getX(i), posAttr.getY(i), posAttr.getZ(i));
            if (!recordedPts.some(p => p.distanceTo(pt) < 0.05)) {
                recordedPts.push(pt);
                const nodeMesh = new THREE.Mesh(nodeGeo, nodeMat);
                nodeMesh.position.copy(pt);
                nodeGroup.add(nodeMesh);
            }
        }

        const aiLatticeGroup = new THREE.Group();
        aiLatticeGroup.add(latticeLines);
        aiLatticeGroup.add(nodeGroup);
        modelEye.add(aiLatticeGroup);
        modelEye.userData.aiLattice = aiLatticeGroup;

        // Outer Ophthalmic Alignment Ring
        const ringGeo = new THREE.TorusGeometry(2.8, 0.015, 12, 64);
        const ringMat = new THREE.MeshBasicMaterial({ color: 0x0284c7, transparent: true, opacity: 0.45 });
        const ringMesh = new THREE.Mesh(ringGeo, ringMat);
        modelEye.add(ringMesh);

        masterGroup.add(modelEye);
    }

    /* ============================================================
       MODEL 2: 3D RETINAL FUNDUS & AI DR LESION DETECTION MAP
       ============================================================ */
    function buildModelRetina() {
        modelRetina = new THREE.Group();

        // 2.1 Hemispherical Fundus Bowl (facing user at start)
        const fundusTex = createRetinaFundusTexture();
        const fundusGeo = new THREE.SphereGeometry(2.7, 48, 36, 0, Math.PI * 2, 0, Math.PI * 0.56);
        const fundusMat = new THREE.MeshStandardMaterial({
            map: fundusTex,
            roughness: 0.45,
            side: THREE.BackSide
        });
        const fundusMesh = new THREE.Mesh(fundusGeo, fundusMat);
        fundusMesh.rotation.x = Math.PI;
        modelRetina.add(fundusMesh);

        // Subtle Outer Rim
        const rimGeo = new THREE.TorusGeometry(2.7, 0.03, 12, 64);
        const rimMat = new THREE.MeshBasicMaterial({ color: 0x0d9488 });
        const rimMesh = new THREE.Mesh(rimGeo, rimMat);
        rimMesh.rotation.x = Math.PI / 2;
        rimMesh.position.z = -1.25;
        modelRetina.add(rimMesh);

        // 2.2 3D AI DR Lesion Detection Markers (The core clinical intelligence)
        const LESIONS = [
            {
                id: 'ma-1',
                type: 'Microaneurysm',
                badge: 'Grade 1 • Mild NPDR',
                desc: 'Focal capillary dilatation in deep plexus. AI Conf: 98.7%',
                color: 0xef4444,
                pos: new THREE.Vector3(0.55, 0.35, -1.8),
                radius: 0.08
            },
            {
                id: 'ma-2',
                type: 'Microaneurysm',
                badge: 'Grade 1 • Mild NPDR',
                desc: 'Perifoveal capillary loop micro-lesion. AI Conf: 97.4%',
                color: 0xef4444,
                pos: new THREE.Vector3(-0.45, -0.3, -1.9),
                radius: 0.075
            },
            {
                id: 'dbh-1',
                type: 'Dot & Blot Hemorrhage',
                badge: 'Grade 2 • Mod NPDR',
                desc: 'Ruptured microaneurysm in inner nuclear layer. AI Conf: 99.2%',
                color: 0xdc2626,
                pos: new THREE.Vector3(1.15, -0.55, -1.5),
                radius: 0.12
            },
            {
                id: 'dbh-2',
                type: 'Flame Hemorrhage',
                badge: 'Grade 2 • Mod NPDR',
                desc: 'Superficial retinal nerve fiber layer streak. AI Conf: 96.8%',
                color: 0xb91c1c,
                pos: new THREE.Vector3(-1.1, 0.8, -1.4),
                radius: 0.13
            },
            {
                id: 'he-1',
                type: 'Hard Exudates',
                badge: 'Grade 3 • Severe NPDR',
                desc: 'Circinate yellow lipid lipoprotein rings. AI Conf: 98.9%',
                color: 0xfbbf24,
                pos: new THREE.Vector3(0.3, 0.95, -1.7),
                radius: 0.11
            },
            {
                id: 'cws-1',
                type: 'Cotton Wool Spot',
                badge: 'Grade 3 • Ischemia',
                desc: 'Axoplasmic stasis & nerve fiber layer infarct. AI Conf: 95.8%',
                color: 0xf8fafc,
                pos: new THREE.Vector3(-0.55, 1.25, -1.45),
                radius: 0.14
            },
            {
                id: 'nvd-1',
                type: 'Neovascularization',
                badge: 'Grade 4 • PDR High Risk',
                desc: 'Fragile abnormal neovascular frond at optic disc. AI Conf: 99.5%',
                color: 0xf43f5e,
                pos: new THREE.Vector3(1.3, 0.1, -1.55),
                radius: 0.16
            }
        ];

        lesionBeacons = [];
        const lesionGroup = new THREE.Group();
        lesionGroup.name = 'lesionGroup';

        LESIONS.forEach(les => {
            const bGroup = new THREE.Group();
            bGroup.position.copy(les.pos);

            // Core beacon sphere
            const bGeo = new THREE.SphereGeometry(les.radius, 16, 16);
            const bMat = new THREE.MeshStandardMaterial({
                color: les.color,
                emissive: les.color,
                emissiveIntensity: 0.8,
                roughness: 0.2
            });
            const bMesh = new THREE.Mesh(bGeo, bMat);
            bMesh.userData = les;
            bGroup.add(bMesh);

            // Pulsing target beacon ring
            const rGeo = new THREE.RingGeometry(les.radius * 1.5, les.radius * 1.9, 24);
            const rMat = new THREE.MeshBasicMaterial({
                color: les.color,
                transparent: true,
                opacity: 0.75,
                side: THREE.DoubleSide
            });
            const rMesh = new THREE.Mesh(rGeo, rMat);
            bGroup.add(rMesh);

            // Marker Pin line projecting outward
            const pinGeo = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(0, 0, 0.35)
            ]);
            const pinMat = new THREE.LineBasicMaterial({ color: les.color, transparent: true, opacity: 0.8 });
            const pinLine = new THREE.Line(pinGeo, pinMat);
            bGroup.add(pinLine);

            bGroup.userData = { ring: rMesh, baseScale: 1, ...les };
            lesionBeacons.push(bGroup);
            lesionGroup.add(bGroup);
        });

        modelRetina.add(lesionGroup);

        // 2.3 Sweeping 3D Holographic AI Laser Scan Sweep
        const laserGeo = new THREE.PlaneGeometry(5.2, 0.08);
        const laserMat = new THREE.MeshBasicMaterial({
            color: 0x06d6a0,
            transparent: true,
            opacity: 0.85,
            side: THREE.DoubleSide
        });
        laserSweepMesh = new THREE.Mesh(laserGeo, laserMat);
        laserSweepMesh.position.set(0, 0, -1.8);
        modelRetina.add(laserSweepMesh);

        masterGroup.add(modelRetina);
    }

    /* ============================================================
       MODEL 3: 3D VOLUMETRIC OCT RETINAL LAYER SCANNER
       ============================================================ */
    function buildModelOCT() {
        modelOCT = new THREE.Group();

        // 3.1 Stratified Retinal Slab (3D Curved Block)
        const octTex = createOCTTexture();
        const slabGeo = new THREE.BoxGeometry(4.6, 2.6, 1.2, 16, 8, 8);
        const slabMats = [
            new THREE.MeshStandardMaterial({ color: 0x0f172a }), // right
            new THREE.MeshStandardMaterial({ color: 0x0f172a }), // left
            new THREE.MeshStandardMaterial({ color: 0x1e293b }), // top (vitreous interface)
            new THREE.MeshStandardMaterial({ color: 0x030712 }), // bottom (sclera)
            new THREE.MeshStandardMaterial({ map: octTex, roughness: 0.3 }), // front B-scan face
            new THREE.MeshStandardMaterial({ map: octTex, roughness: 0.3 })  // back face
        ];
        const slabMesh = new THREE.Mesh(slabGeo, slabMats);
        modelOCT.add(slabMesh);

        // 3.2 AI Layer Thickness Caliper Marker (Diabetic Macular Edema: 420 µm)
        const caliperGroup = new THREE.Group();
        caliperGroup.position.set(0.1, 0, 0.65);

        const lineGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(0, 0.9, 0),
            new THREE.Vector3(0, -0.9, 0)
        ]);
        const lineMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 2 });
        const caliperLine = new THREE.Line(lineGeo, lineMat);
        caliperGroup.add(caliperLine);

        // Caliper End Caps
        const capGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-0.2, 0.9, 0), new THREE.Vector3(0.2, 0.9, 0),
            new THREE.Vector3(-0.2, -0.9, 0), new THREE.Vector3(0.2, -0.9, 0)
        ]);
        const capLines = new THREE.LineSegments(capGeo, lineMat);
        caliperGroup.add(capLines);
        modelOCT.add(caliperGroup);

        // 3.3 Dynamic B-Scan Slicing Laser Plane
        const bScanPlaneGeo = new THREE.PlaneGeometry(4.7, 2.7);
        const bScanPlaneMat = new THREE.MeshBasicMaterial({
            color: 0x38bdf8,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide
        });
        laserOCTMesh = new THREE.Mesh(bScanPlaneGeo, bScanPlaneMat);
        laserOCTMesh.position.z = 0;
        modelOCT.add(laserOCTMesh);

        masterGroup.add(modelOCT);
    }

    /* ============================================================
       MODEL 4: 3D DIGITAL AI FUNDUS CAMERA & OPTICAL RIG
       ============================================================ */
    function buildModelCamera() {
        modelCamera = new THREE.Group();

        // 4.1 Ophthalmic Front Objective Barrel
        const barrelGeo = new THREE.CylinderGeometry(1.2, 1.4, 2.2, 32);
        const barrelMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            metalness: 0.7,
            roughness: 0.25
        });
        const barrelMesh = new THREE.Mesh(barrelGeo, barrelMat);
        barrelMesh.rotation.x = Math.PI / 2;
        modelCamera.add(barrelMesh);

        // Front anti-reflective optical glass lens element
        const lensGeo = new THREE.SphereGeometry(1.15, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.3);
        const lensMat = new THREE.MeshPhysicalMaterial({
            color: 0x38bdf8,
            transmission: 0.92,
            roughness: 0.05,
            ior: 1.52,
            reflectivity: 0.8
        });
        const frontLens = new THREE.Mesh(lensGeo, lensMat);
        frontLens.position.set(0, 0, 1.05);
        modelCamera.add(frontLens);

        // 4.2 Annular LED Flash Illumination Ring
        const ledRingGeo = new THREE.TorusGeometry(1.35, 0.08, 12, 36);
        const ledRingMat = new THREE.MeshBasicMaterial({ color: 0x06d6a0 });
        const ledRingMesh = new THREE.Mesh(ledRingGeo, ledRingMat);
        ledRingMesh.position.set(0, 0, 1.05);
        modelCamera.add(ledRingMesh);

        // 4.3 Main Camera Body Chassis (Cube)
        const bodyGeo = new THREE.BoxGeometry(3.0, 2.4, 2.2);
        const bodyMat = new THREE.MeshStandardMaterial({
            color: 0x0f172a,
            metalness: 0.5,
            roughness: 0.3
        });
        const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
        bodyMesh.position.set(0, 0, -2.0);
        modelCamera.add(bodyMesh);

        // Accent medical stripe
        const stripeGeo = new THREE.BoxGeometry(3.05, 0.15, 2.25);
        const stripeMat = new THREE.MeshBasicMaterial({ color: 0x0d9488 });
        const stripeMesh = new THREE.Mesh(stripeGeo, stripeMat);
        stripeMesh.position.set(0, 0.4, -2.0);
        modelCamera.add(stripeMesh);

        // 4.4 Projected Holographic Fundus Targeting Beam
        const beamGeo = new THREE.ConeGeometry(1.2, 3.5, 32, 1, true);
        const beamMat = new THREE.MeshBasicMaterial({
            color: 0x0d9488,
            transparent: true,
            opacity: 0.22,
            side: THREE.DoubleSide
        });
        const beamMesh = new THREE.Mesh(beamGeo, beamMat);
        beamMesh.position.set(0, 0, 2.8);
        beamMesh.rotation.x = -Math.PI / 2;
        modelCamera.add(beamMesh);

        // Target Reticle Crosshairs
        const crossGeo = new THREE.BufferGeometry().setFromPoints([
            new THREE.Vector3(-0.6, 0, 4.4), new THREE.Vector3(0.6, 0, 4.4),
            new THREE.Vector3(0, -0.6, 4.4), new THREE.Vector3(0, 0.6, 4.4)
        ]);
        const crossMat = new THREE.LineBasicMaterial({ color: 0x06d6a0, linewidth: 2 });
        const crossMesh = new THREE.LineSegments(crossGeo, crossMat);
        modelCamera.add(crossMesh);

        masterGroup.add(modelCamera);
    }

    /* ============================================================
       MODEL SWITCHING & HUD SYNCHRONIZATION
       ============================================================ */
    function switchModel(modelId, animateTransition = true) {
        activeModelId = modelId;

        const models = { eye: modelEye, retina: modelRetina, oct: modelOCT, camera: modelCamera };

        Object.keys(models).forEach(key => {
            if (models[key]) {
                models[key].visible = (key === modelId);
            }
        });

        // Update toolbar active buttons
        document.querySelectorAll('.hero-3d-model-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.model === modelId);
        });

        // Update HUD Telemetry tags
        const data = MODEL_TELEMETRY[modelId];
        if (data) {
            const elL1 = document.getElementById('ai-marker-label-1');
            const elV1 = document.getElementById('ai-marker-val-1');
            const elS1 = document.getElementById('ai-marker-sub-1');
            const elL2 = document.getElementById('ai-marker-label-2');
            const elV2 = document.getElementById('ai-marker-val-2');
            const elS2 = document.getElementById('ai-marker-sub-2');
            const elLaser = document.getElementById('scan-laser-text');

            if (elL1) elL1.textContent = data.label1;
            if (elV1) elV1.textContent = data.val1;
            if (elS1) elS1.textContent = data.sub1;
            if (elL2) elL2.textContent = data.label2;
            if (elV2) elV2.textContent = data.val2;
            if (elS2) elS2.textContent = data.sub2;
            if (elLaser) elLaser.textContent = data.laser;
        }

        // Adjust default camera distance per model
        if (modelId === 'camera') orbit.targetZoom = 9.5;
        else if (modelId === 'oct') orbit.targetZoom = 7.5;
        else if (modelId === 'retina') orbit.targetZoom = 7.8;
        else orbit.targetZoom = 8.5;

        // Reset rotation gracefully
        orbit.targetRotX = 0.15;
        orbit.targetRotY = (modelId === 'retina') ? 0 : -0.2;
    }

    /* ============================================================
       INTERACTION & TOOLBAR HANDLERS
       ============================================================ */
    function setupToolbarListeners() {
        // 1. Model switch buttons
        document.querySelectorAll('.hero-3d-model-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const target = btn.dataset.model;
                if (target) switchModel(target, true);
            });
        });

        // 2. Cutaway toggle
        const btnCross = document.getElementById('btn-toggle-cross');
        if (btnCross) {
            btnCross.addEventListener('click', () => {
                showCutaway = !showCutaway;
                btnCross.classList.toggle('active', showCutaway);
                if (modelEye && modelEye.userData.fullMesh && modelEye.userData.cutMesh) {
                    modelEye.userData.fullMesh.visible = !showCutaway;
                    modelEye.userData.cutMesh.visible = showCutaway;
                }
                if (window.showPortalToast) {
                    window.showPortalToast(showCutaway ? 'Cross-section cutaway revealed' : 'Full anatomical globe restored', 'info');
                }
            });
        }

        // 3. AI Lesions toggle
        const btnLesions = document.getElementById('btn-toggle-lesions');
        if (btnLesions) {
            btnLesions.addEventListener('click', () => {
                showLesions = !showLesions;
                btnLesions.classList.toggle('active', showLesions);
                if (modelRetina) {
                    const lGroup = modelRetina.getObjectByName('lesionGroup');
                    if (lGroup) lGroup.visible = showLesions;
                }
                if (window.showPortalToast) {
                    window.showPortalToast(showLesions ? 'AI DR Lesion beacons visible' : 'AI Lesion markers hidden', 'info');
                }
            });
        }

        // 4. Reset View button
        const btnReset = document.getElementById('btn-reset-3d');
        if (btnReset) {
            btnReset.addEventListener('click', () => {
                orbit.targetRotX = 0.15;
                orbit.targetRotY = 0;
                orbit.targetZoom = 8.5;
            });
        }
    }

    function setupEventListeners() {
        if (!container) return;

        // Mouse Drag Orbit
        container.addEventListener('mousedown', (e) => {
            if (e.target.closest('.hero-3d-model-bar') || e.target.closest('.hero-3d-tools-row')) return;
            orbit.isDragging = true;
            orbit.prevMouseX = e.clientX;
            orbit.prevMouseY = e.clientY;
            autoRotate = false;
        });

        window.addEventListener('mousemove', (e) => {
            const rect = container.getBoundingClientRect();
            // Raycasting coordinate calculation
            if (e.clientX >= rect.left && e.clientX <= rect.right &&
                e.clientY >= rect.top && e.clientY <= rect.bottom) {
                mouseNormalized.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                mouseNormalized.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
            } else {
                mouseNormalized.x = -999;
                mouseNormalized.y = -999;
            }

            if (!orbit.isDragging) return;

            const deltaX = e.clientX - orbit.prevMouseX;
            const deltaY = e.clientY - orbit.prevMouseY;

            orbit.targetRotY += deltaX * 0.008;
            orbit.targetRotX += deltaY * 0.008;

            // Clamp vertical rotation
            orbit.targetRotX = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, orbit.targetRotX));

            orbit.prevMouseX = e.clientX;
            orbit.prevMouseY = e.clientY;
        });

        window.addEventListener('mouseup', () => {
            orbit.isDragging = false;
            setTimeout(() => { autoRotate = true; }, 4000);
        });

        // Touch support
        container.addEventListener('touchstart', (e) => {
            if (e.touches.length === 1) {
                orbit.isDragging = true;
                orbit.prevMouseX = e.touches[0].clientX;
                orbit.prevMouseY = e.touches[0].clientY;
                autoRotate = false;
            }
        }, { passive: true });

        window.addEventListener('touchmove', (e) => {
            if (!orbit.isDragging || e.touches.length !== 1) return;
            const deltaX = e.touches[0].clientX - orbit.prevMouseX;
            const deltaY = e.touches[0].clientY - orbit.prevMouseY;

            orbit.targetRotY += deltaX * 0.008;
            orbit.targetRotX += deltaY * 0.008;
            orbit.targetRotX = Math.max(-Math.PI / 2.2, Math.min(Math.PI / 2.2, orbit.targetRotX));

            orbit.prevMouseX = e.touches[0].clientX;
            orbit.prevMouseY = e.touches[0].clientY;
        }, { passive: true });

        window.addEventListener('touchend', () => {
            orbit.isDragging = false;
            setTimeout(() => { autoRotate = true; }, 4000);
        });

        // Wheel Zoom
        container.addEventListener('wheel', (e) => {
            e.preventDefault();
            orbit.targetZoom += e.deltaY * 0.006;
            orbit.targetZoom = Math.max(4.5, Math.min(14.0, orbit.targetZoom));
        }, { passive: false });

        // Resize Listener
        window.addEventListener('resize', onWindowResize);
    }

    function onWindowResize() {
        if (!container || !camera || !renderer) return;
        const width = container.clientWidth || 560;
        const height = container.clientHeight || 560;

        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }

    /* ============================================================
       RENDER & ANIMATION LOOP
       ============================================================ */
    function animate(time) {
        animationFrameId = requestAnimationFrame(animate);

        const delta = 0.016;

        // Smooth spring lerp for rotation & zoom
        orbit.rotX += (orbit.targetRotX - orbit.rotX) * 0.1;
        orbit.rotY += (orbit.targetRotY - orbit.rotY) * 0.1;
        orbit.zoom += (orbit.targetZoom - orbit.zoom) * 0.1;

        camera.position.z = orbit.zoom;

        if (masterGroup) {
            masterGroup.rotation.x = orbit.rotX;
            masterGroup.rotation.y = orbit.rotY;

            // Gentle auto-rotation when idle
            if (autoRotate && !orbit.isDragging) {
                orbit.targetRotY += 0.0025;
            }
        }

        // 0. Animate AI Neural Constellation Lattice (Model 1: Eye matching reference image)
        if (modelEye && modelEye.userData.aiLattice && activeModelId === 'eye') {
            modelEye.userData.aiLattice.rotation.y += 0.0022;
            modelEye.userData.aiLattice.rotation.x += 0.0012;
            const lines = modelEye.userData.aiLattice.children[0];
            if (lines && lines.material) {
                lines.material.opacity = 0.42 + Math.sin(time * 0.0025) * 0.18;
            }
        }

        // 1. Animate Retinal Sweeping Laser
        if (laserSweepMesh && activeModelId === 'retina') {
            laserSweepMesh.position.y = Math.sin(time * 0.002) * 1.6;
            laserSweepMesh.rotation.z = Math.sin(time * 0.001) * 0.1;
        }

        // 2. Animate OCT B-Scan laser plane
        if (laserOCTMesh && activeModelId === 'oct') {
            laserOCTMesh.position.z = Math.sin(time * 0.0025) * 0.5;
        }

        // 3. Pulse 3D AI DR Lesion Beacon rings
        if (activeModelId === 'retina' && showLesions) {
            lesionBeacons.forEach((bGroup, i) => {
                const ring = bGroup.userData.ring;
                if (ring) {
                    const scale = 1 + Math.sin(time * 0.004 + i) * 0.35;
                    ring.scale.set(scale, scale, 1);
                    ring.material.opacity = 0.4 + Math.cos(time * 0.004 + i) * 0.35;
                }
            });

            // Raycasting for hovered lesion marker
            checkLesionRaycasting();
        } else {
            hideTooltip();
        }

        renderer.render(scene, camera);
    }

    function checkLesionRaycasting() {
        if (!tooltipEl || mouseNormalized.x === -999) {
            hideTooltip();
            return;
        }

        raycaster.setFromCamera(mouseNormalized, camera);

        const meshes = lesionBeacons.map(b => b.children[0]).filter(Boolean);
        const intersects = raycaster.intersectObjects(meshes, false);

        if (intersects.length > 0) {
            const hitMesh = intersects[0].object;
            const data = hitMesh.userData;

            if (data && data.type) {
                showTooltip(data, intersects[0].point);
                return;
            }
        }

        hideTooltip();
    }

    function showTooltip(data, worldPos) {
        if (!tooltipEl) return;

        const titleEl = document.getElementById('tooltip-title');
        const badgeEl = document.getElementById('tooltip-badge');
        const descEl = document.getElementById('tooltip-desc');

        if (titleEl) titleEl.textContent = data.type;
        if (badgeEl) {
            badgeEl.textContent = data.badge;
            badgeEl.style.background = '#' + data.color.toString(16).padStart(6, '0') + '25';
            badgeEl.style.color = '#' + data.color.toString(16).padStart(6, '0');
        }
        if (descEl) descEl.textContent = data.desc;

        // Project 3D point to 2D screen coordinate
        const screenPos = worldPos.clone().project(camera);
        const halfW = container.clientWidth / 2;
        const halfH = container.clientHeight / 2;
        const px = screenPos.x * halfW + halfW;
        const py = -(screenPos.y * halfH) + halfH;

        tooltipEl.style.left = `${px + 15}px`;
        tooltipEl.style.top = `${py - 20}px`;
        tooltipEl.classList.add('visible');
    }

    function hideTooltip() {
        if (tooltipEl) tooltipEl.classList.remove('visible');
    }

})();
