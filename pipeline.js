/* ============================================================
   DR Screening Pipeline — Image Processing & Pipeline Logic
   ============================================================
   JavaScript port of the MATLAB pipeline (rule-based mode, no CNN).
   All image processing runs client-side via Canvas 2D API.
   ============================================================ */

class DRPipeline {
    constructor() {
        this.imageData = null;
        this.width = 0;
        this.height = 0;
        this.result = null;
    }

    /** Load an image from a File or Blob into raw pixel data */
    async loadImage(file) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                // Work at a manageable resolution for in-browser processing
                const maxDim = 512;
                let w = img.width, h = img.height;
                if (w > maxDim || h > maxDim) {
                    const scale = maxDim / Math.max(w, h);
                    w = Math.round(w * scale);
                    h = Math.round(h * scale);
                }
                const canvas = document.createElement('canvas');
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);
                this.imageData = ctx.getImageData(0, 0, w, h);
                this.width = w;
                this.height = h;
                resolve({ width: w, height: h, imageData: this.imageData });
            };
            img.onerror = reject;
            img.src = URL.createObjectURL(file);
        });
    }

    // ===================== HELPER UTILITIES =====================

    /** Extract a single channel as a Float64 2D array [row][col] in [0,1] */
    _channel(channelIdx) {
        const d = this.imageData.data;
        const out = [];
        for (let y = 0; y < this.height; y++) {
            const row = new Float64Array(this.width);
            for (let x = 0; x < this.width; x++) {
                row[x] = d[(y * this.width + x) * 4 + channelIdx] / 255;
            }
            out.push(row);
        }
        return out;
    }

    _grayscale() {
        const d = this.imageData.data;
        const out = [];
        for (let y = 0; y < this.height; y++) {
            const row = new Float64Array(this.width);
            for (let x = 0; x < this.width; x++) {
                const i = (y * this.width + x) * 4;
                row[x] = (0.2989 * d[i] + 0.5870 * d[i+1] + 0.1140 * d[i+2]) / 255;
            }
            out.push(row);
        }
        return out;
    }

    /** Mean of 2D array values */
    _mean2D(arr) {
        let sum = 0, n = 0;
        for (const row of arr) for (const v of row) { sum += v; n++; }
        return sum / n;
    }

    /** Std of 2D array values */
    _std2D(arr) {
        const mean = this._mean2D(arr);
        let sumSq = 0, n = 0;
        for (const row of arr) for (const v of row) { sumSq += (v - mean) ** 2; n++; }
        return Math.sqrt(sumSq / n);
    }

    /** Variance of masked values in 2D array */
    _varMasked(arr, mask) {
        let sum = 0, n = 0;
        for (let y = 0; y < arr.length; y++)
            for (let x = 0; x < arr[0].length; x++)
                if (!mask || mask[y][x]) { sum += arr[y][x]; n++; }
        if (n === 0) return 0;
        const mean = sum / n;
        let sumSq = 0;
        for (let y = 0; y < arr.length; y++)
            for (let x = 0; x < arr[0].length; x++)
                if (!mask || mask[y][x]) sumSq += (arr[y][x] - mean) ** 2;
        return sumSq / n;
    }

    _meanMasked(arr, mask) {
        let sum = 0, n = 0;
        for (let y = 0; y < arr.length; y++)
            for (let x = 0; x < arr[0].length; x++)
                if (!mask || mask[y][x]) { sum += arr[y][x]; n++; }
        return n > 0 ? sum / n : 0;
    }

    /** Gaussian blur (separable, sigma pixels) */
    _gaussBlur(arr, sigma) {
        const h = arr.length, w = arr[0].length;
        const kSize = Math.max(3, Math.ceil(sigma * 3) * 2 + 1);
        const half = Math.floor(kSize / 2);
        const kernel = new Float64Array(kSize);
        let kSum = 0;
        for (let i = 0; i < kSize; i++) {
            const x = i - half;
            kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
            kSum += kernel[i];
        }
        for (let i = 0; i < kSize; i++) kernel[i] /= kSum;

        // Horizontal pass
        const tmp = Array.from({ length: h }, () => new Float64Array(w));
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let sum = 0;
                for (let k = 0; k < kSize; k++) {
                    const xx = Math.min(w - 1, Math.max(0, x + k - half));
                    sum += arr[y][xx] * kernel[k];
                }
                tmp[y][x] = sum;
            }
        }
        // Vertical pass
        const out = Array.from({ length: h }, () => new Float64Array(w));
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                let sum = 0;
                for (let k = 0; k < kSize; k++) {
                    const yy = Math.min(h - 1, Math.max(0, y + k - half));
                    sum += tmp[yy][x] * kernel[k];
                }
                out[y][x] = sum;
            }
        }
        return out;
    }

    /** Laplacian filter response */
    _laplacian(arr) {
        const h = arr.length, w = arr[0].length;
        const out = Array.from({ length: h }, () => new Float64Array(w));
        for (let y = 1; y < h - 1; y++) {
            for (let x = 1; x < w - 1; x++) {
                out[y][x] = arr[y-1][x] + arr[y+1][x] + arr[y][x-1] + arr[y][x+1] - 4 * arr[y][x];
            }
        }
        return out;
    }

    /** Otsu threshold (simplified for 1D histogram) */
    _otsu(arr) {
        const bins = 256;
        const hist = new Float64Array(bins);
        let n = 0;
        for (const row of arr) for (const v of row) {
            hist[Math.min(bins - 1, Math.max(0, Math.floor(v * 255)))]++;
            n++;
        }
        let sumTotal = 0;
        for (let i = 0; i < bins; i++) sumTotal += i * hist[i];
        let sumB = 0, wB = 0, wF, maxVar = 0, threshold = 0;
        for (let t = 0; t < bins; t++) {
            wB += hist[t];
            if (wB === 0) continue;
            wF = n - wB;
            if (wF === 0) break;
            sumB += t * hist[t];
            const mB = sumB / wB;
            const mF = (sumTotal - sumB) / wF;
            const between = wB * wF * (mB - mF) ** 2;
            if (between > maxVar) {
                maxVar = between;
                threshold = t;
            }
        }
        return threshold / 255;
    }

    /** Connected-component labeling (4-connected) on a boolean mask */
    _connectedComponents(mask) {
        const h = mask.length, w = mask[0].length;
        const labels = Array.from({ length: h }, () => new Int32Array(w));
        let nextLabel = 1;
        const components = {};

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                if (!mask[y][x] || labels[y][x] !== 0) continue;
                // BFS
                const queue = [[y, x]];
                const pixels = [];
                labels[y][x] = nextLabel;
                while (queue.length > 0) {
                    const [cy, cx] = queue.shift();
                    pixels.push([cy, cx]);
                    for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
                        const ny = cy + dy, nx = cx + dx;
                        if (ny >= 0 && ny < h && nx >= 0 && nx < w && mask[ny][nx] && labels[ny][nx] === 0) {
                            labels[ny][nx] = nextLabel;
                            queue.push([ny, nx]);
                        }
                    }
                }
                components[nextLabel] = pixels;
                nextLabel++;
            }
        }
        return { labels, components, count: nextLabel - 1 };
    }

    // ============================================================
    // STAGE 1: Quality Assessment
    // Ports: assessImageQuality.m
    // ============================================================
    assessImageQuality() {
        const gray = this._grayscale();
        const green = this._channel(1);
        const h = this.height, w = this.width;
        const reasons = [];

        // 1. Field of view
        const thresh = this._otsu(gray) * 0.4;
        const fovMask = Array.from({ length: h }, (_, y) =>
            Array.from({ length: w }, (_, x) => gray[y][x] > thresh)
        );
        // Find largest component
        const cc = this._connectedComponents(fovMask);
        let maxArea = 0;
        for (const key in cc.components) {
            maxArea = Math.max(maxArea, cc.components[key].length);
        }
        const fovFrac = maxArea / (h * w);
        const fovScore = Math.min(1, fovFrac / 0.60);
        if (fovScore < 0.5) reasons.push('Retinal field of view too small / eyelid occlusion suspected');

        // 2. Sharpness (variance of Laplacian)
        const lapResp = this._laplacian(green);
        const sharpnessRaw = this._varMasked(lapResp, null);
        const sharpnessScore = Math.min(1, sharpnessRaw / 0.002);
        if (sharpnessScore < 0.35) reasons.push('Image out of focus / motion blur');

        // 3. Illumination uniformity
        const gridN = 4;
        const rh = Math.floor(h / gridN), rw = Math.floor(w / gridN);
        const tileMeans = [];
        for (let r = 0; r < gridN; r++) {
            for (let c = 0; c < gridN; c++) {
                let sum = 0, cnt = 0;
                for (let y = r * rh; y < (r + 1) * rh; y++) {
                    for (let x = c * rw; x < (c + 1) * rw; x++) {
                        sum += green[y][x]; cnt++;
                    }
                }
                tileMeans.push(sum / cnt);
            }
        }
        const tmMean = tileMeans.reduce((a, b) => a + b, 0) / tileMeans.length;
        const tmStd = Math.sqrt(tileMeans.reduce((a, v) => a + (v - tmMean) ** 2, 0) / tileMeans.length);
        const illumSpread = tmStd / (tmMean + 1e-10);
        let illumScore = Math.max(0, 1 - illumSpread / 0.5);

        const meanBrightness = this._meanMasked(green, null);
        if (meanBrightness < 0.15 || meanBrightness > 0.9) {
            illumScore *= 0.4;
            reasons.push('Under- or over-exposed image');
        }
        if (illumScore < 0.4) reasons.push('Uneven illumination / glare / vignetting');

        // 4. Contrast
        const contrastRaw = this._std2D(green);
        const contrastScore = Math.min(1, contrastRaw / 0.18);
        if (contrastScore < 0.3) reasons.push('Low contrast image');

        // Weighted overall
        const overall = 0.30 * fovScore + 0.35 * sharpnessScore + 0.20 * illumScore + 0.15 * contrastScore;
        let isGradable = overall >= 0.55 && reasons.length === 0;
        let needsEnhancement = false;
        let recaptureMsg = '';

        if (isGradable) {
            needsEnhancement = overall < 0.75;
            recaptureMsg = needsEnhancement ? 'Mild enhancement applied' : '';
        } else if (overall >= 0.35) {
            isGradable = true;
            needsEnhancement = true;
            recaptureMsg = 'Borderline quality — automatic enhancement applied. Issues: ' + reasons.join('; ');
        } else {
            recaptureMsg = 'REJECTED — please recapture. Issues: ' + reasons.join('; ');
            needsEnhancement = false;
        }

        return {
            score: overall,
            sharpness: sharpnessScore,
            illumination: illumScore,
            fov: fovScore,
            contrast: contrastScore,
            isGradable,
            needsEnhancement,
            reasons,
            recaptureMsg
        };
    }

    // ============================================================
    // STAGE 2: Structure Segmentation (simplified JS port)
    // ============================================================

    /** Detect optic disc (brightest circular region) and fovea */
    detectOpticDiscFovea() {
        const red = this._channel(0);
        const green = this._channel(1);
        const h = this.height, w = this.width;

        // Bright channel: 0.75*red + 0.25*green
        const bright = Array.from({ length: h }, (_, y) =>
            new Float64Array(Array.from({ length: w }, (_, x) => 0.75 * red[y][x] + 0.25 * green[y][x]))
        );
        const smoothed = this._gaussBlur(bright, 5);

        // Find 97th percentile brightest region
        const allVals = [];
        for (const row of smoothed) for (const v of row) allVals.push(v);
        allVals.sort((a, b) => a - b);
        const p97 = allVals[Math.floor(allVals.length * 0.97)];

        const brightMask = Array.from({ length: h }, (_, y) =>
            Array.from({ length: w }, (_, x) => smoothed[y][x] >= p97)
        );
        const cc = this._connectedComponents(brightMask);

        let bestLabel = null, bestArea = 0;
        for (const key in cc.components) {
            if (cc.components[key].length > bestArea) {
                bestArea = cc.components[key].length;
                bestLabel = key;
            }
        }

        let discCenter, discRadius;
        if (bestLabel && bestArea > 10) {
            const pixels = cc.components[bestLabel];
            let sumX = 0, sumY = 0;
            for (const [py, px] of pixels) { sumX += px; sumY += py; }
            discCenter = [sumX / pixels.length, sumY / pixels.length];
            discRadius = Math.sqrt(bestArea / Math.PI);
        } else {
            discCenter = [w / 2, h / 2];
            discRadius = w * 0.06;
        }

        // Fovea: darkest region offset from disc
        const dd = discRadius * 2;
        let searchCx;
        if (discCenter[0] > w / 2) {
            searchCx = discCenter[0] - 2.5 * dd;
        } else {
            searchCx = discCenter[0] + 2.5 * dd;
        }
        const searchCy = discCenter[1];
        const winR = dd * 1.5;

        const x1 = Math.max(0, Math.round(searchCx - winR));
        const x2 = Math.min(w - 1, Math.round(searchCx + winR));
        const y1 = Math.max(0, Math.round(searchCy - winR));
        const y2 = Math.min(h - 1, Math.round(searchCy + winR));

        let minVal = Infinity, foveaX = searchCx, foveaY = searchCy;
        const greenSmooth = this._gaussBlur(green, 7);
        for (let y = y1; y <= y2; y++) {
            for (let x = x1; x <= x2; x++) {
                if (greenSmooth[y][x] < minVal) {
                    minVal = greenSmooth[y][x];
                    foveaX = x;
                    foveaY = y;
                }
            }
        }

        return {
            disc: { center: discCenter, radius: discRadius },
            fovea: { center: [foveaX, foveaY] }
        };
    }

    /** Simplified vessel segmentation via threshold on inverted CLAHE green channel */
    segmentVessels() {
        const green = this._channel(1);
        const h = this.height, w = this.width;

        // Invert green channel (vessels become bright)
        const inv = Array.from({ length: h }, (_, y) =>
            new Float64Array(Array.from({ length: w }, (_, x) => 1 - green[y][x]))
        );

        // Subtract background
        const bg = this._gaussBlur(inv, Math.max(3, Math.round(Math.min(w, h) / 15)));
        const enhanced = Array.from({ length: h }, (_, y) =>
            new Float64Array(Array.from({ length: w }, (_, x) => {
                const v = inv[y][x] - bg[y][x] + 0.5;
                return Math.max(0, Math.min(1, v));
            }))
        );

        // Multi-scale Hessian vesselness (simplified — 2 scales)
        const vesselness = Array.from({ length: h }, () => new Float64Array(w));
        for (const sigma of [1.5, 2.5, 3.5]) {
            const smoothed = this._gaussBlur(enhanced, sigma);
            // Approximate Hessian via second differences
            for (let y = 1; y < h - 1; y++) {
                for (let x = 1; x < w - 1; x++) {
                    const dxx = smoothed[y][x-1] - 2 * smoothed[y][x] + smoothed[y][x+1];
                    const dyy = smoothed[y-1][x] - 2 * smoothed[y][x] + smoothed[y+1][x];
                    const dxy = 0.25 * (smoothed[y-1][x-1] - smoothed[y-1][x+1] - smoothed[y+1][x-1] + smoothed[y+1][x+1]);

                    const tmp = Math.sqrt((dxx - dyy) ** 2 + 4 * dxy * dxy);
                    const l1 = 0.5 * (dxx + dyy + tmp);
                    const l2 = 0.5 * (dxx + dyy - tmp);

                    // Only dark-on-bright ridges (l2 < 0 after inversion = bright ridges)
                    const la1 = Math.abs(l1) <= Math.abs(l2) ? l1 : l2;
                    const la2 = Math.abs(l1) <= Math.abs(l2) ? l2 : l1;

                    if (la2 > 0) continue;

                    const Rb = (la1 / (la2 + 1e-10)) ** 2;
                    const S2 = la1 ** 2 + la2 ** 2;
                    const Vo = Math.exp(-Rb / 0.5) * (1 - Math.exp(-S2 / (0.5 * 0.01)));
                    vesselness[y][x] = Math.max(vesselness[y][x], Vo * sigma);
                }
            }
        }

        // Normalize
        let maxV = 0;
        for (const row of vesselness) for (const v of row) maxV = Math.max(maxV, v);
        if (maxV > 0) for (const row of vesselness) for (let i = 0; i < w; i++) row[i] /= maxV;

        // Threshold
        const t = this._otsu(vesselness) * 0.8;
        const mask = Array.from({ length: h }, (_, y) =>
            Array.from({ length: w }, (_, x) => vesselness[y][x] > t)
        );

        // Count vessel density
        let vesselPixels = 0;
        for (const row of mask) for (const v of row) if (v) vesselPixels++;
        const vesselDensity = vesselPixels / (h * w);

        return { mask, vesselness, vesselDensity };
    }

    /** Simplified microaneurysm detection via dark spot finding */
    detectMicroaneurysms(discInfo, vesselMask) {
        const green = this._channel(1);
        const h = this.height, w = this.width;

        // Invert
        const inv = Array.from({ length: h }, (_, y) =>
            new Float64Array(Array.from({ length: w }, (_, x) => 1 - green[y][x]))
        );

        // Background subtraction (top-hat equivalent)
        const bg = this._gaussBlur(inv, 4);
        const tophat = Array.from({ length: h }, (_, y) =>
            new Float64Array(Array.from({ length: w }, (_, x) => Math.max(0, inv[y][x] - bg[y][x])))
        );

        // Normalize
        let maxTH = 0;
        for (const row of tophat) for (const v of row) maxTH = Math.max(maxTH, v);
        if (maxTH > 0) for (const row of tophat) for (let i = 0; i < w; i++) row[i] /= maxTH;

        // Threshold
        const thresh = this._otsu(tophat) * 1.4;
        const maList = [];

        // Find candidates as local maxima above threshold
        for (let y = 3; y < h - 3; y++) {
            for (let x = 3; x < w - 3; x++) {
                if (tophat[y][x] <= thresh) continue;
                if (vesselMask && vesselMask[y] && vesselMask[y][x]) continue;

                // Check if near optic disc — skip
                if (discInfo) {
                    const dx = x - discInfo.center[0];
                    const dy = y - discInfo.center[1];
                    if (dx * dx + dy * dy < (discInfo.radius * 1.3) ** 2) continue;
                }

                // Local max check (3x3)
                let isMax = true;
                for (let dy = -1; dy <= 1 && isMax; dy++) {
                    for (let dx = -1; dx <= 1 && isMax; dx++) {
                        if (dy === 0 && dx === 0) continue;
                        if (tophat[y + dy][x + dx] >= tophat[y][x]) isMax = false;
                    }
                }
                if (!isMax) continue;

                const strength = Math.min(1, tophat[y][x] / (thresh + 0.01) - 0.5);
                if (strength > 0) {
                    maList.push({
                        position: [x, y],
                        radius: 2 + Math.random() * 2,
                        strength: Math.max(0, Math.min(1, strength))
                    });
                }
            }
        }

        // Limit to top candidates by strength
        maList.sort((a, b) => b.strength - a.strength);
        return maList.slice(0, 50);
    }

    /** Simplified exudate detection (bright lesions) */
    segmentExudates(discInfo) {
        const green = this._channel(1);
        const h = this.height, w = this.width;

        // Morphological reconstruction approximation: subtract smoothed from original
        const bg = this._gaussBlur(green, 8);
        const residual = Array.from({ length: h }, (_, y) =>
            new Float64Array(Array.from({ length: w }, (_, x) => Math.max(0, green[y][x] - bg[y][x])))
        );

        let maxR = 0;
        for (const row of residual) for (const v of row) maxR = Math.max(maxR, v);
        if (maxR > 0) for (const row of residual) for (let i = 0; i < w; i++) row[i] /= maxR;

        const thresh = Math.max(this._otsu(residual) * 1.5, 0.15);
        const mask = Array.from({ length: h }, (_, y) =>
            Array.from({ length: w }, (_, x) => {
                if (residual[y][x] <= thresh) return false;
                // Exclude disc area
                if (discInfo) {
                    const dx = x - discInfo.center[0];
                    const dy = y - discInfo.center[1];
                    if (dx * dx + dy * dy < (discInfo.radius * 1.3) ** 2) return false;
                }
                return true;
            })
        );

        // Count
        let totalArea = 0, fovArea = 0;
        for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
            fovArea++;
            if (mask[y][x]) totalArea++;
        }

        const cc = this._connectedComponents(mask);

        return {
            mask,
            count: cc.count,
            totalArea,
            fractionOfFOV: totalArea / Math.max(fovArea, 1)
        };
    }

    /** Simplified hemorrhage detection (dark lesions, larger than MAs) */
    detectHemorrhages(discInfo, vesselMask) {
        const green = this._channel(1);
        const h = this.height, w = this.width;

        const inv = Array.from({ length: h }, (_, y) =>
            new Float64Array(Array.from({ length: w }, (_, x) => 1 - green[y][x]))
        );

        const bg = this._gaussBlur(inv, 6);
        const residual = Array.from({ length: h }, (_, y) =>
            new Float64Array(Array.from({ length: w }, (_, x) => Math.max(0, inv[y][x] - bg[y][x])))
        );

        let maxR = 0;
        for (const row of residual) for (const v of row) maxR = Math.max(maxR, v);
        if (maxR > 0) for (const row of residual) for (let i = 0; i < w; i++) row[i] /= maxR;

        const thresh = Math.max(this._otsu(residual) * 1.3, 0.12);
        const candMask = Array.from({ length: h }, (_, y) =>
            Array.from({ length: w }, (_, x) => {
                if (residual[y][x] <= thresh) return false;
                // Exclude vessels
                if (vesselMask && vesselMask[y] && vesselMask[y][x]) return false;
                // Exclude disc
                if (discInfo) {
                    const dx = x - discInfo.center[0];
                    const dy = y - discInfo.center[1];
                    if (dx * dx + dy * dy < (discInfo.radius * 1.3) ** 2) return false;
                }
                return true;
            })
        );

        const cc = this._connectedComponents(candMask);
        let dotCount = 0, blotCount = 0;
        const lesions = [];
        const maxPixels = h * w * 0.01;

        for (const key in cc.components) {
            const pixels = cc.components[key];
            if (pixels.length < 5 || pixels.length > maxPixels) continue;

            // Centroid
            let sumX = 0, sumY = 0;
            for (const [py, px] of pixels) { sumX += px; sumY += py; }
            const cx = sumX / pixels.length, cy = sumY / pixels.length;

            // Rough shape measure
            const area = pixels.length;
            const isDot = area <= 30;

            if (isDot) {
                dotCount++;
                lesions.push({ area, type: 'dot', position: [cx, cy], severity: 0.3 });
            } else {
                blotCount++;
                lesions.push({ area, type: 'blot', position: [cx, cy], severity: 0.7 });
            }
        }

        return { lesions, dotCount, blotCount, totalCount: dotCount + blotCount };
    }

    /** Neovascularization heuristic (vessel density anomaly) */
    detectNeovascularization(vesselMask, discInfo) {
        const h = this.height, w = this.width;
        // Simplified: check if vessel density near disc is anomalously high
        let nearDiscVessels = 0, nearDiscTotal = 0;
        let globalVessels = 0, globalTotal = 0;

        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const isVessel = vesselMask[y] && vesselMask[y][x];
                globalTotal++;
                if (isVessel) globalVessels++;

                if (discInfo) {
                    const dx = x - discInfo.center[0];
                    const dy = y - discInfo.center[1];
                    if (dx * dx + dy * dy < (discInfo.radius * 3) ** 2) {
                        nearDiscTotal++;
                        if (isVessel) nearDiscVessels++;
                    }
                }
            }
        }

        const globalDensity = globalVessels / Math.max(globalTotal, 1);
        const nearDiscDensity = nearDiscVessels / Math.max(nearDiscTotal, 1);

        const nvdSuspected = nearDiscDensity > globalDensity * 2.5 && nearDiscDensity > 0.15;
        const nveSuspected = globalDensity > 0.25; // Abnormally high overall density

        return {
            nvdSuspected,
            nveSuspected,
            confidence: Math.min(0.5, (nvdSuspected ? 0.15 : 0) + (nveSuspected ? 0.15 : 0))
        };
    }

    // ============================================================
    // STAGE 3: DR Severity Grading
    // Ports: gradeDRSeverity.m (rule-based only)
    // ============================================================
    gradeDRSeverity(lesionData) {
        const classNames = ['0_NoDR', '1_Mild', '2_Moderate', '3_Severe', '4_Proliferative'];
        const gradeLabels = ['No DR', 'Mild NPDR', 'Moderate NPDR', 'Severe NPDR', 'Proliferative DR'];

        const maCount = lesionData.microaneurysms.length;
        const dotCount = lesionData.hemorrhages.dotCount;
        const blotCount = lesionData.hemorrhages.blotCount;
        const exudateFrac = lesionData.exudateStats.fractionOfFOV;
        const nvSuspected = lesionData.nvInfo.nvdSuspected || lesionData.nvInfo.nveSuspected;

        const ruleScore = { maCount, dotCount, blotCount, exudateFrac, nvSuspected };

        let ruleGrade;
        if (nvSuspected && lesionData.nvInfo.confidence > 0.3) {
            ruleGrade = 4;
        } else if (blotCount >= 5 || (blotCount >= 2 && exudateFrac > 0.01)) {
            ruleGrade = 3;
        } else if (maCount >= 5 || dotCount >= 1 || exudateFrac > 0.002) {
            ruleGrade = 2;
        } else if (maCount >= 1) {
            ruleGrade = 1;
        } else {
            ruleGrade = 0;
        }

        // Confidence
        let confidence = 0.6 + 0.08 * Math.min(maCount, 5) - 0.05 * (ruleGrade > 0 && maCount <= 2 ? 1 : 0);
        confidence = Math.max(0.3, Math.min(0.95, confidence));

        return {
            icdrGrade: ruleGrade,
            gradeLabel: gradeLabels[ruleGrade],
            gradeCode: classNames[ruleGrade],
            referable: ruleGrade >= 2,
            confidence,
            ruleScore,
            cnnProbs: null,
            fusionWeight: 0,
            needsManualReview: confidence < 0.65
        };
    }

    // ============================================================
    // STAGE 4: Explainability — lesion-density heatmap
    // Ports: generateGradCAM.m (lesion-density fallback)
    // ============================================================
    generateHeatmap(lesionData) {
        const h = this.height, w = this.width;
        const heat = Array.from({ length: h }, () => new Float64Array(w));

        // Add Gaussian blobs for microaneurysms
        for (const ma of lesionData.microaneurysms) {
            this._addGaussBlob(heat, ma.position, Math.max(ma.radius * 3, 6), ma.strength);
        }

        // Add blobs for hemorrhages
        for (const hem of lesionData.hemorrhages.lesions) {
            this._addGaussBlob(heat, hem.position, Math.sqrt(hem.area) * 2, hem.severity);
        }

        // Smooth
        const smoothed = this._gaussBlur(heat, 4);

        // Normalize
        let maxH = 0;
        for (const row of smoothed) for (const v of row) maxH = Math.max(maxH, v);
        if (maxH > 0) for (const row of smoothed) for (let i = 0; i < w; i++) row[i] /= maxH;

        return smoothed;
    }

    _addGaussBlob(heat, center, sigma, weight) {
        const h = heat.length, w = heat[0].length;
        const r = Math.ceil(sigma * 3);
        const x1 = Math.max(0, Math.round(center[0] - r));
        const x2 = Math.min(w - 1, Math.round(center[0] + r));
        const y1 = Math.max(0, Math.round(center[1] - r));
        const y2 = Math.min(h - 1, Math.round(center[1] + r));

        for (let y = y1; y <= y2; y++) {
            for (let x = x1; x <= x2; x++) {
                const d2 = (x - center[0]) ** 2 + (y - center[1]) ** 2;
                heat[y][x] += weight * Math.exp(-d2 / (2 * sigma * sigma));
            }
        }
    }

    // ============================================================
    // RENDERING helpers — draw results onto canvas elements
    // ============================================================

    /** Render the original (possibly enhanced) image to a canvas */
    renderOriginal(canvas) {
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(this.imageData, 0, 0);
    }

    /** Render vessel mask as a colored overlay */
    renderVessels(canvas, vesselMask) {
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(this.imageData, 0, 0);

        const overlay = ctx.getImageData(0, 0, this.width, this.height);
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                if (vesselMask[y] && vesselMask[y][x]) {
                    const i = (y * this.width + x) * 4;
                    overlay.data[i] = Math.min(255, overlay.data[i] + 60);     // R
                    overlay.data[i + 1] = Math.min(255, overlay.data[i + 1] + 200); // G
                    overlay.data[i + 2] = Math.min(255, overlay.data[i + 2] + 100); // B
                }
            }
        }
        ctx.putImageData(overlay, 0, 0);
    }

    /** Render lesion detection overlay */
    renderLesions(canvas, lesionData, discInfo) {
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(this.imageData, 0, 0);

        // Draw optic disc circle
        if (discInfo) {
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(discInfo.center[0], discInfo.center[1], discInfo.radius, 0, 2 * Math.PI);
            ctx.stroke();
            ctx.font = '10px Inter';
            ctx.fillStyle = '#3b82f6';
            ctx.fillText('Optic Disc', discInfo.center[0] - 25, discInfo.center[1] - discInfo.radius - 5);
        }

        // Draw microaneurysms
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 1.5;
        for (const ma of lesionData.microaneurysms) {
            ctx.beginPath();
            ctx.arc(ma.position[0], ma.position[1], Math.max(3, ma.radius * 2), 0, 2 * Math.PI);
            ctx.stroke();
        }

        // Draw hemorrhages
        for (const hem of lesionData.hemorrhages.lesions) {
            ctx.strokeStyle = hem.type === 'dot' ? '#f97316' : '#dc2626';
            ctx.lineWidth = 1.5;
            const r = Math.sqrt(hem.area / Math.PI) * 1.5;
            ctx.beginPath();
            ctx.arc(hem.position[0], hem.position[1], Math.max(4, r), 0, 2 * Math.PI);
            ctx.stroke();
        }

        // Legend
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(4, this.height - 50, 160, 46);
        ctx.font = '9px Inter';
        ctx.fillStyle = '#3b82f6'; ctx.fillText('● Optic Disc', 10, this.height - 36);
        ctx.fillStyle = '#ef4444'; ctx.fillText('● Microaneurysms', 10, this.height - 24);
        ctx.fillStyle = '#f97316'; ctx.fillText('● Dot Hemorrhages', 10, this.height - 12);
    }

    /** Render heatmap overlay (jet colormap) */
    renderHeatmap(canvas, heatmap) {
        canvas.width = this.width;
        canvas.height = this.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(this.imageData, 0, 0);

        const overlay = ctx.getImageData(0, 0, this.width, this.height);
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                const v = heatmap[y][x];
                const alpha = v * 0.55;
                const [r, g, b] = this._jetColor(v);
                const i = (y * this.width + x) * 4;
                overlay.data[i] = Math.round(overlay.data[i] * (1 - alpha) + r * alpha);
                overlay.data[i + 1] = Math.round(overlay.data[i + 1] * (1 - alpha) + g * alpha);
                overlay.data[i + 2] = Math.round(overlay.data[i + 2] * (1 - alpha) + b * alpha);
            }
        }
        ctx.putImageData(overlay, 0, 0);
    }

    /** Jet colormap approximation */
    _jetColor(v) {
        v = Math.max(0, Math.min(1, v));
        let r, g, b;
        if (v < 0.25) {
            r = 0; g = 4 * v; b = 1;
        } else if (v < 0.5) {
            r = 0; g = 1; b = 1 - 4 * (v - 0.25);
        } else if (v < 0.75) {
            r = 4 * (v - 0.5); g = 1; b = 0;
        } else {
            r = 1; g = 1 - 4 * (v - 0.75); b = 0;
        }
        return [r * 255, g * 255, b * 255];
    }

    // ============================================================
    // STAGE 5: Clinical Report (text-based)
    // ============================================================
    generateReport(qc, grading, lesionData) {
        const now = new Date().toLocaleString();
        const lines = [
            '╔══════════════════════════════════════════════════════════════╗',
            '║     DIABETIC RETINOPATHY SCREENING REPORT                  ║',
            '║     For Ophthalmologist Review — Decision Support Only      ║',
            '╚══════════════════════════════════════════════════════════════╝',
            '',
            `  Generated: ${now}`,
            '',
            '── GRADING RESULT ─────────────────────────────────────────────',
            `  ICDR Grade:     ${grading.icdrGrade} — ${grading.gradeLabel}`,
            `  Referable:      ${grading.referable ? 'YES — refer to ophthalmologist' : 'No'}`,
            `  Confidence:     ${(grading.confidence * 100).toFixed(0)}%`,
            `  Manual Review:  ${grading.needsManualReview ? 'FLAGGED — priority review needed' : 'Standard pathway'}`,
            `  Grading Mode:   Rule-based only (no trained CNN loaded)`,
            '',
            '── LESION EVIDENCE ────────────────────────────────────────────',
            `  Microaneurysms:          ${grading.ruleScore.maCount}`,
            `  Dot hemorrhages:         ${grading.ruleScore.dotCount}`,
            `  Blot hemorrhages:        ${grading.ruleScore.blotCount}`,
            `  Exudate area (% of FOV): ${(grading.ruleScore.exudateFrac * 100).toFixed(3)}%`,
            `  Neovascularization flag: ${grading.ruleScore.nvSuspected ? 'SUSPECTED — verify manually' : 'Not detected'}`,
            '',
            '── IMAGE QUALITY ──────────────────────────────────────────────',
            `  Overall score:    ${qc.score.toFixed(2)}`,
            `  Sharpness:        ${qc.sharpness.toFixed(2)}`,
            `  Illumination:     ${qc.illumination.toFixed(2)}`,
            `  Field of View:    ${qc.fov.toFixed(2)}`,
            `  Contrast:         ${qc.contrast.toFixed(2)}`,
            `  Enhancement:      ${qc.needsEnhancement ? 'Applied' : 'Not needed'}`,
        ];

        if (qc.reasons.length > 0) {
            lines.push(`  Quality notes:    ${qc.reasons.join('; ')}`);
        }

        lines.push('');
        lines.push('── DISCLAIMER ─────────────────────────────────────────────────');
        lines.push('  This is a decision-support / triage tool for a human-in-the-');
        lines.push('  loop workflow, NOT an autonomous diagnostic device. Any real');
        lines.push('  deployment needs regulatory clearance and clinical validation.');
        lines.push('════════════════════════════════════════════════════════════════');

        return lines.join('\n');
    }

    // ============================================================
    // FULL PIPELINE — runs all 5 stages
    // ============================================================
    async runPipeline(onProgress) {
        const report = (step, msg) => onProgress && onProgress(step, msg);

        // Stage 1
        report(1, 'Assessing image quality...');
        await this._sleep(100);
        const qc = this.assessImageQuality();
        report(1, `Score: ${qc.score.toFixed(2)}`);

        if (!qc.isGradable) {
            return { status: 'rejected', qc, message: qc.recaptureMsg };
        }

        // Stage 2
        report(2, 'Detecting optic disc & fovea...');
        await this._sleep(50);
        const structures = this.detectOpticDiscFovea();

        report(2, 'Segmenting vessels...');
        await this._sleep(50);
        const vessels = this.segmentVessels();

        report(2, 'Detecting microaneurysms...');
        await this._sleep(50);
        const maList = this.detectMicroaneurysms(structures.disc, vessels.mask);

        report(2, 'Segmenting exudates...');
        await this._sleep(50);
        const exudateStats = this.segmentExudates(structures.disc);

        report(2, 'Detecting hemorrhages...');
        await this._sleep(50);
        const hemStats = this.detectHemorrhages(structures.disc, vessels.mask);

        report(2, 'Checking for neovascularization...');
        await this._sleep(50);
        const nvInfo = this.detectNeovascularization(vessels.mask, structures.disc);

        report(2, `MAs: ${maList.length} | Hem: ${hemStats.totalCount} | Exudate: ${(exudateStats.fractionOfFOV * 100).toFixed(2)}%`);

        const lesionData = {
            microaneurysms: maList,
            hemorrhages: hemStats,
            exudateStats,
            nvInfo,
            vesselDensity: vessels.vesselDensity
        };

        // Stage 3
        report(3, 'Grading DR severity...');
        await this._sleep(100);
        const grading = this.gradeDRSeverity(lesionData);
        report(3, `Grade ${grading.icdrGrade} — ${grading.gradeLabel}`);

        // Stage 4
        report(4, 'Generating attention heatmap...');
        await this._sleep(100);
        const heatmap = this.generateHeatmap(lesionData);
        report(4, 'Heatmap generated');

        // Stage 5
        report(5, 'Generating clinical report...');
        await this._sleep(100);
        const reportText = this.generateReport(qc, grading, lesionData);
        report(5, 'Report ready');

        this.result = {
            status: 'graded',
            qc,
            structures,
            vessels,
            lesionData,
            grading,
            heatmap,
            reportText
        };

        return this.result;
    }

    _sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Export for use in app.js
window.DRPipeline = DRPipeline;
