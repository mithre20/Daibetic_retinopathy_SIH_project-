function qc = assessImageQuality(img)
%ASSESSIMAGEQUALITY  Grade a fundus photo for gradability before any DR analysis.
%
%   qc = ASSESSIMAGEQUALITY(img) returns a struct:
%       qc.score        overall quality score in [0,1] (1 = excellent)
%       qc.isGradable   logical, true if score >= threshold and no hard fail
%       qc.sharpness    normalized focus score [0,1]
%       qc.illumination normalized illumination-uniformity score [0,1]
%       qc.fov          normalized field-of-view coverage score [0,1]
%       qc.contrast     normalized contrast score [0,1]
%       qc.reason       cellstr of human-readable reasons if rejected
%       qc.recaptureMsg feedback string to show the field operator
%
%   Designed for portable non-mydriatic camera output (JPEG, variable
%   compression, frequent vignetting / partial eyelid occlusion).

    if size(img,3) == 3
        gray = rgb2gray(img);
        green = img(:,:,2); % green channel has best vessel/lesion contrast
    else
        gray = img;
        green = img;
    end
    gray = im2double(gray);
    green = im2double(green);

    reasons = {};

    % --- 1. Field of view: does the retina fill enough of the frame? ---
    % Retinal circle is bright/mid-tone; background outside the camera
    % aperture is near-black. Otsu threshold + largest connected component.
    bw = imbinarize(gray, graythresh(gray) * 0.4); % lenient: catch dim edges too
    bw = imfill(bw, 'holes');
    cc = bwconncomp(bw);
    if cc.NumObjects == 0
        fovFrac = 0;
    else
        areas = cellfun(@numel, cc.PixelIdxList);
        fovFrac = max(areas) / numel(gray);
    end
    fovScore = min(1, fovFrac / 0.60); % expect retina to cover >=60% of frame
    if fovScore < 0.5
        reasons{end+1} = 'Retinal field of view too small / eyelid occlusion suspected';
    end

    % --- 2. Sharpness: variance of Laplacian on the retinal region only ---
    lap = fspecial('laplacian', 0.2);
    lapResp = imfilter(green, lap, 'replicate');
    mask = bw;
    if nnz(mask) < 100
        mask = true(size(green));
    end
    sharpnessRaw = var(lapResp(mask));
    sharpnessScore = min(1, sharpnessRaw / 0.02); % empirical calibration constant
    if sharpnessScore < 0.35
        reasons{end+1} = 'Image out of focus / motion blur';
    end

    % --- 3. Illumination uniformity: divide frame into quadrant/grid tiles,
    %     compare mean brightness across tiles; large spread = uneven flash,
    %     glare, or dark vignetting corners ---
    [h, w] = size(green);
    gridN = 4;
    tileMeans = zeros(gridN, gridN);
    rh = floor(h / gridN); rw = floor(w / gridN);
    for r = 1:gridN
        for c = 1:gridN
            tile = green((r-1)*rh+1 : r*rh, (c-1)*rw+1 : c*rw);
            tileMeans(r,c) = mean(tile(:));
        end
    end
    illumSpread = std(tileMeans(:)) / (mean(tileMeans(:)) + eps);
    illumScore = max(0, 1 - illumSpread / 0.5);
    % Global over/under exposure
    meanBrightness = mean(green(mask));
    if meanBrightness < 0.15 || meanBrightness > 0.9
        illumScore = illumScore * 0.4;
        reasons{end+1} = 'Under- or over-exposed image';
    end
    if illumScore < 0.4
        reasons{end+1} = 'Uneven illumination / glare / vignetting';
    end

    % --- 4. Contrast: local RMS contrast on retinal region ---
    contrastRaw = std(green(mask));
    contrastScore = min(1, contrastRaw / 0.18);
    if contrastScore < 0.3
        reasons{end+1} = 'Low contrast image';
    end

    % --- Weighted overall score ---
    weights = struct('fov', 0.30, 'sharp', 0.35, 'illum', 0.20, 'contrast', 0.15);
    overall = weights.fov*fovScore + weights.sharp*sharpnessScore + ...
              weights.illum*illumScore + weights.contrast*contrastScore;

    qc.score        = overall;
    qc.sharpness    = sharpnessScore;
    qc.illumination = illumScore;
    qc.fov          = fovScore;
    qc.contrast     = contrastScore;
    qc.isGradable   = (overall >= 0.55) && isempty(reasons);
    qc.reason       = reasons;

    if qc.isGradable
        qc.recaptureMsg = '';
    elseif overall >= 0.35
        qc.recaptureMsg = ['Borderline quality - will attempt automatic enhancement. Issues: ' ...
                            strjoin(reasons, '; ')];
        qc.isGradable = true; % allow the enhancement stage to try to rescue it
        qc.needsEnhancement = true;
    else
        qc.recaptureMsg = ['REJECTED - please recapture. Issues: ' strjoin(reasons, '; ')];
        qc.needsEnhancement = false;
    end
    if ~isfield(qc, 'needsEnhancement')
        qc.needsEnhancement = (overall < 0.75); % still enhance mildly good images
    end
end
