function outImg = enhanceImage(img, qc)
%ENHANCEIMAGE  Adaptive enhancement for borderline-quality fundus images.
%
%   outImg = ENHANCEIMAGE(img, qc) applies, conditionally on the quality
%   report qc from ASSESSIMAGEQUALITY:
%       - illumination normalization (background estimation + division)
%       - CLAHE contrast enhancement on luminance channel
%       - edge-preserving denoising
%   and returns an RGB image in the original class/range.
%
%   If qc is omitted, all steps are applied at default strength.

    if nargin < 2
        qc = struct('illumination', 0.5, 'contrast', 0.5, 'sharpness', 0.7);
    end

    wasUint8 = isa(img, 'uint8');
    imgD = im2double(img);

    % Work in Lab color space so we enhance luminance without distorting
    % the red/green channel relationships lesion detectors rely on.
    labImg = rgb2lab(imgD);
    L = labImg(:,:,1) / 100; % normalize to [0,1]

    % --- 1. Illumination normalization ---
    % Estimate slowly-varying background illumination with a large
    % Gaussian blur, then flatten it out (common fundus preprocessing trick).
    if qc.illumination < 0.75
        sigma = round(min(size(L,1), size(L,2)) / 12);
        background = imgaussfilt(L, max(sigma,10));
        L = L - background + mean(background(:));
        L = mat2gray(L);
    end

    % --- 2. CLAHE ---
    claheStrength = 1 - qc.contrast; % worse contrast -> stronger CLAHE
    clipLimit = 0.01 + 0.03 * claheStrength; % keep it gentle to avoid noise amplification
    L = adapthisteq(L, 'ClipLimit', clipLimit, 'Distribution', 'rayleigh');

    % --- 3. Denoising (only if focus score suggests noise, not real blur) ---
    if isfield(qc, 'sharpness') && qc.sharpness < 0.6
        L = imbilatfilt(L, 0.01, 3); % edge-preserving, protects vessel edges
    end

    labImg(:,:,1) = L * 100;
    outImgD = lab2rgb(labImg);
    outImgD = min(max(outImgD, 0), 1);

    if wasUint8
        outImg = im2uint8(outImgD);
    else
        outImg = outImgD;
    end
end
