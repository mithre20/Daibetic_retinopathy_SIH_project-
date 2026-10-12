function [exudateMask, exudateStats] = segmentExudates(img, discInfo)
%SEGMENTEXUDATES  Hard exudate segmentation (bright yellowish-white lesions).
%
%   [exudateMask, exudateStats] = SEGMENTEXUDATES(img, discInfo)
%
%   exudateStats: struct with .count, .totalArea, .fractionOfFOV
%
%   Method: exudates are bright, high-contrast lesions distinct from the
%   optic disc (which must be excluded) and from vessel specular
%   reflections (filtered by shape/size). Uses adaptive thresholding on
%   a green-channel local-contrast-enhanced image plus morphological
%   reconstruction to capture irregular exudate boundaries.

    if size(img,3) == 3
        green = im2double(img(:,:,2));
    else
        green = im2double(img);
    end
    fovMask = green > 0.02;
    if nargin >= 2 && ~isempty(discInfo) && isfield(discInfo, 'mask')
        excludeMask = imdilate(discInfo.mask, strel('disk', round(discInfo.radius*0.3)));
    else
        excludeMask = false(size(green));
    end

    enhanced = adapthisteq(green, 'ClipLimit', 0.015);

    % Morphological reconstruction to flatten background, leaving bright
    % lesions standing proud of their local surround
    se = strel('disk', 8);
    marker = imerode(enhanced, se);
    opened = imreconstruct(marker, enhanced);
    residual = enhanced - opened; % bright lesions pop out here
    residual = mat2gray(residual);

    thresh = graythresh(residual) * 1.5;
    candMask = residual > max(thresh, 0.15);
    candMask = candMask & fovMask & ~excludeMask;
    candMask = imopen(candMask, strel('disk', 1));
    candMask = bwareaopen(candMask, 3);

    % Reject long thin candidates (likely vessel specular highlights)
    cc = bwconncomp(candMask);
    stats = regionprops(cc, 'Area', 'Eccentricity', 'PixelIdxList');
    keepMask = false(size(candMask));
    for k = 1:numel(stats)
        if stats(k).Eccentricity < 0.93
            keepMask(stats(k).PixelIdxList) = true;
        end
    end
    exudateMask = keepMask;

    exudateStats.count = 0;
    if any(exudateMask(:))
        cc2 = bwconncomp(exudateMask);
        exudateStats.count = cc2.NumObjects;
    end
    exudateStats.totalArea = nnz(exudateMask);
    exudateStats.fractionOfFOV = nnz(exudateMask) / max(nnz(fovMask), 1);
end
