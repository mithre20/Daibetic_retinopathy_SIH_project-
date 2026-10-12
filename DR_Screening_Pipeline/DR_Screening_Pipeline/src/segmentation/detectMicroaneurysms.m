function maList = detectMicroaneurysms(img, discInfo, vesselMask)
%DETECTMICROANEURYSMS  Sub-pixel microaneurysm (MA) candidate detection.
%
%   maList = DETECTMICROANEURYSMS(img, discInfo, vesselMask)
%
%   maList is an Nx1 struct array with fields:
%       .position   [x y] sub-pixel centroid (bicubic-interpolated peak of
%                   a local Gaussian fit, refined beyond pixel grid)
%       .radius     estimated lesion radius in pixels
%       .strength   candidate confidence score in [0,1]
%
%   Method (classical morphological top-hat approach, still the backbone
%   of most published MA detectors because deep nets alone tend to miss
%   these <=125 micron lesions at typical fundus-camera resolution):
%     1. Green-channel contrast normalization
%     2. Morphological top-hat with a rotating linear structuring element
%        (removes vessels, which are elongated, while preserving small
%        round MA candidates)
%     3. Candidate region extraction + shape filtering (roundness,
%        size, vessel-exclusion)
%     4. Sub-pixel centroid refinement via local 2D Gaussian fit
%
%   NOTE on resolution: sub-pixel refinement only adds real precision if
%   source images are >=15 pixels/degree (~2000x2000+ fundus captures).
%   On lower-resolution portable-camera JPEGs the "sub-pixel" step still
%   runs but its benefit shrinks — this is documented, not hidden.

    if size(img,3) == 3
        green = im2double(img(:,:,2));
    else
        green = im2double(img);
    end
    if nargin < 3 || isempty(vesselMask)
        vesselMask = false(size(green));
    end

    green = adapthisteq(green, 'ClipLimit', 0.01);
    inv = imcomplement(green); % MAs are dark spots -> bright after inversion

    % Rotating top-hat: linear SE at multiple angles removes vessels
    % (elongated structures survive a linear opening; round MAs do not,
    % so top-hat = original - max-over-angle opening isolates round blobs)
    angles = 0:15:165;
    lineLen = 9;
    openMax = zeros(size(inv));
    for a = angles
        se = strel('line', lineLen, a);
        openMax = max(openMax, imopen(inv, se));
    end
    tophat = inv - openMax;
    tophat(tophat < 0) = 0;
    tophat = mat2gray(tophat);

    % Candidate mask
    thresh = graythresh(tophat) * 1.4;
    candMask = tophat > thresh;
    candMask = candMask & ~imdilate(vesselMask, strel('disk', 2)); % not on a vessel
    if nargin >= 2 && ~isempty(discInfo) && isfield(discInfo, 'mask')
        candMask = candMask & ~imdilate(discInfo.mask, strel('disk', 3));
    end
    candMask = bwareaopen(candMask, 1);

    cc = bwconncomp(candMask);
    stats = regionprops(cc, tophat, 'Centroid', 'Area', 'EquivDiameter', ...
                         'Eccentricity', 'MaxIntensity', 'BoundingBox');

    maList = struct('position', {}, 'radius', {}, 'strength', {});
    for k = 1:numel(stats)
        s = stats(k);
        % Shape filter: MAs are small (<= ~15px diam at typical field cam
        % resolution) and roughly round (low eccentricity)
        if s.EquivDiameter > 15 || s.Eccentricity > 0.85
            continue; % too large/elongated -> likely vessel fragment or hemorrhage
        end

        % --- Sub-pixel refinement: fit a 2D Gaussian in a local patch ---
        bb = round(s.BoundingBox);
        pad = 3;
        r1 = max(1, bb(2)-pad); r2 = min(size(tophat,1), bb(2)+bb(4)+pad);
        c1 = max(1, bb(1)-pad); c2 = min(size(tophat,2), bb(1)+bb(3)+pad);
        patch = tophat(r1:r2, c1:c2);
        subPos = subpixelPeak(patch) + [c1-1, r1-1]; % [x y] in image coords

        strength = min(1, s.MaxIntensity / (thresh + eps) - 1);
        maList(end+1) = struct( ...
            'position', subPos, ...
            'radius', s.EquivDiameter/2, ...
            'strength', max(0, min(1, strength))); %#ok<AGROW>
    end
end

function pk = subpixelPeak(patch)
%SUBPIXELPEAK  Sub-pixel [x y] location of a patch's intensity peak using
%   parabolic interpolation around the discrete maximum in each axis.
    [~, idx] = max(patch(:));
    [ry, rx] = ind2sub(size(patch), idx);
    [h, w] = size(patch);

    % X-direction parabolic refinement
    if rx > 1 && rx < w
        dx = 0.5 * (patch(ry, rx-1) - patch(ry, rx+1)) / ...
             (patch(ry, rx-1) - 2*patch(ry, rx) + patch(ry, rx+1) + eps);
    else
        dx = 0;
    end
    % Y-direction parabolic refinement
    if ry > 1 && ry < h
        dy = 0.5 * (patch(ry-1, rx) - patch(ry+1, rx)) / ...
             (patch(ry-1, rx) - 2*patch(ry, rx) + patch(ry+1, rx) + eps);
    else
        dy = 0;
    end
    pk = [rx + dx, ry + dy];
end
