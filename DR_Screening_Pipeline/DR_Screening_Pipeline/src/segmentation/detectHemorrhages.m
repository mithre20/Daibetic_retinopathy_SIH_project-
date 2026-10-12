function [hemMask, hemStats] = detectHemorrhages(img, discInfo, vesselMask)
%DETECTHEMORRHAGES  Dark lesion (hemorrhage) detection and dot/blot typing.
%
%   [hemMask, hemStats] = DETECTHEMORRHAGES(img, discInfo, vesselMask)
%
%   hemStats: struct array per detected lesion with
%       .area, .type ('dot'|'blot'), .position, .severity
%   plus a summary field hemStats(1).summary = struct('dotCount',.,'blotCount',.)
%
%   Method: dark-lesion candidates via inverted top-hat (same family of
%   operator as MA detection but with larger structuring elements and no
%   sub-pixel step, since hemorrhages are larger, irregular blobs). Dot
%   hemorrhages (small, round) vs. blot hemorrhages (larger, irregular,
%   indicating deeper/more severe intraretinal bleeding, an ICDR severity
%   marker) are separated by area + eccentricity + solidity.

    if size(img,3) == 3
        green = im2double(img(:,:,2));
    else
        green = im2double(img);
    end
    if nargin < 3 || isempty(vesselMask)
        vesselMask = false(size(green));
    end

    green = adapthisteq(green, 'ClipLimit', 0.01);
    inv = imcomplement(green);

    se = strel('disk', 6);
    marker = imerode(inv, se);
    opened = imreconstruct(marker, inv);
    residual = inv - opened;
    residual = mat2gray(residual);

    thresh = graythresh(residual) * 1.3;
    candMask = residual > max(thresh, 0.12);
    candMask = candMask & ~imdilate(vesselMask, strel('disk', 3));
    if nargin >= 2 && ~isempty(discInfo) && isfield(discInfo, 'mask')
        candMask = candMask & ~imdilate(discInfo.mask, strel('disk', round(discInfo.radius*0.3)));
    end
    candMask = bwareaopen(candMask, 4);

    cc = bwconncomp(candMask);
    props = regionprops(cc, 'Area', 'Eccentricity', 'Solidity', 'Centroid', 'PixelIdxList');

    hemMask = false(size(candMask));
    hemStats = struct('area', {}, 'type', {}, 'position', {}, 'severity', {});
    dotCount = 0; blotCount = 0;

    for k = 1:numel(props)
        p = props(k);
        if p.Area < 5 || p.Area > round(numel(green)*0.01)
            continue; % too small (noise) or too large (likely artifact)
        end
        hemMask(p.PixelIdxList) = true;

        isDot = p.Area <= 30 && p.Eccentricity < 0.8 && p.Solidity > 0.85;
        if isDot
            type = 'dot';
            dotCount = dotCount + 1;
            severity = 0.3; % dot hemorrhages: mild-moderate NPDR marker
        else
            type = 'blot';
            blotCount = blotCount + 1;
            severity = 0.7; % blot hemorrhages: more severe NPDR marker
        end
        hemStats(end+1) = struct('area', p.Area, 'type', type, ...
                                  'position', p.Centroid, 'severity', severity); %#ok<AGROW>
    end

    hemStats = struct('lesions', {hemStats}, ...
                       'dotCount', dotCount, ...
                       'blotCount', blotCount, ...
                       'totalCount', dotCount + blotCount);
end
