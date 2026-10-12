function [vesselMask, vesselDensity] = segmentVessels(img, fovMask)
%SEGMENTVESSELS  Multi-scale Frangi-vesselness retinal vessel segmentation.
%
%   [vesselMask, vesselDensity] = SEGMENTVESSELS(img, fovMask)
%
%   vesselMask    - binary mask of segmented vasculature
%   vesselDensity - fraction of the field-of-view occupied by vessels
%                   (useful downstream as a vascular-abnormality feature
%                   for neovascularization screening)
%
%   Method: green channel -> background-subtracted contrast normalization
%   -> multi-scale Hessian-based Frangi vesselness (implemented directly,
%   no extra toolbox dependency beyond Image Processing Toolbox) -> hysteresis
%   thresholding -> small-object removal.

    if size(img,3) == 3
        green = im2double(img(:,:,2));
    else
        green = im2double(img);
    end
    if nargin < 2 || isempty(fovMask)
        fovMask = true(size(green));
    end

    % Contrast-limited adaptive normalization before vesselness (vessels are
    % low-contrast against background especially near the periphery)
    green = adapthisteq(green, 'ClipLimit', 0.008);
    green = imcomplement(green); % vessels become bright ridges

    scales = [1, 1.5, 2, 2.5, 3, 3.5, 4]; % pixel sigma range; tune to camera resolution
    vesselness = zeros(size(green));

    for s = scales
        [Dxx, Dxy, Dyy] = hessian2D(green, s);
        % Hessian eigenvalues (2x2 closed form)
        tmp = sqrt((Dxx - Dyy).^2 + 4*Dxy.^2);
        lambda1 = 0.5*(Dxx + Dyy + tmp);
        lambda2 = 0.5*(Dxx + Dyy - tmp);
        % order by magnitude, |lambda1| <= |lambda2|
        swap = abs(lambda1) > abs(lambda2);
        tmp1 = lambda1; tmp1(swap) = lambda2(swap);
        tmp2 = lambda2; tmp2(swap) = lambda1(swap);
        lambda1 = tmp1; lambda2 = tmp2;

        Rb = (lambda1 ./ (lambda2 + eps)).^2;
        S2 = lambda1.^2 + lambda2.^2;
        beta = 0.5; c = 0.5 * max(S2(:));
        Vo = exp(-Rb ./ (2*beta^2)) .* (1 - exp(-S2 ./ (2*c^2)));
        Vo(lambda2 > 0) = 0; % only bright ridges on dark background
        vesselness = max(vesselness, Vo * s); % scale-normalized max response
    end

    vesselness = mat2gray(vesselness) .* fovMask;
    t1 = graythresh(vesselness(fovMask)) * 0.6; % low hysteresis bound
    t2 = graythresh(vesselness(fovMask)) * 1.1; % high hysteresis bound
    strongMask = vesselness > t2;
    if any(strongMask(:))
        vesselMask = imreconstruct(strongMask, weakMask);
    else
        vesselMask = false(size(weakMask));
    end
    vesselMask = bwareaopen(vesselMask, 15); % drop speckle noise
    vesselMask = vesselMask & fovMask;

    vesselDensity = nnz(vesselMask) / max(nnz(fovMask), 1);
end

function [Dxx, Dxy, Dyy] = hessian2D(img, sigma)
%HESSIAN2D  Gaussian-derivative Hessian components at scale sigma.
    hsize = 2*ceil(3*sigma) + 1;
    [x, y] = meshgrid(-floor(hsize/2):floor(hsize/2));
    g   = exp(-(x.^2 + y.^2) / (2*sigma^2));
    Gxx = ((x.^2 - sigma^2) ./ sigma^4) .* g;
    Gyy = ((y.^2 - sigma^2) ./ sigma^4) .* g;
    Gxy = ((x.*y) ./ sigma^4) .* g;
    Dxx = imfilter(img, Gxx, 'replicate') * sigma^2;
    Dyy = imfilter(img, Gyy, 'replicate') * sigma^2;
    Dxy = imfilter(img, Gxy, 'replicate') * sigma^2;
end
