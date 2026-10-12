function [discInfo, foveaInfo] = detectOpticDiscFovea(img)
%DETECTOPTICDISCFOVEA  Localize optic disc and fovea in a fundus image.
%
%   [discInfo, foveaInfo] = DETECTOPTICDISCFOVEA(img)
%
%   discInfo:  struct with .center [x y], .radius, .mask, .confidence
%   foveaInfo: struct with .center [x y], .confidence
%
%   Method:
%     Optic disc  - the disc is the brightest large, roughly circular,
%                   high-contrast region on the red/green channels. We
%                   combine a percentile brightness mask with Hough-circle
%                   search constrained to a plausible radius range.
%     Fovea       - located in the darkest macular region within a search
%                   window at ~2.5 disc-diameters temporal to the disc
%                   (nasal/temporal side inferred from which half of the
%                   image the disc falls in, a standard fundus convention).

    if size(img,3) == 3
        green = im2double(img(:,:,2));
        red   = im2double(img(:,:,1));
    else
        green = im2double(img);
        red = green;
    end
    [h, w] = size(green);

    % ---------- Optic disc ----------
    brightChannel = 0.75*red + 0.25*green; % disc is brightest in red/green
    smoothed = imgaussfilt(brightChannel, 5);

    % Candidate radius range scaled to image size (disc ~= 1/8 to 1/5 of
    % typical 45-degree FOV fundus image width)
    rMin = round(w * 0.03);
    rMax = round(w * 0.10);

    try
        [centers, radii, metric] = imfindcircles(smoothed, [rMin rMax], ...
            'ObjectPolarity', 'bright', 'Sensitivity', 0.94, 'EdgeThreshold', 0.1);
    catch
        centers = []; radii = []; metric = [];
    end

    if ~isempty(centers)
        [~, idx] = max(metric);
        discInfo.center = centers(idx, :);
        discInfo.radius = radii(idx);
        discInfo.confidence = min(1, metric(idx) / 0.2);
    else
        % Fallback: brightest-region centroid via percentile thresholding
        thresh = prctile(smoothed(:), 97);
        bw = smoothed >= thresh;
        bw = bwareafilt(bw, 1);
        stats = regionprops(bw, 'Centroid', 'EquivDiameter');
        if ~isempty(stats)
            discInfo.center = stats(1).Centroid;
            discInfo.radius = stats(1).EquivDiameter / 2;
            discInfo.confidence = 0.4; % low-confidence fallback
        else
            discInfo.center = [w/2, h/2];
            discInfo.radius = round(w*0.06);
            discInfo.confidence = 0.1;
        end
    end
    [xg, yg] = meshgrid(1:w, 1:h);
    discInfo.mask = ((xg - discInfo.center(1)).^2 + (yg - discInfo.center(2)).^2) ...
                     <= discInfo.radius^2;

    % ---------- Fovea ----------
    % Search window: horizontally offset from disc by ~2-3 disc diameters,
    % on the side away from the image edge nearest the disc (temporal side),
    % vertically centered near the disc's vertical position.
    dd = discInfo.radius * 2;
    if discInfo.center(1) > w/2
        searchCx = discInfo.center(1) - 2.5*dd; % disc on right -> fovea to the left
    else
        searchCx = discInfo.center(1) + 2.5*dd;
    end
    searchCy = discInfo.center(2);
    winR = dd * 1.5;

    xr = max(1, round(searchCx - winR)) : min(w, round(searchCx + winR));
    yr = max(1, round(searchCy - winR)) : min(h, round(searchCy + winR));
    if isempty(xr) || isempty(yr)
        foveaInfo.center = [searchCx, searchCy];
        foveaInfo.confidence = 0.1;
        return;
    end
    window = green(yr, xr);
    windowSmooth = imgaussfilt(window, 7);
    [minVal, linIdx] = min(windowSmooth(:));
    [ry, rx] = ind2sub(size(windowSmooth), linIdx);
    foveaInfo.center = [xr(1) + rx - 1, yr(1) + ry - 1];
    % confidence: how much darker than local surround (contrast-based)
    localContrast = mean(windowSmooth(:)) - minVal;
    foveaInfo.confidence = min(1, localContrast / 0.15);
end
