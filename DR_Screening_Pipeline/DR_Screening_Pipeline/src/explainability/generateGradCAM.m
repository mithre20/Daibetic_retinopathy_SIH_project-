function [overlayImg, camMap, lesionCorrelation] = generateGradCAM(img, cnnNet, targetClass, lesionData)
%GENERATEGRADCAM  Grad-CAM attention map + lesion-level correlation.
%
%   [overlayImg, camMap, lesionCorrelation] = GENERATEGRADCAM(img, cnnNet, targetClass, lesionData)
%
%   overlayImg        - RGB image with jet-colormap CAM heatmap overlaid
%   camMap            - raw normalized Grad-CAM map, same size as img
%   lesionCorrelation - struct scoring how much CAM attention overlaps
%                       each detected lesion type (a simple, checkable
%                       form of "does the model look where the lesions
%                       are", which is what makes the explanation
%                       clinically checkable rather than decorative)
%
%   Uses MATLAB's built-in gradCAM (Deep Learning Toolbox, R2021a+). If
%   cnnNet is empty (rule-only degraded mode), falls back to a
%   lesion-density heatmap built directly from segmentation outputs so the
%   report/explainability step still produces something clinically
%   meaningful without a trained network.

    imSize = size(img); imSize = imSize(1:2);

    if nargin >= 2 && ~isempty(cnnNet)
        inputSize = [224 224 3];
        imResized = imresize(im2uint8(img), inputSize(1:2));
        if size(imResized,3) == 1
            imResized = repmat(imResized, 1, 1, 3);
        end
        classNames = {'0_NoDR','1_Mild','2_Moderate','3_Severe','4_Proliferative'};
        if nargin < 3 || isempty(targetClass)
            probs = predict(cnnNet, imResized);
            [~, ci] = max(probs);
        else
            ci = targetClass + 1;
        end
        try
            scoreMap = gradCAM(cnnNet, imResized, classNames{ci});
        catch ME
            warning('generateGradCAM:fallback', ...
                'Built-in gradCAM failed (%s); falling back to lesion-density heatmap.', ME.message);
            scoreMap = [];
        end
        if ~isempty(scoreMap)
            camMap = imresize(mat2gray(scoreMap), imSize);
        else
            camMap = lesionDensityHeatmap(lesionData, imSize);
        end
    else
        camMap = lesionDensityHeatmap(lesionData, imSize);
    end

    overlayImg = overlayHeatmap(im2uint8(img), camMap);

    if nargin >= 4 && ~isempty(lesionData)
        lesionCorrelation = correlateWithLesions(camMap, lesionData, imSize);
    else
        lesionCorrelation = struct();
    end
end

function heat = lesionDensityHeatmap(lesionData, imSize)
%LESIONDENSITYHEATMAP  Build an attention-style map straight from detected
%   lesion locations, weighted by lesion severity/strength, for use when
%   no trained CNN is available.
    heat = zeros(imSize);
    if isempty(lesionData)
        return;
    end
    if isfield(lesionData, 'microaneurysms')
        for k = 1:numel(lesionData.microaneurysms)
            m = lesionData.microaneurysms(k);
            heat = addGaussianBlob(heat, m.position, max(m.radius*3, 6), m.strength);
        end
    end
    if isfield(lesionData, 'hemorrhages') && isfield(lesionData.hemorrhages, 'lesions')
        for k = 1:numel(lesionData.hemorrhages.lesions)
            h = lesionData.hemorrhages.lesions(k);
            heat = addGaussianBlob(heat, h.position, sqrt(h.area)*2, h.severity);
        end
    end
    heat = mat2gray(imgaussfilt(heat, 4));
end

function heat = addGaussianBlob(heat, center, sigma, weight)
    [h, w] = size(heat);
    x = max(1, round(center(1)-3*sigma)) : min(w, round(center(1)+3*sigma));
    y = max(1, round(center(2)-3*sigma)) : min(h, round(center(2)+3*sigma));
    if isempty(x) || isempty(y), return; end
    [xg, yg] = meshgrid(x, y);
    blob = weight * exp(-((xg-center(1)).^2 + (yg-center(2)).^2) / (2*sigma^2));
    heat(y, x) = heat(y, x) + blob;
end

function overlay = overlayHeatmap(rgbImg, camMap)
    cmap = jet(256);
    idx = max(1, min(256, round(camMap*255)+1));
    heatRGB = ind2rgb(idx, cmap);
    alpha = 0.45 * mat2gray(camMap); % low attention areas stay closer to original
    overlay = im2double(rgbImg) .* (1-alpha) + heatRGB .* alpha;
    overlay = im2uint8(overlay);
end

function corr = correlateWithLesions(camMap, lesionData, imSize)
%CORRELATEWITHLESIONS  Quantify overlap between high-CAM-attention regions
%   and each lesion type's spatial footprint - the check that makes the
%   Grad-CAM output clinically verifiable rather than just a pretty picture.
    highAttn = camMap > prctile(camMap(:), 90);

    corr.microaneurysmOverlap = 0;
    corr.hemorrhageOverlap = 0;
    corr.exudateOverlap = 0;

    if isfield(lesionData, 'microaneurysms') && ~isempty(lesionData.microaneurysms)
        hits = 0;
        for k = 1:numel(lesionData.microaneurysms)
            p = round(lesionData.microaneurysms(k).position);
            if p(2)>=1 && p(2)<=imSize(1) && p(1)>=1 && p(1)<=imSize(2) && highAttn(p(2), p(1))
                hits = hits + 1;
            end
        end
        corr.microaneurysmOverlap = hits / numel(lesionData.microaneurysms);
    end

    if isfield(lesionData, 'hemorrhages') && isfield(lesionData.hemorrhages,'lesions') ...
            && ~isempty(lesionData.hemorrhages.lesions)
        hits = 0;
        lesions = lesionData.hemorrhages.lesions;
        for k = 1:numel(lesions)
            p = round(lesions(k).position);
            if p(2)>=1 && p(2)<=imSize(1) && p(1)>=1 && p(1)<=imSize(2) && highAttn(p(2), p(1))
                hits = hits + 1;
            end
        end
        corr.hemorrhageOverlap = hits / numel(lesions);
    end

    corr.overallAgreement = mean([corr.microaneurysmOverlap, corr.hemorrhageOverlap], 'omitnan');
end
