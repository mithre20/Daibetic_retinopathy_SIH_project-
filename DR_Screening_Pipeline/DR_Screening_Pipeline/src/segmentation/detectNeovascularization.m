function nvInfo = detectNeovascularization(vesselMask, discInfo, fovMask)
%DETECTNEOVASCULARIZATION  Heuristic screening flag for possible neovascularization.
%
%   nvInfo = DETECTNEOVASCULARIZATION(vesselMask, discInfo, fovMask)
%
%   nvInfo: struct with
%       .suspiciousRegions  bwconncomp-style list of candidate ROIs
%       .nvdSuspected       logical, abnormal vascular tufting at/near disc (NVD)
%       .nveSuspected       logical, abnormal tufting elsewhere (NVE)
%       .confidence         [0,1], LOW by design — see caveat below
%
%   *** IMPORTANT CLINICAL CAVEAT ***
%   True neovascularization (fine, chaotic, leaky new vessels) is reliably
%   distinguished from normal vessels mainly on fluorescein angiography
%   (FA), not color fundus photography, and even on fundus photos the
%   state of the art uses dedicated, separately-validated deep models with
%   large NV-labeled datasets (which are scarce — PDR is a minority class
%   in every public DR dataset). This function is a texture/topology
%   HEURISTIC that flags regions of abnormally high vessel-caliber
%   irregularity and local vessel density near the disc or in the
%   periphery. It exists to route a case for MANDATORY manual review, and
%   must never be used as a standalone NV diagnosis.

    if nargin < 3 || isempty(fovMask)
        fovMask = true(size(vesselMask));
    end

    % Vessel-caliber irregularity: skeletonize, measure local thickness via
    % distance transform along the skeleton; NV tufts show abrupt thin/thick
    % transitions and dense small loops that normal vessels rarely show.
    skel = bwskel(vesselMask, 'MinBranchLength', 5);
    distT = bwdist(~vesselMask);
    calibre = distT .* skel;

    % Sliding-window local density + calibre-variance map
    winSize = round(size(vesselMask,1) / 20);
    winSize = max(winSize, 9);
    densityMap = imboxfilt(double(vesselMask), winSize*2+1);
    calibreVarMap = stdfilt(calibre, true(winSize));

    if any(skel(:))
        skelMean = mean(calibreVarMap(skel));
        skelStd  = std(calibreVarMap(skel));
    else
        skelMean = 0;
        skelStd  = 0;
    end
    abnormalMap = (densityMap > (mean(densityMap(fovMask)) + 2*std(densityMap(fovMask)))) & ...
                  (calibreVarMap > (skelMean + 1.5*skelStd + eps));
    abnormalMap = abnormalMap & fovMask;
    abnormalMap = bwareaopen(abnormalMap, round(numel(vesselMask)*0.0002));

    cc = bwconncomp(abnormalMap);
    stats = regionprops(cc, 'Centroid', 'Area', 'BoundingBox');

    nvdSuspected = false;
    nveSuspected = false;
    if ~isempty(discInfo) && isfield(discInfo, 'center')
        nvdZoneR = discInfo.radius * 2; % NVD = within ~1 disc diameter of margin
    else
        nvdZoneR = inf;
    end

    for k = 1:numel(stats)
        if ~isempty(discInfo) && isfield(discInfo, 'center')
            d = norm(stats(k).Centroid - discInfo.center);
            if d <= nvdZoneR
                nvdSuspected = true;
                continue;
            end
        end
        nveSuspected = true;
    end

    nvInfo.suspiciousRegions = stats;
    nvInfo.nvdSuspected = nvdSuspected;
    nvInfo.nveSuspected = nveSuspected;
    % Confidence deliberately capped — this signal alone should never drive
    % a PDR classification without human confirmation.
    nvInfo.confidence = min(0.5, 0.15 * numel(stats));
end
