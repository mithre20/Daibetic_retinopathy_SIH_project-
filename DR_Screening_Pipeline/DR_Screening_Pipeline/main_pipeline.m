function result = main_pipeline(imagePath, cnnNet, reportOutPath)
%MAIN_PIPELINE  End-to-end DR screening on a single fundus image.
%
%   result = MAIN_PIPELINE(imagePath, cnnNet, reportOutPath)
%
%   imagePath     - path to a fundus photo (jpg/png)
%   cnnNet        - (optional) trained network from trainDRModel; if
%                   omitted or [], runs in rule-based-only degraded mode
%   reportOutPath - (optional) where to save the PNG report
%                   (default: '<imagePath>_report.png')
%
%   result struct: .qc .discInfo .foveaInfo .vesselMask .maList
%                  .exudateStats .hemStats .nvInfo .grading .reportPath
%
%   Example:
%       result = main_pipeline('data/sample_fundus.jpg');
%       disp(result.grading)

    if nargin < 2, cnnNet = []; end

    % Robust path resolution if relative path not found in current pwd
    if ~exist(imagePath, 'file')
        thisDir = fileparts(mfilename('fullpath'));
        cand1 = fullfile(thisDir, imagePath);
        if exist(cand1, 'file')
            imagePath = cand1;
        else
            [~, fn, fe] = fileparts(imagePath);
            candData = fullfile(thisDir, 'data', [fn fe]);
            if exist(candData, 'file')
                imagePath = candData;
            else
                candWhich = which([fn fe]);
                if ~isempty(candWhich)
                    imagePath = candWhich;
                end
            end
        end
    end

    if nargin < 3 || isempty(reportOutPath)
        [d,f,~] = fileparts(imagePath);
        if isempty(d), d = pwd; end
        reportOutPath = fullfile(d, [f '_report.png']);
    end

    fprintf('--- DR Screening Pipeline ---\n');
    fprintf('Loading: %s\n', imagePath);
    img = imread(imagePath);

    % 1. Quality assessment + enhancement
    fprintf('[1/5] Assessing image quality...\n');
    qc = assessImageQuality(img);
    if ~qc.isGradable
        fprintf('  REJECTED: %s\n', qc.recaptureMsg);
        result.qc = qc;
        result.status = 'rejected';
        return;
    end
    if qc.needsEnhancement
        fprintf('  Applying adaptive enhancement (%s)\n', qc.recaptureMsg);
        img = enhanceImage(img, qc);
    else
        fprintf('  Quality OK (score %.2f), no enhancement needed.\n', qc.score);
    end

    % 2. Structure segmentation
    fprintf('[2/5] Segmenting retinal structures...\n');
    [discInfo, foveaInfo] = detectOpticDiscFovea(img);
    fovMask = true(size(img,1), size(img,2)); % whole-frame FOV proxy; refine with real mask if available
    [vesselMask, vesselDensity] = segmentVessels(img, fovMask);
    maList = detectMicroaneurysms(img, discInfo, vesselMask);
    [exudateMask, exudateStats] = segmentExudates(img, discInfo); %#ok<ASGLU>
    [hemMask, hemStats] = detectHemorrhages(img, discInfo, vesselMask); %#ok<ASGLU>
    nvInfo = detectNeovascularization(vesselMask, discInfo, fovMask);
    fprintf('  MAs: %d | Dot hem: %d | Blot hem: %d | Exudate area: %.2f%% | NV flag: %d\n', ...
        numel(maList), hemStats.dotCount, hemStats.blotCount, ...
        exudateStats.fractionOfFOV*100, nvInfo.nvdSuspected || nvInfo.nveSuspected);

    % 3. Severity grading
    fprintf('[3/5] Grading DR severity...\n');
    lesionData = struct('microaneurysms', maList, 'hemorrhages', hemStats, ...
                         'exudateStats', exudateStats, 'nvInfo', nvInfo, ...
                         'vesselDensity', vesselDensity);
    grading = gradeDRSeverity(img, lesionData, cnnNet);
    fprintf('  ICDR Grade %d (%s) | Referable: %d | Confidence: %.0f%%\n', ...
        grading.icdrGrade, grading.gradeLabel, grading.referable, grading.confidence*100);
    if grading.needsManualReview
        fprintf('  >> FLAGGED for priority manual review.\n');
    end

    % 4. Explainability
    fprintf('[4/5] Generating explainability outputs...\n');
    [overlayImg, camMap, lesionCorr] = generateGradCAM(img, cnnNet, grading.icdrGrade, lesionData); %#ok<ASGLU>
    reportPath = generateClinicalReport(img, overlayImg, grading, lesionData, qc, reportOutPath);
    fprintf('  Report saved: %s\n', reportPath);

    fprintf('[5/5] Done.\n');

    result.qc = qc;
    result.discInfo = discInfo;
    result.foveaInfo = foveaInfo;
    result.vesselMask = vesselMask;
    result.vesselDensity = vesselDensity;
    result.maList = maList;
    result.exudateStats = exudateStats;
    result.hemStats = hemStats;
    result.nvInfo = nvInfo;
    result.grading = grading;
    result.camMap = camMap;
    result.lesionCorrelation = lesionCorr;
    result.reportPath = reportPath;
    result.status = 'graded';
end
