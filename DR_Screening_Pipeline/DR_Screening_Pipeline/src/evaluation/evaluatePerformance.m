function metrics = evaluatePerformance(imageFolder, labelCsvPath, cnnNet, varargin)
%EVALUATEPERFORMANCE  Sensitivity/specificity/AUC for referable DR (ICDR>=2).
%
%   metrics = EVALUATEPERFORMANCE(imageFolder, labelCsvPath, cnnNet, ...)
%
%   labelCsvPath - CSV with columns {image, grade}, grade in {0,1,2,3,4}
%   cnnNet       - trained network (or [] for rule-only evaluation)
%
%   Name-Value:
%       'ShowROC'      true/false (default true)
%       'CompareTo'    struct array of benchmark systems to plot alongside,
%                      default = illustrative published reference points
%                      (see docs/VALIDATION_PLAN.md for sourcing caveats)
%
%   metrics struct: .sensitivity .specificity .accuracy .auc .confMat
%                   .perGradeAccuracy .n

    p = inputParser;
    addParameter(p, 'ShowROC', true);
    addParameter(p, 'CompareTo', defaultBenchmarks());
    parse(p, varargin{:});
    opt = p.Results;

    T = readtable(labelCsvPath);
    n = height(T);
    trueGrades = T.grade;
    predGrades = nan(n,1);
    predScores = nan(n,1); % probability/score of referable, for ROC/AUC

    fprintf('Evaluating %d images...\n', n);
    for i = 1:n
        if iscell(T.image)
            imgName = T.image{i};
        else
            imgName = char(string(T.image(i)));
        end
        imgPath = fullfile(imageFolder, imgName);
        if ~isfile(imgPath)
            warning('Missing file: %s (skipped)', imgPath);
            continue;
        end
        img = imread(imgPath);

        qc = assessImageQuality(img);
        if qc.needsEnhancement
            img = enhanceImage(img, qc);
        end
        [discInfo, ~] = detectOpticDiscFovea(img);
        [vesselMask, vesselDensity] = segmentVessels(img, discInfo.mask | true);
        maList = detectMicroaneurysms(img, discInfo, vesselMask);
        [~, exudateStats] = segmentExudates(img, discInfo);
        [~, hemStats] = detectHemorrhages(img, discInfo, vesselMask);
        nvInfo = detectNeovascularization(vesselMask, discInfo, true(size(vesselMask,1),size(vesselMask,2)));

        lesionData = struct('microaneurysms', maList, 'hemorrhages', hemStats, ...
                             'exudateStats', exudateStats, 'nvInfo', nvInfo, ...
                             'vesselDensity', vesselDensity);

        grading = gradeDRSeverity(img, lesionData, cnnNet);
        predGrades(i) = grading.icdrGrade;
        predScores(i) = grading.confidence * double(grading.referable) + ...
                         (1-grading.confidence) * (grading.icdrGrade/4); % monotonic referable-likelihood proxy

        if mod(i,25)==0, fprintf('  %d/%d done\n', i, n); end
    end

    valid = ~isnan(predGrades);
    trueGrades = trueGrades(valid); predGrades = predGrades(valid); predScores = predScores(valid);

    trueReferable = trueGrades >= 2;
    predReferable = predGrades >= 2;

    TP = sum(trueReferable & predReferable);
    FN = sum(trueReferable & ~predReferable);
    TN = sum(~trueReferable & ~predReferable);
    FP = sum(~trueReferable & predReferable);

    metrics.n = numel(trueGrades);
    metrics.sensitivity = TP / max(TP+FN, 1);
    metrics.specificity = TN / max(TN+FP, 1);
    metrics.accuracy = mean(predGrades == trueGrades);
    metrics.confMat = confusionmat(trueGrades, predGrades);

    perGrade = zeros(5,1);
    for g = 0:4
        idx = trueGrades == g;
        if any(idx)
            perGrade(g+1) = mean(predGrades(idx) == g);
        else
            perGrade(g+1) = NaN;
        end
    end
    metrics.perGradeAccuracy = perGrade;

    try
        [xROC, yROC, ~, auc] = perfcurve(trueReferable, predScores, true);
        metrics.auc = auc;
        metrics.rocX = xROC; metrics.rocY = yROC;
    catch
        metrics.auc = NaN;
    end

    fprintf('\n=== Referable DR (ICDR >= 2) Performance ===\n');
    fprintf('N = %d\n', metrics.n);
    fprintf('Sensitivity: %.1f%%  (target > 90%%)\n', metrics.sensitivity*100);
    fprintf('Specificity: %.1f%%  (target > 85%%)\n', metrics.specificity*100);
    fprintf('AUC:         %.3f\n', metrics.auc);
    fprintf('Overall 5-class accuracy: %.1f%%\n', metrics.accuracy*100);

    fprintf('\n=== Comparison to illustrative published benchmarks ===\n');
    fprintf('%-28s %10s %10s\n', 'System', 'Sens.', 'Spec.');
    fprintf('%-28s %9.1f%% %9.1f%%\n', 'This pipeline', metrics.sensitivity*100, metrics.specificity*100);
    for k = 1:numel(opt.CompareTo)
        b = opt.CompareTo(k);
        fprintf('%-28s %9.1f%% %9.1f%%  (%s)\n', b.name, b.sensitivity*100, b.specificity*100, b.note);
    end
    fprintf(['\nNOTE: benchmark rows above are illustrative reference points from the\n' ...
             'public literature and should be re-verified/re-cited against the specific\n' ...
             'published papers before use in any regulatory or clinical claim.\n']);

    if opt.ShowROC && isfield(metrics, 'rocX')
        figure('Name', 'ROC - Referable DR');
        plot(metrics.rocX, metrics.rocY, 'LineWidth', 2); hold on;
        plot([0 1],[0 1],'k--');
        xlabel('False Positive Rate'); ylabel('True Positive Rate');
        title(sprintf('ROC (AUC = %.3f) - Referable DR Detection', metrics.auc));
        grid on;
    end
end

function b = defaultBenchmarks()
    b(1) = struct('name','IDx-DR (pivotal trial)', 'sensitivity',0.872, 'specificity',0.907, ...
                   'note','FDA pivotal trial, illustrative - verify exact figures before citing');
    b(2) = struct('name','Google ARDA', 'sensitivity',0.90, 'specificity',0.90, ...
                   'note','reported range across validation cohorts, verify per paper/cohort');
    b(3) = struct('name','EyePACS baseline CNN', 'sensitivity',0.85, 'specificity',0.85, ...
                   'note','representative single-CNN baseline, not a specific paper');
end
