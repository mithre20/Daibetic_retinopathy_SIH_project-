function reportPath = generateClinicalReport(img, overlayImg, grading, lesionData, qc, outPath)
%GENERATECLINICALREPORT  One-page annotated report for <30-second review.
%
%   reportPath = GENERATECLINICALREPORT(img, overlayImg, grading, lesionData, qc, outPath)
%
%   Produces a single figure (saved as PNG, and PDF if requested via file
%   extension) laid out for rapid ophthalmologist triage:
%     - Original image | Grad-CAM overlay, side by side
%     - ICDR grade, referable flag, confidence, review-needed flag
%     - Lesion count summary table
%     - Quality-assessment note (so reviewers know if grading ran on an
%       enhanced/borderline image)
%
%   This is intentionally a single glance-able panel: the 30-second budget
%   in a human-in-the-loop workflow means the report must lead with the
%   decision and the evidence for it, not bury it in prose.

    if nargin < 6 || isempty(outPath)
        outPath = fullfile(pwd, 'DR_report.png');
    end

    fig = figure('Visible', 'off', 'Position', [100 100 1200 750], 'Color', 'w');

    % --- Images ---
    subplot(2,3,1);
    imshow(img); title('Original (post-enhancement)', 'FontSize', 10);

    subplot(2,3,2);
    imshow(overlayImg); title('Grad-CAM Attention', 'FontSize', 10);

    subplot(2,3,3);
    axis off;
    gradeColors = [0.2 0.7 0.2; 0.6 0.8 0.2; 0.95 0.6 0.1; 0.9 0.3 0.1; 0.7 0.1 0.1];
    gc = gradeColors(min(max(grading.icdrGrade+1,1),5), :);
    rectangle('Position',[0.05 0.55 0.9 0.35], 'FaceColor', gc, 'Curvature', 0.15);
    text(0.5, 0.72, sprintf('ICDR Grade %d', grading.icdrGrade), ...
        'HorizontalAlignment','center', 'FontSize', 18, 'FontWeight','bold', 'Color','w');
    text(0.5, 0.60, strrep(grading.gradeLabel, '_', ' '), ...
        'HorizontalAlignment','center', 'FontSize', 12, 'Color','w');
    referText = 'NOT REFERABLE';
    if grading.referable, referText = 'REFERABLE - refer to ophthalmologist'; end
    text(0.5, 0.42, referText, 'HorizontalAlignment','center', 'FontSize', 12, 'FontWeight','bold');
    text(0.5, 0.30, sprintf('Confidence: %.0f%%', grading.confidence*100), ...
        'HorizontalAlignment','center', 'FontSize', 11);
    if grading.needsManualReview
        text(0.5, 0.16, '\bf FLAGGED FOR PRIORITY REVIEW', ...
            'HorizontalAlignment','center', 'FontSize', 11, 'Color', [0.8 0 0], 'Interpreter','tex');
    end
    xlim([0 1]); ylim([0 1]);

    % --- Lesion count table ---
    subplot(2,3,4:5);
    axis off;
    rs = grading.ruleScore;
    rows = {
        'Microaneurysms', num2str(rs.maCount);
        'Dot hemorrhages', num2str(rs.dotCount);
        'Blot hemorrhages', num2str(rs.blotCount);
        'Exudate area (% of FOV)', sprintf('%.2f%%', rs.exudateFrac*100);
        'Neovascularization flag', ternary(rs.nvSuspected, 'SUSPECTED - verify manually', 'Not detected');
    };
    yPos = 0.9;
    text(0, yPos+0.08, 'Lesion Evidence Summary', 'FontWeight','bold', 'FontSize', 12);
    for r = 1:size(rows,1)
        text(0.02, yPos - (r-1)*0.16, rows{r,1}, 'FontSize', 10);
        text(0.55, yPos - (r-1)*0.16, rows{r,2}, 'FontSize', 10);
    end
    if ~isempty(grading.cnnProbs)
        text(0.02, yPos - size(rows,1)*0.16 - 0.05, ...
            sprintf('CNN/rule fusion weight (CNN side): %.0f%%', grading.fusionWeight*100), ...
            'FontSize', 9, 'Color', [0.3 0.3 0.3]);
    else
        text(0.02, yPos - size(rows,1)*0.16 - 0.05, ...
            'Rule-based grading only (no trained CNN loaded)', ...
            'FontSize', 9, 'Color', [0.6 0.2 0.2]);
    end
    xlim([0 1]); ylim([0 1]);

    % --- QC note ---
    subplot(2,3,6);
    axis off;
    qcLines = {sprintf('Image quality score: %.2f', qc.score)};
    if isfield(qc, 'needsEnhancement') && qc.needsEnhancement
        qcLines{end+1} = 'Adaptive enhancement was applied.';
    end
    if ~isempty(qc.reason)
        qcLines{end+1} = ['Notes: ' strjoin(qc.reason, '; ')];
    end
    text(0, 0.9, 'Capture Quality', 'FontWeight','bold', 'FontSize', 12);
    for k = 1:numel(qcLines)
        text(0.02, 0.9 - k*0.18, qcLines{k}, 'FontSize', 9);
    end
    text(0.02, 0.05, sprintf('Generated: %s', datestr(now)), 'FontSize', 8, 'Color', [0.5 0.5 0.5]);
    xlim([0 1]); ylim([0 1]);

    sgtitle('Diabetic Retinopathy Screening Report - For Ophthalmologist Review', ...
        'FontSize', 13, 'FontWeight', 'bold');

    exportgraphics(fig, outPath, 'Resolution', 150);
    close(fig);
    reportPath = outPath;
end

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
