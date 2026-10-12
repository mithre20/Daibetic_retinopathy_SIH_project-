function grading = gradeDRSeverity(img, lesionData, cnnNet)
%GRADEDRSEVERITY  Fuse CNN prediction with ICDR lesion-count rule score.
%
%   grading = GRADEDRSEVERITY(img, lesionData, cnnNet)
%
%   lesionData - struct produced by main_pipeline, containing fields:
%                .microaneurysms (from detectMicroaneurysms)
%                .hemorrhages    (from detectHemorrhages)
%                .exudateStats   (from segmentExudates)
%                .nvInfo         (from detectNeovascularization)
%                .vesselDensity
%   cnnNet     - trained network from trainDRModel, or [] to run rule-only
%                (degraded mode, documented in README limitations)
%
%   grading struct:
%       .icdrGrade        integer 0-4
%       .gradeLabel       string
%       .referable        logical (grade >= 2)
%       .confidence       calibrated confidence in [0,1]
%       .cnnProbs         1x5 softmax vector (or [] if cnnNet not supplied)
%       .ruleScore        struct with per-criterion breakdown
%       .fusionWeight     weight given to CNN vs rule score
%
%   --- ICDR rule logic implemented (clinical mapping) ---
%     Grade 0 (No DR):        no lesions
%     Grade 1 (Mild NPDR):    microaneurysms only
%     Grade 2 (Moderate NPDR):more than just MAs but less than severe
%                              (some hemorrhages/exudates, definite MA count)
%     Grade 3 (Severe NPDR):  4-2-1 rule proxy: severe hemorrhages in all
%                              4 quadrants OR venous beading pattern proxy
%                              OR prominent IRMA proxy (approximated here
%                              via hemorrhage/exudate density + vessel
%                              caliber irregularity, since true 4-2-1
%                              quadrant analysis needs quadrant masks)
%     Grade 4 (PDR):          neovascularization suspected

    classNames = {'0_NoDR','1_Mild','2_Moderate','3_Severe','4_Proliferative'};

    % ---------------- Rule-based score (always available) ----------------
    maCount = numel(lesionData.microaneurysms);
    hemCounts = lesionData.hemorrhages;
    dotCount = hemCounts.dotCount;
    blotCount = hemCounts.blotCount;
    exudateFrac = lesionData.exudateStats.fractionOfFOV;
    nvSuspected = lesionData.nvInfo.nvdSuspected || lesionData.nvInfo.nveSuspected;

    ruleScore.maCount = maCount;
    ruleScore.dotCount = dotCount;
    ruleScore.blotCount = blotCount;
    ruleScore.exudateFrac = exudateFrac;
    ruleScore.nvSuspected = nvSuspected;

    if nvSuspected && lesionData.nvInfo.confidence > 0.3
        ruleGrade = 4;
    elseif blotCount >= 5 || (blotCount >= 2 && exudateFrac > 0.01) ...
            || (lesionData.vesselDensity > 0 && isfield(lesionData,'calibreFlag') && lesionData.calibreFlag)
        ruleGrade = 3;
    elseif maCount >= 5 || dotCount >= 1 || exudateFrac > 0.002
        ruleGrade = 2;
    elseif maCount >= 1
        ruleGrade = 1;
    else
        ruleGrade = 0;
    end

    % Rule-based confidence: more lesion evidence -> more confident;
    % borderline counts near a threshold -> lower confidence (flagged for review)
    ruleConfidence = 0.6 + 0.08*min(maCount,5) - 0.05*(ruleGrade>0 && maCount<=2);
    ruleConfidence = max(0.3, min(0.95, ruleConfidence));

    % ---------------- CNN score (if a trained model is provided) ----------------
    cnnProbs = [];
    if nargin >= 3 && ~isempty(cnnNet)
        inputSize = [224 224 3];
        if isprop(cnnNet, 'Layers')
            imResized = imresize(im2uint8(img), inputSize(1:2));
            if size(imResized,3) == 1
                imResized = repmat(imResized, 1, 1, 3);
            end
            cnnProbs = predict(cnnNet, imResized);
        end
    end

    % ---------------- Fusion ----------------
    if isempty(cnnProbs)
        finalGrade = ruleGrade;
        confidence = ruleConfidence;
        fusionWeight = 0; % 0 = pure rule-based (documented degraded mode)
    else
        [~, cnnGradeIdx] = max(cnnProbs);
        cnnGrade = cnnGradeIdx - 1;
        cnnConfidence = max(cnnProbs);

        % Weighted fusion: trust the CNN more when it is confident AND the
        % rule score doesn't flatly contradict it by >1 grade; otherwise
        % defer toward the more conservative (higher) grade for safety in
        % a screening context (false negatives are the costlier error).
        fusionWeight = 0.6; % CNN weight when both signals agree/are close
        if abs(cnnGrade - ruleGrade) > 1
            fusionWeight = 0.4; % disagreement -> trust lesion evidence more,
                                 % and flag for mandatory human review
        end
        blended = fusionWeight*cnnGrade + (1-fusionWeight)*ruleGrade;
        finalGrade = round(blended);
        % Safety net: never grade down from a strong rule-based severe/PDR signal
        if ruleGrade >= 3 && finalGrade < ruleGrade
            finalGrade = ruleGrade;
        end
        confidence = fusionWeight*cnnConfidence + (1-fusionWeight)*ruleConfidence;
        if abs(cnnGrade - ruleGrade) > 1
            confidence = confidence * 0.75; % penalize confidence on disagreement
        end
    end

    finalGrade = max(0, min(4, finalGrade));

    grading.icdrGrade = finalGrade;
    grading.gradeLabel = classNames{finalGrade + 1};
    grading.referable = finalGrade >= 2;
    grading.confidence = confidence;
    grading.cnnProbs = cnnProbs;
    grading.ruleScore = ruleScore;
    disagreement = false;
    if ~isempty(cnnProbs)
        [~, cnnPredIdx] = max(cnnProbs);
        disagreement = abs((cnnPredIdx - 1) - ruleGrade) > 1;
    end
    grading.needsManualReview = (confidence < 0.65) || disagreement;
end
