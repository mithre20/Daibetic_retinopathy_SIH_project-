function planReport = capacityPlanningAnalysis(varargin)
%CAPACITYPLANNINGANALYSIS  Size a district DR-screening telemedicine program.
%
%   planReport = CAPACITYPLANNINGANALYSIS(...)
%
%   Analytical (M/M/1 and M/M/c) queueing model of the screening workflow:
%
%       Patients --camera--> [Upload queue, bandwidth-limited]
%                --AI--> [AI processing queue, GPU/CPU-limited]
%                --review--> [Ophthalmologist review queue, human-limited]
%
%   This does not require Simulink/SimEvents to run and gives immediate
%   sizing numbers; use buildScreeningWorkflowModel.m for a visual,
%   editable Simulink block-diagram version of the same logic.
%
%   Name-Value parameters (all optional, defaults reflect a mid-size
%   Tamil Nadu district program):
%       'AnnualPatients'        100000
%       'ScreeningDaysPerYear'  300     (accounts for holidays/downtime)
%       'HoursPerDay'           8
%       'ImageSizeMB'           2.5     (compressed fundus JPEG pair, 2 eyes)
%       'UplinkMbps'            4       (typical rural 4G/broadband uplink)
%       'AIProcessingSecPerImage' 8     (full pipeline incl. segmentation)
%       'ReviewMinPerCase'      0.5     (<30s target from the explainability
%                                        module, expressed in minutes)
%       'ReferableFraction'     0.18    (matches the ~18% DR prevalence
%                                        figure quoted in the brief - only
%                                        referable/flagged cases need full
%                                        ophthalmologist review; others get
%                                        a lighter auto-cleared pathway)
%       'TargetUtilization'     0.75    (queueing theory: keep servers below
%                                        ~75-80% utilization or wait times
%                                        explode; this is a design choice,
%                                        not a hard constraint)
%
%   planReport: struct with arrival rates, required camera stations,
%   required reviewers, expected wait times, and a sweep table showing
%   sensitivity of reviewer count to review time and referral rate.

    p = inputParser;
    addParameter(p, 'AnnualPatients', 100000);
    addParameter(p, 'ScreeningDaysPerYear', 300);
    addParameter(p, 'HoursPerDay', 8);
    addParameter(p, 'ImageSizeMB', 2.5);
    addParameter(p, 'UplinkMbps', 4);
    addParameter(p, 'AIProcessingSecPerImage', 8);
    addParameter(p, 'ReviewMinPerCase', 0.5);
    addParameter(p, 'ReferableFraction', 0.18);
    addParameter(p, 'TargetUtilization', 0.75);
    parse(p, varargin{:});
    o = p.Results;

    % --- Arrival rate ---
    totalScreeningHours = o.ScreeningDaysPerYear * o.HoursPerDay;
    lambdaPerHour = o.AnnualPatients / totalScreeningHours;
    lambdaPerSec = lambdaPerHour / 3600;

    fprintf('=== District DR Screening Capacity Plan ===\n');
    fprintf('Target volume: %d patients/year over %d screening days x %d h/day\n', ...
        o.AnnualPatients, o.ScreeningDaysPerYear, o.HoursPerDay);
    fprintf('Arrival rate: %.2f patients/hour (%.4f/sec)\n', lambdaPerHour, lambdaPerSec);

    % --- 1. Upload / bandwidth stage (M/M/1 per link; c parallel links if needed) ---
    uploadSecPerImage = (o.ImageSizeMB * 8) / o.UplinkMbps; % Mb / Mbps = sec
    muUpload = 1 / uploadSecPerImage; % services/sec per link
    [cUpload, rhoUpload, WqUpload] = sizeServers(lambdaPerSec, muUpload, o.TargetUtilization);
    fprintf('\n-- Upload / bandwidth stage --\n');
    fprintf('Per-case upload time: %.1f s at %.1f Mbps uplink\n', uploadSecPerImage, o.UplinkMbps);
    fprintf('Required parallel upload channels/PHC gateways: %d (util %.0f%%, avg wait %.1f s)\n', ...
        cUpload, rhoUpload*100, WqUpload);

    % --- 2. AI processing stage ---
    muAI = 1 / o.AIProcessingSecPerImage;
    [cAI, rhoAI, WqAI] = sizeServers(lambdaPerSec, muAI, o.TargetUtilization);
    fprintf('\n-- AI processing stage --\n');
    fprintf('Per-case processing time: %.1f s\n', o.AIProcessingSecPerImage);
    fprintf('Required parallel processing workers (CPU/GPU instances): %d (util %.0f%%, avg wait %.1f s)\n', ...
        cAI, rhoAI*100, WqAI);

    % --- 3. Human review stage (only referable/flagged cases queue here) ---
    lambdaReview = lambdaPerSec * o.ReferableFraction;
    muReview = 1 / (o.ReviewMinPerCase * 60);
    [cReview, rhoReview, WqReview] = sizeServers(lambdaReview, muReview, o.TargetUtilization);
    fprintf('\n-- Ophthalmologist review stage (referable/flagged cases only) --\n');
    fprintf('Referable arrival rate: %.4f/sec (%.1f%% of all cases)\n', lambdaReview, o.ReferableFraction*100);
    fprintf('Per-case review time: %.1f s (<=30s target)\n', o.ReviewMinPerCase*60);
    fprintf('Required reviewing ophthalmologists/technicians ONLINE at any time: %d (util %.0f%%, avg wait %.1f s)\n', ...
        cReview, rhoReview*100, WqReview);
    reviewerHoursPerDay = cReview * o.HoursPerDay;
    fprintf('  (i.e. %d reviewer-hours/day needed; can be covered by fewer people\n   working longer, or shared across multiple screening sites via telemedicine)\n', ...
        reviewerHoursPerDay);

    % --- Sensitivity sweep: reviewer count vs. review time & referral rate ---
    reviewTimes = [0.25 0.5 0.75 1.0]; % minutes
    referralFracs = [0.10 0.18 0.25];
    sweep = zeros(numel(reviewTimes), numel(referralFracs));
    for i = 1:numel(reviewTimes)
        for j = 1:numel(referralFracs)
            lam = lambdaPerSec * referralFracs(j);
            mu = 1/(reviewTimes(i)*60);
            [c, ~, ~] = sizeServers(lam, mu, o.TargetUtilization);
            sweep(i,j) = c;
        end
    end
    fprintf('\n-- Sensitivity: required concurrent reviewers --\n');
    fprintf('%-22s', 'Review time \\ Referral%');
    for j = 1:numel(referralFracs), fprintf('%10.0f%%', referralFracs(j)*100); end
    fprintf('\n');
    for i = 1:numel(reviewTimes)
        fprintf('%-22s', sprintf('%.2f min', reviewTimes(i)));
        for j = 1:numel(referralFracs), fprintf('%10d', sweep(i,j)); end
        fprintf('\n');
    end

    planReport.lambdaPerHour = lambdaPerHour;
    planReport.upload = struct('servers', cUpload, 'utilization', rhoUpload, 'waitSec', WqUpload);
    planReport.aiProcessing = struct('servers', cAI, 'utilization', rhoAI, 'waitSec', WqAI);
    planReport.review = struct('servers', cReview, 'utilization', rhoReview, 'waitSec', WqReview, ...
                                'reviewerHoursPerDay', reviewerHoursPerDay);
    planReport.sensitivitySweep = struct('reviewTimesMin', reviewTimes, ...
                                          'referralFractions', referralFracs, 'requiredServers', sweep);
end

function [c, rho, Wq] = sizeServers(lambda, mu, targetUtil)
%SIZESERVERS  Smallest c (M/M/c servers) keeping utilization <= targetUtil,
%   with the resulting M/M/c average queue wait time Wq (Erlang-C).
    if lambda <= 0
        c = 0; rho = 0; Wq = 0; return;
    end
    c = max(1, ceil(lambda / (mu * targetUtil)));
    a = lambda / mu; % offered load (Erlangs)
    rho = a / c;
    % Erlang-C probability of waiting
    sumTerm = 0;
    for k = 0:c-1
        sumTerm = sumTerm + a^k / factorial(k);
    end
    lastTerm = (a^c / factorial(c)) * (1/(1-rho));
    Pwait = lastTerm / (sumTerm + lastTerm);
    Wq = (Pwait / (c*mu - lambda));
    if ~isfinite(Wq) || Wq < 0, Wq = 0; end
end
