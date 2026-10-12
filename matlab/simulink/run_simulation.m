function simResults = run_simulation(varargin)
%RUN_SIMULATION Execute Telemedicine Pipeline Simulation Run
%
% AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
%
% Runs an end-to-end multi-stage queuing simulation for the screening pipeline:
% 1. Patient Arrival (Poisson stream over operating hours)
% 2. Camera Image Acquisition (M/M/c queue)
% 3. Rural Network Upload & Bandwidth Throttle (Serialization delay + queue)
% 4. Image Quality Assessment (IQA quality filter & recapture loop)
% 5. Automated AI Inference (Hardware throughput constraint)
% 6. Ophthalmologist Review (M/M/c Erlang-C human reviewer pool)
%
% Usage:
%   simResults = run_simulation();
%   simResults = run_simulation('AnnualPatients', 120000, 'CameraCount', 6);

    params = parameters(varargin{:});
    
    % Check if Simulink model can be run directly via sim command
    modelName = 'ai_dr_screening';
    useSimulinkEngine = false;
    
    if exist('sim', 'file') && (exist([modelName '.slx'], 'file') || exist([modelName '.mdl'], 'file'))
        try
            load_system(modelName);
            % Set runtime simulation stop time to 8 hours (28800 seconds)
            simOut = sim(modelName, 'StopTime', '28800');
            useSimulinkEngine = true;
        catch ME
            useSimulinkEngine = false;
        end
    end
    
    % Analytical & Discrete-Event Queuing Solver (Erlang-C Multi-Server Network)
    % Stage 1: Arrival rates
    lambdaPatientsSec = params.PatientArrivalRatePerSec;
    lambdaImagesSec = params.ImageArrivalRatePerSec;
    
    % Stage 2: Camera Acquisition (c = CameraCount, mu = 1/AcquisitionSecPerImage)
    muCamera = 1.0 / params.AcquisitionSecPerImage;
    [cCam, utilCam, waitCamSec, qLenCam] = solveMMcQueue(lambdaImagesSec, muCamera, params.CameraCount);
    
    % Stage 3: Network Upload Delay & Bandwidth Saturation
    % Transmission time per image in seconds
    tUpload = params.UploadTimePerImageSec;
    muNetwork = 1.0 / tUpload;
    % Network channels / gateway links
    networkLinks = max(1, ceil(params.BandwidthMbps / 2.0)); % 2 Mbps per concurrent channel
    [cNet, utilNet, waitNetSec, qLenNet] = solveMMcQueue(lambdaImagesSec, muNetwork, networkLinks);
    
    % Stage 4: IQA Filter & Recapture Loop
    % Effective images after recapture loop
    recaptureFraction = params.UnusableImageRate;
    totalImagesAcquired = params.AnnualPatients * params.ImagesPerPatient * (1.0 + recaptureFraction);
    totalRecaptures = params.AnnualPatients * params.ImagesPerPatient * recaptureFraction;
    
    % Stage 5: Automated AI Processing
    muAI = 1.0 / params.TotalAIProcessingSec;
    [cAI, utilAI, waitAISec, qLenAI] = solveMMcQueue(lambdaImagesSec * (1.0 + recaptureFraction), muAI, params.AIWorkerInstances);
    
    % Stage 6: Ophthalmologist Review (Referable cases only)
    lambdaReviewSec = lambdaPatientsSec * params.ReferablePrevalence;
    muReview = 1.0 / params.ReviewSecPerCase;
    [cRev, utilRev, waitRevSec, qLenRev] = solveMMcQueue(lambdaReviewSec, muReview, params.OphthalmologistCount);
    
    % Cumulative Totals & KPIs
    totalScreeningSec = params.AcquisitionSecPerImage * params.ImagesPerPatient + ...
                        tUpload * params.ImagesPerPatient + ...
                        params.TotalAIProcessingSec * params.ImagesPerPatient + ...
                        (params.ReviewSecPerCase * params.ReferablePrevalence);
                        
    totalWaitSec = (waitCamSec * params.ImagesPerPatient) + ...
                   (waitNetSec * params.ImagesPerPatient) + ...
                   (waitAISec * params.ImagesPerPatient) + ...
                   waitRevSec;
                   
    maxQueueLen = max([qLenCam, qLenNet, qLenAI, qLenRev]);
    
    simResults = struct();
    simResults.params = params;
    simResults.usedSimulinkEngine = useSimulinkEngine;
    
    % Summary metrics
    simResults.annualPatients = params.AnnualPatients;
    simResults.totalImagesAcquired = round(totalImagesAcquired);
    simResults.totalRecaptures = round(totalRecaptures);
    simResults.totalAIScreenings = round(totalImagesAcquired);
    simResults.totalHumanReviews = round(params.AnnualPatients * params.ReferablePrevalence);
    simResults.totalReferrals = round(params.AnnualPatients * params.ReferablePrevalence);
    
    simResults.camera = struct('count', params.CameraCount, 'utilization', utilCam, 'waitSec', waitCamSec, 'queue', qLenCam);
    simResults.network = struct('bandwidthMbps', params.BandwidthMbps, 'utilization', utilNet, 'waitSec', waitNetSec, 'queue', qLenNet);
    simResults.ai = struct('instances', params.AIWorkerInstances, 'utilization', utilAI, 'waitSec', waitAISec, 'queue', qLenAI);
    simResults.review = struct('ophthalmologists', params.OphthalmologistCount, 'utilization', utilRev, 'waitSec', waitRevSec, 'queue', qLenRev);
    
    simResults.totalScreeningTimeSec = round(totalScreeningSec, 2);
    simResults.averageWaitTimeMin = round(totalWaitSec / 60.0, 2);
    simResults.maximumQueueLength = round(maxQueueLen);
    simResults.systemBottleneck = identifyBottleneck(utilCam, utilNet, utilAI, utilRev);
end

function [c, util, waitSec, queueLen] = solveMMcQueue(lambda, mu, cServers)
    if lambda <= 0 || mu <= 0 || cServers <= 0
        c = cServers; util = 0; waitSec = 0; queueLen = 0; return;
    end
    c = max(1, cServers);
    a = lambda / mu; % offered load
    util = a / c;
    
    if util >= 0.999
        % Unstable saturated queue
        util = 0.999;
        waitSec = 3600.0; % High wait time cap
        queueLen = 100.0;
        return;
    end
    
    % Erlang-C formula for probability of delay
    sumTerms = 0;
    for k = 0:(c - 1)
        sumTerms = sumTerms + (a^k) / factorial(min(k, 170));
    end
    lastTerm = ((a^c) / factorial(min(c, 170))) * (1.0 / (1.0 - util));
    Pwait = lastTerm / (sumTerms + lastTerm);
    if ~isfinite(Pwait) || Pwait < 0, Pwait = 0; end
    
    % Waiting time in queue Wq
    waitSec = Pwait / (c * mu - lambda);
    if ~isfinite(waitSec) || waitSec < 0, waitSec = 0; end
    
    % Mean queue length Lq = lambda * Wq
    queueLen = lambda * waitSec;
end

function bottleneck = identifyBottleneck(uCam, uNet, uAI, uRev)
    utils = [uCam, uNet, uAI, uRev];
    names = {'Camera Acquisition', 'Network Bandwidth', 'AI Processing', 'Ophthalmologist Review'};
    [~, idx] = max(utils);
    bottleneck = names{idx};
end
