function params = parameters(varargin)
%PARAMETERS Global simulation parameters for AI DR Screening Telemedicine
%
% AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
%
% Defines default parameters and assumptions for:
% 1. Patient Arrival (M/M/c or Poisson stream)
% 2. Image Acquisition (Fundus camera capacity & operator availability)
% 3. Network & Bandwidth Constraints (Uplink bandwidth, transfer latency)
% 4. Image Quality Assessment (IQA quality rates & recapture loops)
% 5. Automated AI Processing (Measured inference latencies for EfficientNet, U-Net, Grad-CAM, SHAP)
% 6. Human Review (Ophthalmologist review capacity, triage, waiting queues)
%
% Usage:
%   params = parameters();
%   params = parameters('AnnualPatients', 120000, 'BandwidthMbps', 10.0);

    p = inputParser;
    
    % --- District Program & Patient Arrival Assumptions ---
    addParameter(p, 'AnnualPatients', 100000);             % Target screening volume per year
    addParameter(p, 'WorkingDaysPerYear', 300);            % Operational screening days (excl. holidays)
    addParameter(p, 'WorkingHoursPerDay', 8);              % Operational clinic hours per day
    addParameter(p, 'PeakArrivalMultiplier', 1.6);         % Peak morning rush multiplier (e.g. 9am-12pm)
    
    % --- Image Acquisition (Camera & Operator) ---
    addParameter(p, 'ImagesPerPatient', 2);                % Bilateral screening (2 fundus photos per patient)
    addParameter(p, 'AcquisitionSecPerImage', 45);         % Alignment, positioning, and flash capture (sec)
    addParameter(p, 'CameraCount', 4);                     % Number of active non-mydriatic fundus cameras
    addParameter(p, 'CameraAvailability', 0.95);           % Camera uptime / hardware availability
    
    % --- Network Bandwidth & Communication Delay ---
    addParameter(p, 'ImageSizeMB', 2.5);                   % Compressed 2048x2048 high-quality fundus JPEG (MB)
    addParameter(p, 'BandwidthMbps', 4.0);                 % Rural 3G/4G or broadband uplink speed (Mbps)
    addParameter(p, 'NetworkLatencySec', 0.15);            % Fixed round-trip ping / handshake delay (sec)
    addParameter(p, 'NetworkPacketLossRate', 0.01);        % Network retransmission packet loss rate
    
    % --- Image Quality Assessment (IQA) & Recapture ---
    addParameter(p, 'GoodImageRate', 0.82);                % Fraction of images meeting diagnostic quality (>80%)
    addParameter(p, 'UnusableImageRate', 0.18);            % Fraction requiring prompt on-site recapture
    addParameter(p, 'MaxRecapturesPerPatient', 2);         % Maximum recapture attempts allowed per patient
    addParameter(p, 'RecaptureDelaySec', 60);              % Time to reposition patient and take recapture (sec)
    
    % --- Automated AI Processing Throughput (Empirically Measured on Host) ---
    % Measured values: IQA=11ms, EfficientNet=24ms, ResNet=63ms, U-Net=391ms, Grad-CAM=121ms
    % Retinal Analysis: OpticDisc=8.6ms, Fovea=3.8ms, Vessels=16.2ms, Lesions=39.6ms, Full Retinal Pipeline=100.7ms
    addParameter(p, 'IQAInferenceSec', 0.012);             % IQA MobileNetV3-Small latency (sec)
    addParameter(p, 'DRInferenceSec', 0.025);              % EfficientNet-B0 DR classification latency (sec)
    addParameter(p, 'UNetSegmentationSec', 0.390);         % U-Net multi-class lesion segmentation latency (sec)
    addParameter(p, 'GradCAMSec', 0.120);                  % Grad-CAM heatmap generation latency (sec)
    addParameter(p, 'ReferralEngineSec', 0.001);           % Rule-based referral decision engine (sec)
    addParameter(p, 'OpticDiscSec', 0.0086);               % Optic disc localization latency (sec)
    addParameter(p, 'FoveaSec', 0.0038);                   % Fovea localization latency (sec)
    addParameter(p, 'VesselSegmentationSec', 0.0162);      % Retinal vessel segmentation latency (sec)
    addParameter(p, 'LesionAnalysisSec', 0.0396);          % Microaneurysm, Hemorrhage & Exudate analysis (sec)
    addParameter(p, 'RetinalPipelineSec', 0.1007);         % Complete retinal structural + lesion analysis (sec)
    addParameter(p, 'AIWorkerInstances', 2);               % Number of parallel AI worker processes/threads
    
    % --- Ophthalmologist Review Capacity (Human-in-the-Loop) ---
    addParameter(p, 'ReferablePrevalence', 0.18);          % Empirical referable DR rate (~18% needing review)
    addParameter(p, 'ReviewSecPerCase', 30);               % Target review time with AI decision-support report
    addParameter(p, 'ManualReviewSecPerCase', 180);        % Manual unassisted review time without AI (3 min)
    addParameter(p, 'OphthalmologistCount', 4);            % Number of remote reviewing ophthalmologists online
    addParameter(p, 'TargetUtilization', 0.75);            % Safe operational queue utilization target (<=75%)
    
    parse(p, varargin{:});
    params = p.Results;
    
    % --- Derived Metrics ---
    totalOperatingHours = params.WorkingDaysPerYear * params.WorkingHoursPerDay;
    params.TotalOperatingHours = totalOperatingHours;
    params.PatientArrivalRatePerHour = params.AnnualPatients / totalOperatingHours;
    params.PatientArrivalRatePerSec = params.PatientArrivalRatePerHour / 3600;
    params.ImageArrivalRatePerSec = params.PatientArrivalRatePerSec * params.ImagesPerPatient;
    
    % Upload time per image: (ImageSize in Megabits) / (Bandwidth in Mbps) + Latency
    params.UploadTimePerImageSec = (params.ImageSizeMB * 8.0) / params.BandwidthMbps + params.NetworkLatencySec;
    
    % Total AI execution time per image (sec)
    params.TotalAIProcessingSec = params.IQAInferenceSec + params.DRInferenceSec + ...
                                  params.UNetSegmentationSec + params.GradCAMSec + ...
                                  params.ReferralEngineSec;
end
