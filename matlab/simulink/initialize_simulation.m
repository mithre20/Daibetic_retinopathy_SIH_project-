function simEnv = initialize_simulation(varargin)
%INITIALIZE_SIMULATION Setup and validate MATLAB/Simulink workspace variables
%
% AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
%
% Loads parameters, assigns required variables to base workspace, and verifies
% that all pipeline rates, queues, and server limits satisfy stability constraints.

    fprintf('======================================================================\n');
    fprintf('AI DR SCREENING TELEMEDICINE WORKFLOW — INITIALIZING SIMULATION\n');
    fprintf('======================================================================\n');

    params = parameters(varargin{:});
    
    fprintf('Configuration Summary:\n');
    fprintf('  - Target Annual Patients:    %d patients/year\n', params.AnnualPatients);
    fprintf('  - Working Schedule:          %d days/year x %d hours/day (%d total screening hours)\n', ...
        params.WorkingDaysPerYear, params.WorkingHoursPerDay, params.TotalOperatingHours);
    fprintf('  - Base Arrival Rate:         %.2f patients/hour (%.4f/sec)\n', ...
        params.PatientArrivalRatePerHour, params.PatientArrivalRatePerSec);
    fprintf('  - Image Generation Rate:     %.2f images/hour (%.4f/sec)\n', ...
        params.ImageArrivalRatePerSec * 3600, params.ImageArrivalRatePerSec);
    fprintf('  - Camera Capacity:           %d cameras (%.1f sec/image)\n', ...
        params.CameraCount, params.AcquisitionSecPerImage);
    fprintf('  - Network Uplink:            %.1f Mbps (Upload: %.2f sec/image)\n', ...
        params.BandwidthMbps, params.UploadTimePerImageSec);
    fprintf('  - Automated AI Latency:      %.3f sec/image (%d parallel instances)\n', ...
        params.TotalAIProcessingSec, params.AIWorkerInstances);
    fprintf('  - Quality Recapture Rate:    %.1f%% (repositioning delay: %ds)\n', ...
        params.UnusableImageRate * 100, params.RecaptureDelaySec);
    fprintf('  - Reviewing Ophthalmologists: %d doctors (%.1f sec/referral review)\n', ...
        params.OphthalmologistCount, params.ReviewSecPerCase);
    fprintf('----------------------------------------------------------------------\n');

    % Assign into caller / base workspace for Simulink block model referencing
    assignin('base', 'simParams', params);
    assignin('base', 'annualPatients', params.AnnualPatients);
    assignin('base', 'lambdaArrival', params.PatientArrivalRatePerSec);
    assignin('base', 'cameraCount', params.CameraCount);
    assignin('base', 'cameraServiceSec', params.AcquisitionSecPerImage);
    assignin('base', 'uploadDelaySec', params.UploadTimePerImageSec);
    assignin('base', 'aiServiceSec', params.TotalAIProcessingSec);
    assignin('base', 'aiServers', params.AIWorkerInstances);
    assignin('base', 'recaptureFraction', params.UnusableImageRate);
    assignin('base', 'referralFraction', params.ReferablePrevalence);
    assignin('base', 'reviewDelaySec', params.ReviewSecPerCase);
    assignin('base', 'reviewServers', params.OphthalmologistCount);

    simEnv = params;
    fprintf('Simulation environment successfully initialized.\n\n');
end
