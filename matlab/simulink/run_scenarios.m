function scenarioResults = run_scenarios()
%RUN_SCENARIOS Execute Scenarios A, B, C, D, and E for Telemedicine Screening
%
% AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
%
% Compares 5 operational deployments:
% - Scenario A: Small Rural PHC (5,000 pts/yr, 1 camera, 1 Mbps, 1 doctor)
% - Scenario B: Medium PHC (25,000 pts/yr, 2 cameras, 4 Mbps, 2 doctors)
% - Scenario C: District-Level Screening (100,000+ pts/yr, centralized AI, review pool)
% - Scenario D: Peak Load Surge (1.8x arrival rush at morning mobile camp)
% - Scenario E: Improved Infrastructure (Edge AI nodes, 15 Mbps uplink, optimized capacity)
%
% Usage:
%   scenarioResults = run_scenarios();

    fprintf('======================================================================\n');
    fprintf('EXECUTING 5 TELEMEDICINE OPERATIONAL SCENARIOS\n');
    fprintf('======================================================================\n\n');

    % Scenario A: Small Rural PHC
    fprintf('[1/5] Simulating Scenario A: Small Rural PHC...\n');
    resA = run_simulation('AnnualPatients', 5000, ...
                          'CameraCount', 1, ...
                          'BandwidthMbps', 1.0, ...
                          'AIWorkerInstances', 1, ...
                          'OphthalmologistCount', 1);
                          
    % Scenario B: Medium PHC
    fprintf('[2/5] Simulating Scenario B: Medium PHC...\n');
    resB = run_simulation('AnnualPatients', 25000, ...
                          'CameraCount', 2, ...
                          'BandwidthMbps', 4.0, ...
                          'AIWorkerInstances', 1, ...
                          'OphthalmologistCount', 2);
                          
    % Scenario C: District-Level Screening (>= 100,000 patients/year)
    fprintf('[3/5] Simulating Scenario C: District-Level Screening (>=100,000 pts/yr)...\n');
    resC = run_simulation('AnnualPatients', 100000, ...
                          'CameraCount', 4, ...
                          'BandwidthMbps', 8.0, ...
                          'AIWorkerInstances', 2, ...
                          'OphthalmologistCount', 4);
                          
    % Scenario D: Peak Load Camp Surge (1.8x surge on District program)
    fprintf('[4/5] Simulating Scenario D: Peak Load Surge (Mobile Screening Camp)...\n');
    resD = run_simulation('AnnualPatients', 100000, ...
                          'WorkingHoursPerDay', 6, ... % shorter active screening camp window
                          'CameraCount', 4, ...
                          'BandwidthMbps', 4.0, ...   % degraded mobile camp network
                          'AIWorkerInstances', 2, ...
                          'OphthalmologistCount', 4);
                          
    % Scenario E: Improved Infrastructure (Edge AI + High Bandwidth)
    fprintf('[5/5] Simulating Scenario E: Improved Infrastructure (Edge AI + 15 Mbps)...\n');
    resE = run_simulation('AnnualPatients', 100000, ...
                          'CameraCount', 5, ...
                          'BandwidthMbps', 15.0, ...
                          'AIWorkerInstances', 4, ...
                          'OphthalmologistCount', 5);
                          
    scenarioResults = struct('A', resA, 'B', resB, 'C', resC, 'D', resD, 'E', resE);
    
    % Print Summary Comparison Table
    fprintf('\n----------------------------------------------------------------------------------------------------\n');
    fprintf('%-12s | %-10s | %-7s | %-10s | %-8s | %-12s | %-10s | %-15s\n', ...
        'Scenario', 'Annual Pts', 'Cameras', 'Bandwidth', 'Doctors', 'Wait (min)', 'Max Queue', 'Bottleneck');
    fprintf('----------------------------------------------------------------------------------------------------\n');
    
    scenarios = {'A: Small PHC', 'B: Med PHC', 'C: District', 'D: Peak Camp', 'E: Improved'};
    resultsList = {resA, resB, resC, resD, resE};
    
    for i = 1:5
        r = resultsList{i};
        fprintf('%-12s | %10d | %7d | %7.1f Mb | %8d | %12.2f | %10d | %-15s\n', ...
            scenarios{i}, r.annualPatients, r.camera.count, r.network.bandwidthMbps, ...
            r.review.ophthalmologists, r.averageWaitTimeMin, r.maximumQueueLength, r.systemBottleneck);
    end
    fprintf('----------------------------------------------------------------------------------------------------\n\n');
end
