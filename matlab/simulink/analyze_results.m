function analysis = analyze_results(simResults)
%ANALYZE_RESULTS Post-process and summarize simulation outputs
%
% AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
%
% Generates analytical summary metrics for screening throughput, communication
% delay, queue wait times, and bottleneck identification.

    if nargin < 1 || isempty(simResults)
        simResults = run_simulation();
    end

    fprintf('======================================================================\n');
    fprintf('TELEMEDICINE SCREENING PIPELINE — SIMULATION ANALYSIS REPORT\n');
    fprintf('======================================================================\n');

    p = simResults.params;
    fprintf('Annual Patient Volume:      %d patients/year\n', simResults.annualPatients);
    fprintf('Total Images Acquired:      %d images (incl. %d recaptures, %.1f%%)\n', ...
        simResults.totalImagesAcquired, simResults.totalRecaptures, p.UnusableImageRate*100);
    fprintf('Total Automated Screenings: %d image passes\n', simResults.totalAIScreenings);
    fprintf('Referrals to Human Review:  %d cases (%.1f%% of cohort)\n', ...
        simResults.totalHumanReviews, p.ReferablePrevalence*100);
        
    fprintf('\nSubsystem Performance & Queueing Metrics:\n');
    fprintf('  1. Fundus Camera Acquisition: %d units | Util: %.1f%% | Avg Queue: %.1f | Avg Wait: %.1fs\n', ...
        simResults.camera.count, simResults.camera.utilization*100, simResults.camera.queue, simResults.camera.waitSec);
    fprintf('  2. Network Transmission:      %.1f Mbps | Util: %.1f%% | Avg Queue: %.1f | Avg Wait: %.1fs\n', ...
        simResults.network.bandwidthMbps, simResults.network.utilization*100, simResults.network.queue, simResults.network.waitSec);
    fprintf('  3. Automated AI Engine:       %d instances | Util: %.1f%% | Avg Queue: %.1f | Avg Wait: %.1fs\n', ...
        simResults.ai.instances, simResults.ai.utilization*100, simResults.ai.queue, simResults.ai.waitSec);
    fprintf('  4. Ophthalmologist Review:    %d doctors | Util: %.1f%% | Avg Queue: %.1f | Avg Wait: %.1fs\n', ...
        simResults.review.ophthalmologists, simResults.review.utilization*100, simResults.review.queue, simResults.review.waitSec);
        
    fprintf('\nEnd-to-End System Performance:\n');
    fprintf('  - Average Total Patient Waiting Time: %.2f minutes\n', simResults.averageWaitTimeMin);
    fprintf('  - Total Automated Pipeline Latency:  %.2f seconds / image\n', p.TotalAIProcessingSec);
    fprintf('  - Maximum Concurrent Queue:           %d patients\n', simResults.maximumQueueLength);
    fprintf('  - Primary System Bottleneck:          %s\n', simResults.systemBottleneck);
    fprintf('======================================================================\n\n');

    analysis = simResults;
end
