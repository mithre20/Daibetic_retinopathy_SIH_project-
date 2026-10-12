function export_results(simResults, outDir)
%EXPORT_RESULTS Export simulation results to CSV and JSON formats
%
% AI DR Screen — Explainable AI for Diabetic Retinopathy Screening in Rural India
%
% Exports:
%   - district_screening_summary.csv
%   - district_screening_summary.json
%   - matlab/results/throughput/throughput_summary.csv
%   - matlab/results/bandwidth/bandwidth_sweep.csv
%   - matlab/results/queue/queue_statistics.csv

    if nargin < 1 || isempty(simResults)
        simResults = run_simulation();
    end
    if nargin < 2 || isempty(outDir)
        outDir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
    end
    
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end
    
    % 1. Export district_screening_summary.csv
    csvPath = fullfile(outDir, 'district_screening_summary.csv');
    fid = fopen(csvPath, 'w');
    if fid ~= -1
        fprintf(fid, 'Metric,Value,Unit\n');
        fprintf(fid, 'AnnualPatients,%d,patients/year\n', simResults.annualPatients);
        fprintf(fid, 'TotalImagesAcquired,%d,images/year\n', simResults.totalImagesAcquired);
        fprintf(fid, 'TotalRecaptures,%d,images/year\n', simResults.totalRecaptures);
        fprintf(fid, 'TotalAIScreenings,%d,image-screenings/year\n', simResults.totalAIScreenings);
        fprintf(fid, 'TotalHumanReviews,%d,cases/year\n', simResults.totalHumanReviews);
        fprintf(fid, 'CameraCount,%d,units\n', simResults.camera.count);
        fprintf(fid, 'CameraUtilization,%.4f,fraction\n', simResults.camera.utilization);
        fprintf(fid, 'BandwidthMbps,%.1f,Mbps\n', simResults.network.bandwidthMbps);
        fprintf(fid, 'NetworkUtilization,%.4f,fraction\n', simResults.network.utilization);
        fprintf(fid, 'AIWorkers,%d,instances\n', simResults.ai.instances);
        fprintf(fid, 'AIUtilization,%.4f,fraction\n', simResults.ai.utilization);
        fprintf(fid, 'Ophthalmologists,%d,doctors\n', simResults.review.ophthalmologists);
        fprintf(fid, 'ReviewerUtilization,%.4f,fraction\n', simResults.review.utilization);
        fprintf(fid, 'AverageWaitTimeMin,%.2f,minutes\n', simResults.averageWaitTimeMin);
        fprintf(fid, 'MaximumQueueLength,%d,patients\n', simResults.maximumQueueLength);
        fprintf(fid, 'PrimaryBottleneck,"%s",subsystem\n', simResults.systemBottleneck);
        fclose(fid);
        fprintf('Saved district screening summary to: %s\n', csvPath);
    end
    
    % 2. Export district_screening_summary.json
    jsonPath = fullfile(outDir, 'district_screening_summary.json');
    jsonStr = jsonencode(simResults);
    fid = fopen(jsonPath, 'w');
    if fid ~= -1
        fprintf(fid, '%s', jsonStr);
        fclose(fid);
        fprintf('Saved district screening summary JSON to: %s\n', jsonPath);
    end
    
    % Also copy to root results/ if exists
    rootResults = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'results');
    if exist(rootResults, 'dir')
        try
            copyfile(csvPath, fullfile(rootResults, 'district_screening_summary.csv'));
            copyfile(jsonPath, fullfile(rootResults, 'district_screening_summary.json'));
        catch
        end
    end
end
