function modelPath = buildScreeningWorkflowModel(params)
%BUILDSCREENINGWORKFLOWMODEL  Programmatically build the telemedicine
%   screening-workflow Simulink model (ScreeningWorkflow.slx).
%
%   modelPath = BUILDSCREENINGWORKFLOWMODEL(params)
%
%   MUST be run inside MATLAB with Simulink installed - it cannot be
%   pre-built as a static file outside MATLAB, since .slx is a MATLAB-
%   specific binary model format. Running this script creates and saves
%   ScreeningWorkflow.slx in the current folder, which you can then open,
%   inspect, and extend directly in the Simulink editor (e.g. swap the
%   simplified rate-based blocks for full SimEvents Entity Generator /
%   Queue / Server blocks for stochastic discrete-event simulation).
%
%   Model structure built:
%
%     [Patient Arrival Rate] -> [Upload/Bandwidth Delay] -> [AI Processing
%     Delay] -> [Referral Switch] -> [Review Queue Delay] -> [Scope: Cases
%     Cleared vs. Backlog vs. Wait Time]
%
%   Each stage is a Transfer Fcn / Rate Limiter / Saturation combo
%   parameterized from capacityPlanningAnalysis.m outputs, giving a
%   continuous-time approximation of the discrete-event queueing system -
%   fast to simulate and easy to read in a review meeting; swap in
%   SimEvents for a higher-fidelity stochastic version if needed.
%
%   params (optional) - struct from capacityPlanningAnalysis output, or
%   [] to use built-in defaults matching a 100,000-patients/year program.

    if nargin < 1 || isempty(params)
        params = capacityPlanningAnalysis(); %#ok<NASGU> % also prints the sizing report
        params = struct('lambdaPerHour', 100000/(300*8), ...
                         'uploadServers', 4, 'aiServers', 3, 'reviewServers', 5, ...
                         'uploadSec', 5, 'aiSec', 8, 'reviewSec', 30);
    end

    modelName = 'ScreeningWorkflow';
    close_system(modelName, 0); %#ok<*TRYNC> % harmless if not open
    try %#ok<TRYNC>
    end
    new_system(modelName);
    open_system(modelName);

    % --- Patient arrival source ---
    add_block('simulink/Sources/Pulse Generator', [modelName '/PatientArrivals'], ...
        'Position', [30 100 90 130], ...
        'Period', num2str(3600/params.lambdaPerHour), ... % one "pulse" per average inter-arrival
        'PulseWidth', '10', 'Amplitude', '1');

    % --- Upload/bandwidth stage: transport delay + saturation representing
    %     channel capacity limit ---
    add_block('simulink/Continuous/Transport Delay', [modelName '/UploadDelay'], ...
        'Position', [160 90 260 140], 'DelayTime', num2str(getFieldOr(params,'uploadSec',5)));
    add_block('simulink/Discontinuities/Saturation', [modelName '/UploadCapacity'], ...
        'Position', [290 90 340 140], 'UpperLimit', num2str(getFieldOr(params,'uploadServers',4)), ...
        'LowerLimit', '0');

    % --- AI processing stage ---
    add_block('simulink/Continuous/Transport Delay', [modelName '/AIProcessingDelay'], ...
        'Position', [380 90 480 140], 'DelayTime', num2str(getFieldOr(params,'aiSec',8)));
    add_block('simulink/Discontinuities/Saturation', [modelName '/AIThroughputCapacity'], ...
        'Position', [510 90 560 140], 'UpperLimit', num2str(getFieldOr(params,'aiServers',3)), ...
        'LowerLimit', '0');

    % --- Referral switch: only ~18% of cases (ReferableFraction) proceed
    %     to human review; the rest go straight to "cleared" ---
    add_block('simulink/Signal Routing/Switch', [modelName '/ReferralSwitch'], ...
        'Position', [600 60 650 200]);
    add_block('simulink/Sources/Constant', [modelName '/ReferableFraction'], ...
        'Position', [520 20 570 50], 'Value', '0.18');

    % --- Review queue stage ---
    add_block('simulink/Continuous/Transport Delay', [modelName '/ReviewDelay'], ...
        'Position', [690 40 790 90], 'DelayTime', num2str(getFieldOr(params,'reviewSec',30)));
    add_block('simulink/Discontinuities/Saturation', [modelName '/ReviewerCapacity'], ...
        'Position', [820 40 870 90], 'UpperLimit', num2str(getFieldOr(params,'reviewServers',5)), ...
        'LowerLimit', '0');

    % --- Outcomes ---
    add_block('simulink/Sinks/Scope', [modelName '/CasesClearedScope'], 'Position', [690 170 750 210]);
    add_block('simulink/Sinks/Scope', [modelName '/ReferredCasesScope'], 'Position', [910 40 970 90]);
    add_block('simulink/Sinks/To Workspace', [modelName '/BacklogLog'], ...
        'Position', [910 170 990 210], 'VariableName', 'backlogLog');

    % --- Wiring ---
    add_line(modelName, 'PatientArrivals/1', 'UploadDelay/1', 'autorouting','on');
    add_line(modelName, 'UploadDelay/1', 'UploadCapacity/1', 'autorouting','on');
    add_line(modelName, 'UploadCapacity/1', 'AIProcessingDelay/1', 'autorouting','on');
    add_line(modelName, 'AIProcessingDelay/1', 'AIThroughputCapacity/1', 'autorouting','on');
    add_line(modelName, 'AIThroughputCapacity/1', 'ReferralSwitch/2', 'autorouting','on');
    add_line(modelName, 'ReferableFraction/1', 'ReferralSwitch/1', 'autorouting','on');
    add_line(modelName, 'ReferralSwitch/1', 'ReviewDelay/1', 'autorouting','on');
    add_line(modelName, 'ReviewDelay/1', 'ReviewerCapacity/1', 'autorouting','on');
    add_line(modelName, 'ReviewerCapacity/1', 'ReferredCasesScope/1', 'autorouting','on');
    add_line(modelName, 'AIThroughputCapacity/1', 'CasesClearedScope/1', 'autorouting','on');
    add_line(modelName, 'ReviewerCapacity/1', 'BacklogLog/1', 'autorouting','on');

    set_param(modelName, 'StopTime', '28800'); % simulate one 8-hour screening day (sec)

    modelPath = fullfile(pwd, [modelName '.slx']);
    save_system(modelName, modelPath);
    fprintf('Simulink model saved to: %s\n', modelPath);
    fprintf(['Open it in MATLAB (open_system(''%s'')) to inspect, run, or\n' ...
             'extend with SimEvents Entity Generator/Queue/Server blocks for a\n' ...
             'full stochastic discrete-event version of the same workflow.\n'], modelName);
end

function v = getFieldOr(s, f, default)
    if isstruct(s) && isfield(s, f)
        v = s.(f);
    else
        v = default;
    end
end
