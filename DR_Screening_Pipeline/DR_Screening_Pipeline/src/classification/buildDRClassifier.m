function net = buildDRClassifier(baseNetName, inputSize)
%BUILDDRCLASSIFIER  Build a transfer-learning CNN for 5-class ICDR grading.
%
%   net = BUILDDRCLASSIFIER(baseNetName, inputSize)
%
%   baseNetName - 'resnet50' (default) | 'efficientnetb0' | 'densenet201'
%                 (any network available via Deep Learning Toolbox Add-Ons)
%   inputSize   - e.g. [224 224 3]; defaults to the base network's native size
%
%   Returns an uninitialized (untrained) layerGraph-based network with the
%   final classification head replaced for 5 ICDR classes:
%       0 = No DR, 1 = Mild NPDR, 2 = Moderate NPDR,
%       3 = Severe NPDR, 4 = Proliferative DR (PDR)
%
%   Rationale for transfer learning over training from scratch: rural
%   screening datasets are small relative to ImageNet-scale corpora;
%   fine-tuning a pretrained backbone converges faster and generalizes
%   better on limited labeled fundus data (well established in DR-AI
%   literature, e.g. the EyePACS/Kaggle DR competition top solutions).

    if nargin < 1 || isempty(baseNetName)
        baseNetName = 'resnet50';
    end

    classNames = {'0_NoDR', '1_Mild', '2_Moderate', '3_Severe', '4_Proliferative'};
    numClasses = numel(classNames);

    switch lower(baseNetName)
        case 'resnet50'
            base = resnet50;
            if nargin < 2 || isempty(inputSize), inputSize = [224 224 3]; end
            lgraph = layerGraph(base);
            lgraph = replaceLayer(lgraph, 'fc1000', ...
                fullyConnectedLayer(numClasses, 'Name', 'fc_dr', ...
                    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));
            lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', ...
                classificationLayer('Name', 'classoutput', 'Classes', classNames));
            lgraph = replaceLayer(lgraph, 'fc1000_softmax', softmaxLayer('Name', 'softmax_dr'));

        case 'efficientnetb0'
            base = efficientnetb0;
            if nargin < 2 || isempty(inputSize), inputSize = [224 224 3]; end
            lgraph = layerGraph(base);
            lgraph = replaceLayer(lgraph, 'efficientnetb0_top_conv', ...
                fullyConnectedLayer(numClasses, 'Name', 'fc_dr'));
            lgraph = replaceLayer(lgraph, 'Logits_softmax', softmaxLayer('Name', 'softmax_dr'));
            lgraph = replaceLayer(lgraph, 'classification', ...
                classificationLayer('Name', 'classoutput', 'Classes', classNames));

        case 'densenet201'
            base = densenet201;
            if nargin < 2 || isempty(inputSize), inputSize = [224 224 3]; end
            lgraph = layerGraph(base);
            lgraph = replaceLayer(lgraph, 'fc1000', ...
                fullyConnectedLayer(numClasses, 'Name', 'fc_dr', ...
                    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));
            lgraph = replaceLayer(lgraph, 'fc1000_softmax', softmaxLayer('Name', 'softmax_dr'));
            lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', ...
                classificationLayer('Name', 'classoutput', 'Classes', classNames));

        otherwise
            error('buildDRClassifier:unknownBase', ...
                'Unsupported base network "%s". Use resnet50, efficientnetb0, or densenet201.', baseNetName);
    end

    net = lgraph;
    net.UserData = struct('classNames', {classNames}, 'inputSize', inputSize, 'baseNetName', baseNetName);
end
