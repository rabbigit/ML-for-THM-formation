function [trainedModel,validationPredictions, validationRMSE] = LSBModel_without_SUVA(trainingData)
inputTable = trainingData;
predictorNames = {'Cl2_ratio_DOC', 'Temperature', 'pH', 'Br_Con', 'Reaction_time'};
predictors = inputTable(:, predictorNames);
response = inputTable.TTHMs_Con;
isCategoricalPredictor = [false, false, false, false, false];

% Train a regression model
template = templateTree(...
    'MinLeafSize', 2, ...
    'NumVariablesToSample', 'all');
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 324, ...
    'Learners', template, ...
    'LearnRate', 0.15);

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));
trainedModel.RequiredVariables = {'Br_Con', 'Cl2_ratio_DOC', 'Reaction_time', 'Temperature', 'pH'};
trainedModel.RegressionEnsemble = regressionEnsemble;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2021a.';
trainedModel.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

inputTable = trainingData;
predictorNames = {'Cl2_ratio_DOC', 'Temperature', 'pH', 'Br_Con', 'Reaction_time'};
predictors = inputTable(:, predictorNames);
response = inputTable.TTHMs_Con;
isCategoricalPredictor = [false, false, false, false, false];

% Perform cross-validation
rng('default')
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 10);

% Compute validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Compute validation RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
