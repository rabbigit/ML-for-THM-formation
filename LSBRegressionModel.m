function [trainedModel,validationPredictions, validationRMSE] = LSBRegressionModel(trainingData)
inputTable = trainingData;
predictorNames = {'Cl2DOCmgmg', 'Temperature', 'pH', 'Brugl', 'Timeh', 'SUVA'};
predictors = inputTable(:, predictorNames);
response = inputTable.TTMHsugl;
isCategoricalPredictor = [false, false, false, false, false, false];

% Train a regression model
template = templateTree(...
    'MinLeafSize', 1, ...
    'NumVariablesToSample', 'all');
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'LSBoost', ...
    'NumLearningCycles', 219, ...
    'Learners', template, ...
    'LearnRate', 0.16);

predictorExtractionFcn = @(t) t(:, predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

trainedModel.RequiredVariables = {'Brugl', 'Cl2DOCmgmg', 'SUVA', 'Temperature', 'Timeh', 'pH'};
trainedModel.RegressionEnsemble = regressionEnsemble;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2021a.';
trainedModel.HowToPredict = sprintf('To make predictions on a new table, T, use: \n  yfit = c.predictFcn(T) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nThe table, T, must contain the variables returned by: \n  c.RequiredVariables \nVariable formats (e.g. matrix/vector, datatype) must match the original training data. \nAdditional variables are ignored. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

inputTable = trainingData;
predictorNames = {'Cl2DOCmgmg', 'Temperature', 'pH', 'Brugl', 'Timeh', 'SUVA'};
predictors = inputTable(:, predictorNames);
response = inputTable.TTMHsugl;
isCategoricalPredictor = [false, false, false, false, false, false];

% Perform cross-validation
rng('default')
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 3);

% Compute validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Compute validation RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
% to make To make predictions with the returned 'trainedModel' on new data T(table), use
% yfit = trainedModel.predictFcn(T)
