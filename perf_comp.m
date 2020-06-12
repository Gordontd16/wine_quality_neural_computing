% ************************************************************************
%                       PERFORMANCE OPTIMISATION
% *************************************************************************

% AIM: Compare the performance of SVM and MLP in terms of predictive power
% and prediction time.

% Clear workspace and command window
clear; clc; close all;

% Load the dataset
data = readtable('winequality-white.csv', 'PreserveVariableNames', true);
data.good_quality = data.quality >= 7;
input = zscore(table2array(data(:, 1:11))); % Standardise the data
target = data.good_quality;
target_dum = dummyvar(categorical(target)); % Transform target into dummy variables
m = size(input,1); % Number of rows
n = size(input,2) + 1; % Number of columns

% Load optimised models
load optSVM
load optMLP
SVM_best = SVM_grid3;
net_best = net_BO;

% Split into train and test
P = 0.7 ; % 70-30 split
XTrain = input(1:round(P*m), :);
YTrain1 = target(1:round(P*m), :);
YTrain2 = target_dum(1:round(P*m), :);
XTest = input(round(P*m)+1:end, :);
YTest1 = target(round(P*m)+1:end, :);
YTest2 = target_dum(round(P*m)+1:end, :);

%% SVM Model Predictions & Confusion Matrix
%---------------------------------------------------%
[ClassSVM, ScoreSVM] = predict(SVM_best, XTest);
figure('pos',[10 1000 500 400]);
cmSVM = confusionchart(YTest1,ClassSVM);
    % ClassSVM are the predictions of the SVM model and YTest1 are the true values
tnSVM = sum((ClassSVM == 0) & (YTest1 == 0));
tpSVM = sum((ClassSVM == 1) & (YTest1 == 1));
fpSVM = sum((ClassSVM == 1) & (YTest1 == 0));
fnSVM = sum((ClassSVM == 0) & (YTest1 == 1));
    % Accuracy for SVM
accSVM = (tnSVM + tpSVM) / (tnSVM+tpSVM+fpSVM+fnSVM);
SVM_testScore = sum(ClassSVM==YTest1)/length(ClassSVM); % sense checking accSVM
    % Calculating F1 Score for SVM
precisionSVM = tpSVM / (tpSVM + fpSVM);
recallSVM = tpSVM / (tpSVM + fnSVM);
F1SVM = (2 * precisionSVM * recallSVM) / (precisionSVM + recallSVM);

%% MLP Model Predictions & Confusion Matrix
%---------------------------------------------------%
MLP_pred_test = net_best(XTest');
target_test_ind = vec2ind(YTest2');
MLP_pred_test_ind = vec2ind(MLP_pred_test);
figure('pos',[1000 1000 500 400]);
cmMLP = confusionchart(target_test_ind, MLP_pred_test_ind);
    % ClassMLP are the predictions of the MLP model and target_test_ind are the true values
tnMLP = sum((MLP_pred_test_ind == 1) & (target_test_ind == 1));
tpMLP = sum((MLP_pred_test_ind == 2) & (target_test_ind == 2));
fpMLP = sum((MLP_pred_test_ind == 2) & (target_test_ind == 1));
fnMLP = sum((MLP_pred_test_ind == 1) & (target_test_ind == 2));
    % Accuracy for MLP
accMLP = (tnMLP + tpMLP) / (tnMLP+tpMLP+fpMLP+fnMLP);
MLP_testScore = sum(target_test_ind == MLP_pred_test_ind)/numel(target_test_ind); % sense checking accMLP
    % Calculating F1 Score for MLP
precisionMLP = tpMLP / (tpMLP + fpMLP);
recallMLP = tpMLP / (tpMLP + fnMLP);
F1MLP = (2 * precisionMLP * recallMLP) / (precisionMLP + recallMLP);

%% ROC Curve
[XSVM, YSVM, TSVM, AUCSVM] = perfcurve(YTest1,ScoreSVM(:,2),'true');
MLP_pred_trans = transpose(MLP_pred_test);
[XMLP, YMLP, TMLP, AUCMLP] = perfcurve(YTest1, MLP_pred_trans(:,2),'true');

%% Model Prediction Time Performance
SVMpredHandle = @() predict(SVM_best, XTest);
MLPpredHandle = @() net_best(XTest');

timeSVMpred = timeit(SVMpredHandle);
timeMLPpred = timeit(MLPpredHandle);

%% Printing Performance
    % Prediction performance
fprintf('AUC Prediction Performance\n')
fprintf('SVM : %.3f\n', AUCSVM)
fprintf('MLP : %.3f\n\n', AUCMLP)
    % Time performance
fprintf('Prediction Time Performance\n')
fprintf('SVM : %.3f\n', timeSVMpred)
fprintf('MLP : %.3f\n', timeMLPpred)

%% Visualise ROC Curve
figure
plot(XSVM, YSVM)
hold on
plot(XMLP, YMLP)
xlabel('False Positive Rate'); ylabel('True Positive Rate');
legend( 'SVM', 'MLP')

%% Save workspace
save ('perf_comp_output.mat')