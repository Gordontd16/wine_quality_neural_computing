%% ************************************************************************
%                         SVM HYPERPARAMETER OPTIMISATION
% *************************************************************************

% AIM: carry out experiments to find the optimised hyperparameters for the
% SVM algorithm, aiming to minimise the cross-validation loss (error) by varying
% the parameters and comparing the results from Bayesian optimisation to grid search. 

% Clear workspace and command window
clear; clc; close all;

% Load the dataset
data = readtable('winequality-white.csv', 'PreserveVariableNames', true);
data.good_quality = data.quality >= 7;
input = zscore(table2array(data(:, 1:11))); % Standardise the data
target = categorical(data.good_quality);
m = size(input,1); % Number of rows
n = size(input,2) + 1; % Number of columns

% Split into train and test
P = 0.7 ; % 70-30 split
XTrain = input(1:round(P*m), :);
YTrain = target(1:round(P*m), :);
XTest = input(round(P*m)+1:end, :);
YTest = target(round(P*m)+1:end, :);

% Define a train/validation split
rng default % Set the seed for reproducibility
cv = cvpartition(size(YTrain,1), 'Holdout', 1/3);

%% OPTIMISATION Pt 1: optimise kernel function %%
%---------------------------------------------------%
    % Bayes Optimisation %
rng default % Set the seed for reproducibility
SVM_BO1 = fitcsvm(XTrain,YTrain, 'OptimizeHyperparameters',{'KernelFunction'},... 
'HyperparameterOptimizationOptions', struct('cvpartition', cv,...
'AcquisitionFunctionName','expected-improvement-plus'));

    %Grid Search % 
rng default % Set the seed for reproducibility
SVM_grid1 = fitcsvm(XTrain,YTrain, 'OptimizeHyperparameters', {'KernelFunction'},...
'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch',...
'cvpartition', cv, 'AcquisitionFunctionName','expected-improvement-plus'));

% These experiments confirm that the gaussian kernel function is the best
% kernel function for our dataset.

%% OPTIMISATION Pt 2: given that the best kernel function is Gaussian,...
% optimise the Box Constraint and the Kernel Scale
%-------------------------------------------------------------------%
    % Optimise SVM using Bayesian Optimisation
rng default % Set the seed for reproducibility
SVM_BO2 = fitcsvm(XTrain,YTrain, 'KernelFunction','rbf', 'OptimizeHyperparameters',{'BoxConstraint','KernelScale'},... 
'HyperparameterOptimizationOptions', struct('cvpartition', cv,...
'AcquisitionFunctionName','expected-improvement-plus'));
MinEstObj1 = SVM_BO2.HyperparameterOptimizationResults.MinEstimatedObjective;

    % Optimise SVM with grid search
rng default % Set the seed for reproducibility
SVM_grid2 = fitcsvm(XTrain,YTrain, 'KernelFunction','rbf', 'OptimizeHyperparameters', {'BoxConstraint','KernelScale'},...
'HyperparameterOptimizationOptions', struct('Optimizer', 'gridsearch',...
'cvpartition', cv, 'AcquisitionFunctionName','expected-improvement-plus'));
tbl = SVM_grid2.HyperparameterOptimizationResults;
MinEstObj2 = tbl(tbl.Rank==1, :).Objective;

fprintf('Bayesian Optimisation minimises the objective function to %4.4f vs grid search which achieves a value of %4.4f.\n\n', MinEstObj1, MinEstObj2)

%% Comparison of models post-optimisation
%---------------------------------------------------%
    % Optimised SVM using Bayes optimisation
tic
SVM_BO3 = fitcsvm(XTrain, YTrain, 'KernelFunction','rbf', 'BoxConstraint', SVM_BO2.ModelParameters.BoxConstraint,...
            'KernelScale', SVM_BO2.ModelParameters.KernelScale);
toc
ClassSVM1 = predict(SVM_BO3, XTest);
acc1 = sum(ClassSVM1==YTest)/length(ClassSVM1);

    % Optimised SVM using grid search 
tic
SVM_grid3 = fitcsvm(XTrain, YTrain, 'KernelFunction','rbf', 'BoxConstraint', SVM_grid2.ModelParameters.BoxConstraint,...
            'KernelScale', SVM_grid2.ModelParameters.KernelScale);
toc
ClassSVM2 = predict(SVM_grid3, XTest);
acc2 = sum(ClassSVM2==YTest)/length(ClassSVM2);

fprintf('The Bayes optimisation SVM model has a test accuracy of %4.4f and the grid search model achieves %4.4f.\n\n', acc1, acc2)

%% Save best model
if acc1>acc2
save ('optSVM.mat', 'SVM_BO3')
else
save ('optSVM.mat', 'SVM_grid3')
end

%% Save workspace
save ('SVM_opt_output.mat')