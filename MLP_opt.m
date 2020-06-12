%% ************************************************************************
%                   MLP HYPERPARAMETER OPTIMISATION
% *************************************************************************

% AIM: carry out experiments to find the optimised hyperparameters for the
% MLP algorithm, aiming to minimise the classification error by varying
% the parameters and comparing the results from Bayesian optimisation to grid search.
% We use 2 objective functions for the 2 Bayes optimisations: makeObjFcn
% and classError. We use gridSearchNN to execute the grid search.

% Clear workspace and command window
clear; clc; close all;

% Load the dataset
data = readtable('winequality-white.csv', 'PreserveVariableNames', true);
data.good_quality = data.quality >= 7;
input = zscore(table2array(data(:, 1:11))); % Standardise the data
target = categorical(data.good_quality);
target_dum = dummyvar(target); % Transform target into dummy variables
m = size(input,1); % Number of rows
n = size(input,2) + 1; % Number of columns

% Split into train and test
P = 0.7 ; % 70-30 split
XTrain = input(1:round(P*m), :);
YTrain = target_dum(1:round(P*m), :);
XTest = input(round(P*m)+1:end, :);
YTest = target_dum(round(P*m)+1:end, :);

% Define a train/validation split
rng default % Set the seed for reproducibility
cv = cvpartition(size(YTrain,1), 'Holdout', 1/3); % Hold-out validation inside objective function

%% OPTIMISATION Pt 1: Use Bayes optimisation to find optimal network depth %%
%------------------------------------------------------------------------%

    % Choose variables to optimise
optimVars = [optimizableVariable('networkDepth', [1, 3], 'Type', 'integer')
             optimizableVariable('trainFcn', {'traingda', 'traingdm', 'traingdx', 'trainscg', 'trainoss'}, 'Type', 'categorical')];

    % Perform Bayesian optimisation
ObjFcn = makeObjFcn(XTrain', YTrain');

BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',30,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false);

% Optimal network depth is 2 and the trainFcn = 'trainscg'

%% OPTIMISATION Pt 2: Bayes optimisation - given that network depth =2 and train function ='trainscg' %%
%----------------------------------------------------------------------------------------------------------%

    % Choose Variables to optimise
vars = [optimizableVariable('Layer1Size', [10 50],'Type','integer')
        optimizableVariable('Layer2Size', [10 50],'Type','integer')
	    optimizableVariable('sig', [4.5e-5 5.5e-5])
        optimizableVariable('lam', [4.5e-7 5.5e-7])
        optimizableVariable('transferFcn', {'logsig', 'poslin', 'tansig', 'purelin'}, 'Type', 'categorical')];

    % Optimise hyperparameters
tic
minfn = @(T)classError(XTrain', YTrain', cv, T.Layer1Size,...
    T.Layer2Size, T.sig, T.lam, T.transferFcn);
results = bayesopt(minfn, vars, 'MaxObj',50, 'IsObjectiveDeterministic', false,...
    'AcquisitionFunctionName', 'expected-improvement-plus');
T = bestPoint(results);
toc

%% Train Bayes optimisation model on full train set
rng default % Set the seed for reproducibility
hiddenLayerSize = [T.Layer1Size T.Layer2Size];
net_BO = patternnet(hiddenLayerSize, 'trainscg');
net_BO.trainParam.sigma = T.sig; % Update sigma
net_BO.trainParam.lambda = T.lam; % Update lambda
net_BO.divideMode = 'none'; % Use all data for training
net_BO.layers{2}.transferFcn = char(T.transferFcn); % Update activation function of layers

tic
[net_BO, tr] = train(net_BO, XTrain', YTrain');
toc
plotperform(tr)

%% Evaluate net_BO model on Test Set
yPred = net_BO(XTest');
tind = vec2ind(YTest');
yPredind = vec2ind(yPred);

error1 = sum(tind ~= yPredind)/numel(tind);
fprintf("The Classification Accuracy of the Bayesian optimisation model is : %.2f%%\n", (1-error1)*100)

%% OPTIMISATION Pt 3: Grid Search - given that network depth =2 and train function ='trainscg' %%
%------------------------------------------------------------------------------------------------%
    % Specify hyperparameters
hidlaysize1 = [10 30 50];
hidlaysize2 = [10 30 50];
sig = [4.5e-5 5e-5 5.5e-5];
lam = [4.5e-7 5e-7 5.5e-7];
transferfunc = {'logsig' 'poslin' 'tansig' 'purelin'};
    
    % Optimise hyperharameters
tic
bestparameters = gridSearchNN(XTrain,YTrain,hidlaysize1, hidlaysize2, sig,...
                 lam, transferfunc);
toc

bestparameters = cell2table(bestparameters);

%% Train grid search model on full train set
rng default % Set the seed for reproducibility
hiddenLayerSize2 = table2array([bestparameters(:,1), bestparameters(:,2)]);
net_grid = patternnet(hiddenLayerSize2, 'trainscg');
net_grid.trainParam.sigma = bestparameters.bestparameters3; % Update sigma
net_grid.trainParam.lambda = bestparameters.bestparameters4; % Update lambda
net_grid.divideMode = 'none'; % Use all data for training
net_grid.layers{2}.transferFcn = char(bestparameters.bestparameters5); % Update activation function of layers

tic
[net_grid, tr2] = train(net_grid, XTrain', YTrain');
toc
plotperform(tr2)

%% Evaluate grid search model on Test Set
yPred2 = net_grid(XTest');
tind = vec2ind(YTest');
yPredind2 = vec2ind(yPred2);

error2 = sum(tind ~= yPredind2)/numel(tind);
fprintf("The Classification Accuracy of the grid search model is : %.2f%%\n", (1-error2)*100)

%% Save best model
if error1<error2
save ('optMLP.mat', 'net_BO')
else
save ('optMLP.mat', 'net_grid')
end

%% Save workspace
save ('MLP_opt_output.mat')