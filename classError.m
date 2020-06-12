% ************************************************************************
%                 OBJECTIVE FUNCTION OF BAYESIAN OPTIMISATION
% *************************************************************************

% This script defines the Objective Function that will be used in the
% second Bayesian optimisation procedure for hyperparameter tuning. It builds a
% Neural Network and evaluates its performance on a Holdout set.

% Define cross-validation loss function
function classifError = classError(x, t, cv, Layer1Size, Layer2Size, sig, lam, transferFcn)

% Build network architecture
net = patternnet([Layer1Size Layer2Size], 'trainscg'); % Build network
net.trainParam.epochs = 500; % Specify number of epochs
net.trainParam.max_fail = 6; % Early stopping after 6 consecutive increases of validation performance
net.trainParam.sigma = sig;
net.TrainParam.lambda = lam;
net.layers{2}.transferFcn = char(transferFcn); % Update activation function of layers

% Divide training data into train-validation sets
rng = 1:cv.NumObservations;
net.divideFcn = 'divideind';
net.divideParam.trainInd = rng(cv.training);
net.divideParam.valInd = rng(cv.test);

% Train network
net = train(net, x, t);

% Evaluate on validation set and compute classification error
y = net(x);
tind = vec2ind(t);
yind = vec2ind(y);
classifError = sum(tind(cv.test) ~= yind(cv.test))/numel(tind(cv.test));
% classifError is the value that we are trying to minimise

end