% ************************************************************************
%                 OBJECTIVE FUNCTION FOR OPTIMISATION
% *************************************************************************

% This script defines the Objective Function that will be used in the
% first Bayesian optimisation procedure for hyperparameter tuning. It builds a
% Neural Network and evaluates its performance on a Holdout set.

function ObjFcn = makeObjFcn(XTrain,YTrain)
    ObjFcn = @valErrorFun;
      function [valError,cons] = valErrorFun(optVars)
          % Create a fitting network
          rng default % Set the seed for reproducibility
          net = patternnet(10, char(optVars.trainFcn));
          % Setup division of data for training, validation, testing
          net.divideParam.trainRatio = 2/3;
          net.divideParam.valRatio = 1/3;
          % Train the network
          net.trainParam.showWindow = false;
          net.trainParam.showCommandLine = false;
          [net,~] = train(net,XTrain,YTrain);
          % Test the network
          YPredicted = net(XTrain);
          valError = perform(net,YTrain,YPredicted);
          cons = [];
      end
  end