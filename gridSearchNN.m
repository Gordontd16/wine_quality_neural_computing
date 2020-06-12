% ************************************************************************
%                 GRID SEARCH OPTIMISATION FUNCTION
% *************************************************************************
             
function out = gridSearchNN(trainX,trainY,param1,param2,param3,param4, param5,...
                            varargin)                         
                          
if(nargin > 4)
[p,q,r,s, t] = ndgrid(param1,param2, param3, param4, 1:length(param5));
pairs = [p(:) q(:) r(:) s(:) t(:)];
else
[p,q] = meshgrid(param1,param2);
pairs = [p(:) q(:)];
end

valscores = zeros(size(pairs,1),1);

for i=1:size(pairs,1)
  rng default % Set the seed for reproducibility
  net = patternnet([pairs(i,1) pairs(i,2)]); % Hidden neurons
  net.trainParam.sigma	= pairs(i,3); % Sigma
  net.trainParam.lambda = pairs(i,4); % Lambda
  net.trainFcn = 'trainscg'; % Training function
  net.layers{2}.transferFcn = param5{pairs(i,5)}; % Activation function
  net.trainParam.epochs = 500; % Specify number of epochs
  net.trainParam.max_fail = 6; % Early stopping after 6 consecutive increases of validation performance
  net.divideParam.trainRatio = 2/3;
  net.divideParam.valRatio = 1/3;
  
   vals = crossval(@(XTRAIN, YTRAIN, XTEST, YTEST)NNtrain(XTRAIN, YTRAIN, XTEST, YTEST, net),...
                     trainX, trainY);
   valscores(i) = mean(vals);  

end

 [~,ind] = max(valscores);
 out = {pairs(ind,1) pairs(ind,2) pairs(ind,3) pairs(ind,4) param5{pairs(ind,5)}};


end

function testval = NNtrain(XTRAIN, YTRAIN, XTEST, YTEST, net)

    net = train(net, XTRAIN', YTRAIN');
    y_pred = net(XTEST');
    [~,indicesReal] = max(YTEST',[],1);
    [~,indicesPredicted] = max(y_pred,[],1);
     testval = mean(double(indicesPredicted == indicesReal));      
end