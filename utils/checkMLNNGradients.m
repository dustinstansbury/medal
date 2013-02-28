function net = checkMLNNGradients

nIn = 5;
nOut = 2;
nHid = 5;
nObs = 100;
arch.size = [nIn nHid nOut];
arch.actFun = {'sigmoid'}; % ALL SIGMOIDS

net = mlnn(arch);
net.checkGradients = 1;

data = randn(nObs,nIn);

% CHECK MEAN-SQUARED ERROR COST GRADIENTS
net.costFun = 'mse';
fprintf('\n**MSE Gradients**:\n');
targets = randn(nObs,nOut);
checkNNGradients(net,data,targets);

% CHECK CROSS-ENTROPY COST GRADIENTS
net.costFun = 'xent';
fprintf('\n**Cross-entropy Gradients**:\n');
targets = rand(nObs,nOut) > .5;
checkNNGradients(net,data,targets);

% CHECK MEAN-SQUARED ERROR COST GRADIENTS
% USING HIDDEN UNIT DROPOUT
net.dropout = .5;
fprintf('\n**MSE Gradients With Dropout**:\n');
targets = randn(nObs,nOut);
checkNNGradients(net,data,targets);

% CHECK CROSS-ENTROPY COST GRADIENTS
% USING HIDDEN UNIT DROPOUT
net.costFun = 'xent';
fprintf('\n**Cross-entropy Gradients With Dropout**:\n');
targets = rand(nObs,nOut) > .5;
checkNNGradients(net,data,targets);

% CHECK CROSS-ENTROPY COST GRADIENTS
% USING HIDDEN UNIT DROPOUT
net.sparsity = .1;
net.dropout = 0;
fprintf('\n**Cross-entropy Gradients With Sparsity Target**:\n');
targets = rand(nObs,nOut) > .5;
checkNNGradients(net,data,targets);
