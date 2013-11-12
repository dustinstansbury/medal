fprintf('\nHere we train single and multi-layer Neural Networks via\nstochastic gradient descent on the MNIST dataset.\n');

% LOAD THE MINIST DATASET
load mnistSmall.mat

trainLabels = single(trainLabels);
testLabels = single(testLabels);
[nObs,nIn]= size(trainData);
[~,nOut]=size(trainLabels);

nHid = 100;

% DEFINE SINGLE-LAYERED ARCHITECTURE
arch.size = [nIn nHid nOut];
arch.actFun = {'sigmoid'}; % ALL SIGMOIDS
arch.lRate = [.1 .01];
arch.opts = {'costFun','xent'};%, ...
%               'visFun', @visMLNNLearning};

% INITIALIZE NETWORK
net1 = mlnn(arch);

% TRAIN & TEST
fprintf('\nTraining a single-hidden-layer neural net...\n');
net1 = net1.train(trainData,trainLabels);
[~,classErr1] = net1.test(testData,testLabels,'classerr');

% DEFINE MULTI-LAYERED ARCHITECTURE 
arch2 = arch;
arch2.size = [nIn nHid nHid nOut];
arch2.lRate = [.05 .05 .05];
arch2.opts = {'nEpoch',10, ...
              'costFun','xent'};% ...
              % 'displayEvery', 1000, ...
               % 'visFun', @visMLNNLearning};

% INITIALIZE NETWORK
net2 = mlnn(arch2);

% TRIAN & TEST
fprintf('\n\nTraining a two-hidden-layer neural net...\n');
net2 = net2.train(trainData,trainLabels);
[~,classErr2] = net2.test(testData,testLabels,'classerr');

% DISPLAY
fprintf('\n\nClassification Error For Single-layered Network: %1.2f%%',classErr1*100);
fprintf('\nClassification Error For Multi-layerd Network: %1.2f%%\n',classErr2*100);