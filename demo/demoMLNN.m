<<<<<<< HEAD
fprintf('\nHere we train single and multi-layer Neural Networks via\nstochastic gradient descent on the MNIST dataset.\n');
=======
% In this demo, we train a single-layer neural net, and a multi-layer neural net
% to classify the MNIST hand-written digit dataset. The first neural net has L2
% regularization, and the second has L1-regularization (which should induce
% sparseness in learned features).

>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014

% LOAD THE MINIST DATASET
load mnistSmall.mat

<<<<<<< HEAD
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
arch2.lRate = [.1 .1 .1];
arch2.opts = {'nEpoch',20, ...
              'costFun','xent'};%, ...
%                'visFun', @visMLNNLearning};

% INITIALIZE NETWORK
net2 = mlnn(arch2);

% TRIAN & TEST
fprintf('\n\nTraining a two-hidden-layer neural net...\n');
net2 = net2.train(trainData,trainLabels);
[~,classErr2] = net2.test(testData,testLabels,'classerr');

% DISPLAY
fprintf('\n\nClassification Error For Single-layered Network: %1.2f%%',classErr1*100);
fprintf('\nClassification Error For Multi-layerd Network: %1.2f%%\n',classErr2*100);
=======
trainData = data(1:10000, :);
trainLabels = labels(1:10000, :);

testData = data(10001:end, :);
testLabels = labels(10001:end, :);

clear data labels;

fprintf('\nTraining a single-layer neural net...\n');
% CREATE A SINGLE-LAYER PERCEPTRON -- In(784) -> Hid(50) -> Out(10)
args = {'nHid', [50], ...  	% ONE HIDDEN LAYER, WITH 100 UNITS
        'actFun',{'tanh','sigmoid'} ...% TANH HIDDEN ACTIVATION FUNCTION/SIGMOID OUTPUT NONLINEARITY
        'weightDecay',1e-4, ...	% L2-WEIGHT DECAY/REGULARIZATION
        'lambda',[.05], ...		% LEARNING RATE
        'nEpoch',20, ...		% # TRAINING ITERATIONS
        'batchSize',10};		% MINIBATCH SIZE FOR TRAINING DATA

% TRAIN THE NET
net1 = mlnn(args,trainData,trainLabels);

% TEST THE NET
[testPred,testErrors] = net1.test(testData,testLabels,'classError');

fprintf('Test classification error for single-layer net: %g%%\n',100*mean(testErrors));

figure(11);
missClassIdx = find(testErrors);
visWeights(testData(missClassIdx,:)'); title('Misclassified');


% CREATE A MULTI-LAYER NEURAL NET -- In(784) -> Hid(50) -> Hid(50) -> Out(10)
fprintf('\nNow training a multi-layer neural net...\n');

args = {'nHid', [50,50], ...	% TWO HIDDEN LAYERS, EACH WITH 50 UNITS
        'actFun',{'tanh','tanh','sigmoid'} ... % TANH HIDDEN/SIGMOID OUTPUT
        'weightDecay',-1e-4, ...% L1-WEIGHT DECAY/REGULARIZATION
        'lambda',[.1,.01], ...	% DIFF. LEARNING RATES FOR EACH LAYER
        'nEpoch',20 ...			% # TRAINING ITERATIONS
        'batchSize',10};		% MINIBATCH SIZE FOR TRAINING DATA

net2 = mlnn(args,trainData,trainLabels);

% TEST THE NET
[testPred,testErrors] = net2.test(testData,testLabels,'classError');

fprintf('Test classification error for multi-layer net: %g%%\n',100*mean(testErrors));

figure(22);
missClassIdx = find(testErrors);
visWeights(testData(missClassIdx,:)'); title('Misclassified');
>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014
