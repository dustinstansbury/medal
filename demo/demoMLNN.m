% In this demo, we train a single-layer neural net, and a multi-layer neural net
% to classify the MNIST hand-written digit dataset. The first neural net has L2
% regularization, and the second has L1-regularization (which should induce
% sparseness in learned features).


% LOAD THE MINIST DATASET
load mnistSmall.mat

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
