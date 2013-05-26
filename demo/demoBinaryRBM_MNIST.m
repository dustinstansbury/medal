fprintf('\nHere we train an RBM with Binary inputs (MNIST datastet).\n');

% LOAD DATASET
load('mnistSmall.mat');

[nObs,nVis] = size(trainData);

nHid = 500; % 500 HIDDEN UNITS

% DEFINE A MODEL ARCHITECTURE
arch = struct('size', [nVis,nHid], 'classifier',true, 'inputType','binary');

% GLOBAL OPTIONS
arch.opts = {'verbose', 1, ...
		 'lRate', 0.1, ...
		'momentum', 0.5, ...
		'nEpoch', 10, ...
		'wPenalty', 0.02, ...
		'batchSz', 100, ...
		'beginAnneal', 10, ...
		'nGibbs', 1, ...
		'sparsity', .01, ...
		'varyEta',7, ...
		'displayEvery', 20};
%  		'visFun', @visBinaryRBMLearning};

% INITIALIZE RBM
r = rbm(arch);

% TRAIN THE RBM
r = r.train(trainData,single(trainLabels));

[~,classErr,misClass] = r.classify(testData, single(testLabels));


misClass = testData(misClass,:);
clf; visWeights(misClass',0,[0 1]); title(sprintf('Missclassified -- Error=%1.2f %%',classErr*100));

nVis = 100;
figure; visWeights(r.W(:,1:nVis));
title('Sample of RBM Features');