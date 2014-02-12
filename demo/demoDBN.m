fprintf('\nHere we train a two-layer Deep Belief Network\non the MNIST dataset.');

% LOAD THE MINIST DATASET
load mnistSmall.mat

trainLabels = single(trainLabels);
testLabels = single(testLabels);
[nObs,nIn]= size(trainData);
[~,nOut]=size(trainLabels);

nHid = 256;

% DEFINE MULTI-LAYERED ARCHITECTURE 
arch.size = [nIn nHid nHid];
arch.lRate = [.1 .1];
arch.nEpoch = [200 200];

% GLOBAL OPTIONS
arch.opts = {'inputType', 'binary', ...
			 'classifier',true};

% INITIALIZE NETWORK
d = dbn(arch);

% TRIAN & TEST THE DBN
fprintf('\n\nTraining a two-layer Deep Belief Network...\n');
d = d.train(trainData,trainLabels);

[~,classErr1,misClass] = d.classify(testData,testLabels);
fprintf('\nClassification error: %1.2f%%',classErr1*100);

misClass = testData(misClass,:);
close all
figure;
visWeights(misClass',0,[0 1]); title(sprintf('Missclassified -- Error=%1.2f %%',classErr1*100));
drawnow

fprintf('\n\nNow we fine-tune the DBN using backprop\n')

% USE WEIGHTS TO INITIALIZE NEURAL NETWORK
ftArch.lRate = 0.01;
ftArch.opts = {'costFun','xent', 'nEpoch',30,'sparsity',0.01};

n = d.fineTune(trainData,trainLabels,ftArch);
[~,classErr2] = n.test(testData,testLabels,'classerr');
fprintf('\nFine-tuned classification error: %1.2f%%',classErr2*100);
fprintf('\n...an improvement of %1.2f%%',(classErr1-classErr2)*100);

figure;
nVis = 100;
cLims = [-1.5 1.5];
subplot(131);
visWeights(d.rbmLayers{1}.W(:,1:nVis),cLims);
title('Features Selected from 1st Layer RBM')
drawnow

subplot(132);
visWeights(n.layers{1}.W(1:nVis,:)',cLims);
title('Fine-Tuned First-layer Weights')

subplot(133);
visWeights(d.rbmLayers{1}.W(:,1:nVis)-n.layers{1}.W(1:nVis,:)',cLims);
title('Fine-Tuned First-layer Features')
title('Difference')
