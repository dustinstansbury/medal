fprintf('\nHere we train a Deep autoencoder via\nstochastic gradient descent on the MNIST dataset.\n');

% LOAD THE MINIST DATASET
load mnistSmall.mat

trainLabels = single(trainLabels);
testLabels = single(testLabels);
[nObs,nIn]= size(trainData);
[~,nOut]=size(trainLabels);

nHid = 100;

% DEFINE ARCHITECTURE
arch.size = [nIn nHid nHid nOut];
arch.lRate = [.01 .01 .01];
arch.nEpoch = [50 20 20];

% INITIALIZE DAE
d = dae(arch);

% PRE TRAIN DAE
d = d.train(trainData);

% FINE-TUNE VIA BACKPROP
ftArch.lRate = .025;  % LEARNING RATE FOR ALL LAYERS
ftArch.opts = {'costFun', 'xent', ... % ftArch IS LIKE FOR mlnn.m
               'nEpoch', 20, ...
               'wDecay', .0001};
               
net = d.fineTune(trainData,trainLabels,ftArch);

% TEST
[~,classErr] = net.test(testData,testLabels,'classerr');

% DISPLAY
fprintf('\n\nClassification Error: %1.2f%%',classErr*100);

figure;
cLims = [-1.5 1.5];
subplot(131);
visWeights(d.aLayers{1}.layers{1}.W',0,cLims);
title('Encoding weights of 1st layer autoencoder');

subplot(132);
visWeights(net.layers{1}.W',0,cLims);
title('Fine-tuned weights of 1st layer autoencoder');

subplot(133);
visWeights(d.aLayers{1}.layers{1}.W'-net.layers{1}.W',0,cLims);
title('Difference');
