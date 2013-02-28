<<<<<<< HEAD
fprintf('\nHere we train a two-layer Deep Belief Network\non the MNIST dataset.');

% LOAD THE MINIST DATASET
load mnistSmall.mat

trainLabels = single(trainLabels);
testLabels = single(testLabels);
[nObs,nIn]= size(trainData);
[~,nOut]=size(trainLabels);

nHid = 100;

% DEFINE MULTI-LAYERED ARCHITECTURE 
arch.size = [nIn nHid nHid];
arch.lRate = [.1 .1 ];
arch.nEpoch = [20 20];

% GLOBAL OPTIONS
arch.opts = {'inputType', 'binary', ...
			 'classifier',true};

% INITIALIZE NETWORK
d = dbn(arch);

% TRIAN & TEST THE DBN
fprintf('\n\nTraining a two-layer Deep Belief Network...\n');
d = d.train(trainData,trainLabels);

[classErr1,misClass] = d.classify(testData,testLabels);
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
=======
d = dbn(); % STACK A DEFAULT DBN

%  figure(1);
%  for iL = 1:d.nLayers
%  	subplot(1,d.nLayers,iL);
%  	d.layers{iL}.vis(); title(sprintf('Layer %d Weights',iL));drawnow
%  end

% TEST ABILITY TO RECONSTRUCT
% CORRUPT SOME DATA
load('defaultData.mat');
noiseLevel = 0.1;
testDat = testdata(1:100,:);
noiseIdx = rand(size(testDat))>(1-noiseLevel);
noise = rand(size(testDat));
testDat(noiseIdx) = noise(noiseIdx);

dataUp = d.propDataUp(testDat);
dataUp = d.layers{end}.HtoV(dataUp);

nIters = 1;
recon = d.sample(dataUp,20,nIters);

f2 = figure(2); set(f2,'name','Reconstruction of Noisy Input')

subplot(121);
visWeights(testDat',1); title(sprintf('Corrupted Test Data \n(%2.0f%% noise)',100*noiseLevel));

subplot(122);
visWeights(recon{2}',1);
title(sprintf('Reconstruction \n(%d Gibbs Iterations)',nIters));

nSamples = 25;
nIters = 1000;
samps = d.sample([],nSamples,nIters);

f3 = figure(3); set(f3,'name','Draws From Model');
rc = numSubPlots(nSamples);
for iS = 1:nSamples
	subplot(rc(1),rc(2),iS)
	visWeights(samps{iS}',1);
	drawnow
end
>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014
