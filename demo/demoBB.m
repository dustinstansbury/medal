fprintf('\nHere we train an RBM with Bernouilli visible and hidden units on\n')
fprintf('\nMNIST datastet.\n');
load('defaultData.mat','testdata');

args = {'type', 'BB', ...
		'nHid', 50, ...
		'verbose', 1, ...
		'eta', 0.1, ...
		'momentum', 0.5, ...
		'nEpoch', 20, ...
		'wDecay', 0.02, ...
		'batchSz', 12, ...
		'anneal', 0, ...
		'nGibbs', 1, ...
		'varyEta',1};
		
r = rbm(args);	
r = r.train();	% TRAIN RMB USING CD[1]

noiseLevel = 0.1;
testDat = testdata(1:100,:);
noiseIdx = rand(size(testDat))>(1-noiseLevel);
noise = rand(size(testDat));
testDat(noiseIdx) = noise(noiseIdx);
nIters = 1;
r.verbose = 0;
recon = r.sample(testDat,1,nIters);

f2 = figure(2); set(f2,'name','Reconstruction of Noisy Input')

subplot(121);
visWeights(testDat',1); title(sprintf('Corrupted Test Data \n(%2.0f%% noise)',100*noiseLevel));

subplot(122);
visWeights(recon',1); 
title(sprintf('Reconstruction \n(%d Gibbs Iterations)',nIters));

drawnow
fprintf('\nDrawing samples from the model...')
r.verbose = 1;
samps = r.sample(binornd(1,.5,1,r.nVis),24,1000);
f3 = figure(3); set(f3,'name','Samples From Model')
figure(f3); visWeights(squeeze(samps),1);

