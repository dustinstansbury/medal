
args = {'type','BG', ...
		'eta',.001, ...
		'batchSz',64, ...
		'nEpoch',100, ...
		'nHid',100, ...
		'learnSigma2',1 ...
		'sampleVis',1, ...
		'sparse',0.001};

clear r;  r = rbm(args);r = r.train;

load('defaultData.mat');
noiseLevel = 0.1;
testDat = testdata(1:100,:);
noiseIdx = rand(size(testDat))>(1-noiseLevel);
noise = rand(size(testDat));
testDat(noiseIdx) = noise(noiseIdx);

nIters = 3;
recon = r.sample(testDat,nIters);

figure(2)
subplot(221);
plot(r.e); axis square; title(sprintf('Reconstruction error \nCD[%d]',r.nGibbs)); xlabel('Iteration #')
set(gca,'fontsize',8);

subplot(222);
Wbb = r.vis(); title('Learned Feature Weights');

subplot(223);
visWeights(testDat',1,[0 1]); title(sprintf('Corrupted Test Data \n(%2.0f%% noise)',100*noiseLevel));

subplot(224);
visWeights(recon',1,[0 1]); title(sprintf('Reconstruction \n(%d Gibbs Iterations)',nIters));