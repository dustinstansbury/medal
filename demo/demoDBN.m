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