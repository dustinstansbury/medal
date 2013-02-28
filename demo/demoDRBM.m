fprintf('\nHere we train a Dynamic RBM via Contrastive Divergence \non a toy spatio-temporal dataset.\n');
load('toySpatioTemporal.mat','data');



[nY,nX,nFrames] = size(data);

fprintf('\nDisplaying the toy dataset...\n')
for iF = 1:2:floor(nFrames/20);
	imagesc(data(:,:,iF)); colormap gray; axis image; axis off;
	title('Toy Spatiotemporal Dataset');
	drawnow
	pause(.05);
end
close(gcf);
drawnow

fprintf('\nTraining the model...\n')
data = reshape(data,nX*nY,nFrames)';

nHid = 500;

arch = struct('size',[nX*nY, nHid], ...
			  'nT', 8);

arch.opts = {'lRate', 0.1, ...
			 'nEpoch', 200, ...
			 'wDecay', 0.02, ...
			 'batchSz', 100, ...
			 'beginAnneal', 150, ...
			 'beginWeightDecay',200, ...
			 'sparsity', 0.01, ...
			 'sparseFactor', 10};%, ...
%  			 'visFun', @visToySpatioTemporalLearning, ...
%  			 'displayEvery', 25};

r = drbm(arch);
r = r.train(data);	% TRAIN RMB USING CD[1]

% GENERATE SEQUENCES OF FRAMES FROM THE MODEL

fprintf('\nNow we generate samples from the model...\n');
nFramesGenerated = 200;
nTrials = 5;
samples = [];
randIdx = randperm(nFrames);
for iT = 1:nTrials
%  	v0 = rand(r.nVis,r.nT+1);
	v0 = data(randIdx(iT):randIdx(iT)+8,:)';
	samples = [samples, r.sample(v0,nFramesGenerated)];
	fprintf('\n');
end

[~,nSamps] = size(samples);
s = reshape(samples,16,16,nSamps);

% VISUALIZE SAMPLES
close all
figure;
fprintf('\nDisplaying samples...\n');
for iF = 1:nSamps
	imagesc(s(:,:,iF)); colormap gray; axis image; axis off;
	title(sprintf('Samples Generated From\nTrained dRBM'))
	pause(.05)
end
close(gcf);
drawnow

