<<<<<<< HEAD
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
ftArch.lRate = .01;  % LEARNING RATE FOR ALL LAYERS
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
=======
function d = testDAE(test);

if notDefined('test')
	test = 'mnist';
end

switch test
case 'mnist'
	d = dae(); % STACK A DEFAULT DBN
	load('defaultData.mat');
%  	[m,net] = train_autoencoder(data,[100,100,25]);
	nTest = 100;
	testData = testdata(1:nTest,:);
	
	figure(1);
	clf
	subplot(131);
	visWeights(testData',1);
	title('Test Data Points');

	recon = d.propDataUp(testData);
	err = 1/nTest * sum(sum(testData - recon).^2);
		
	subplot(132)	
	visWeights(recon',1);
	title(sprintf('Top Layer Output:\nError: %4.2f', err));
	drawnow;

	d.nFineTune = 20;
	d = d.fineTune();
	
	recon = d.propDataUp(testData);
	mapX = d.propDataUp(data,'map');
	d.mapX = mapX;
	err = 1/nTest * sum(sum(testData - recon).^2);

	figure(1)
	subplot(133)
	visWeights(recon',1);
	title(sprintf('Top Layer Output (w/FineTuning):\nError: %4.2f', err));
	
case 'mixture'
	load('gaussianData.mat');
	args = {'type',{'GB'}, ...
		'nHid', [10, 3] ...
		'verbose', 1, ...
		'eta', [0.01] ...
		'momentum', 0.5, ...
		'nEpoch', [100] ...
		'wDecay', 0.0002, ...
		'batchSz', 100, ...
		'anneal', 1, ...
		'varyEta',[1], ...
		'nGibbs', 1, ...
		'bpCost', 'xent', ...
		'nFineTune', 20};
		
	d = dae(args,data);
	keyboard
	figure(1);
	subplot(121);
	myScatter(testdata,[],[],testlabels);
	title('Test Data Points');
	colormap jet

	subplot(122)
	try
		myScatter(d.mapX,[],[],labels);
	catch
		myScatter3(d.mapX,[],[],labels);
	end
	colormap jet


case 'faces'
	load('facesDataGray.mat');
	testIdx = randperm(size(data,1));
	trainIdx = testIdx(1:2300);
	testIdx(1:2300) = [];
	trainData = data(trainIdx,:);
	testData = data(testIdx,:);
	clear data
	args = {'type',{'GB'}, ...
		'nHid', [100, 100, 30] ...
		'verbose', 1, ...
		'eta', [0.01] ...
		'momentum', 0.5, ...
		'nEpoch', [100] ...
		'wDecay', 0.0002, ...
		'batchSz', 100, ...
		'anneal', 1, ...
		'varyEta',[1], ...
		'nGibbs', 1, ...
		'bpCost', 'xent', ...
		'nFineTune', 0, ...
		'learnSigma2',[1,0 0]};

	d = dae(args,trainData);
	nTest = size(testData,1);
	recon1 = d.propDataUp(testData);
	err1 = 1/nTest * sum(sum(testData - recon1).^2);
	d.nFineTune = 10;
	d = d.fineTune();
	recon2 = d.propDataUp(testData);
	err2 = 1/nTest * sum(sum(testData - recon2).^2);
	figure(1); clf
	subplot(131); visWeights(testData');
	subplot(132); visWeights(recon1');
	title(sprintf('Reconstruction\n(error %4.2f)',err1));
	subplot(133); visWeights(recon2');
	title(sprintf('Reconstructions\n(w/FineTuning; error %4.2f)',err2));
	
end
>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014
