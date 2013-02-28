fprintf('\nHere we train Sparse and Denoising Autoencoders on the MNIST dataset.\n');

% LOAD THE MINIST DATASET
load('mnistSmall.mat','trainData','testData');

[nObs,nIn]= size(trainData);

nHid = 500;

% DEFINE SINGLE-LAYERED ARCHITECTURE
arch.size = [nIn nHid];
arch.actFun = {'sigmoid','sigmoid'}; % ALL SIGMOIDS
arch.lRate = [.1 .1];

% TRAINING OPTIONS	
arch.opts = {'nEpoch',10, ...
             'sparsity',.01, ...
             'costFun','xent'};

% INITIALIZE AUTOENCODER
a1 = ae(arch);

% TRAIN & TEST
fprintf('\nTraining a Sparse Autoencoder...\n');
a1 = a1.train(trainData);
[~,xent1] = a1.test(testData);

nVis = 225;
figure; visWeights(a1.layers{1}.W',0,[-1 1],1,nVis);
title(sprintf('Encoding Features of the Sparse Autoencoder\n(sorted by L2 norm)'));
drawnow

% DEFINE DENOISING ARCHITECTURE
arch2 = arch;

% ADD 20% DENOISING RATIO
arch2.opts = {'denoise',.2, ...
              'nEpoch',10, ...
              'costFun','xent'};

% INITIALIZE NETWORK
a2 = ae(arch2);

% TRIAN & TEST
fprintf('\n\nNow training a Denoising Autoencoder...\n');
a2 = a2.train(trainData);
[~,xent2] = a2.test(testData);

figure; visWeights(a2.layers{1}.W',0,[-1 1],1,nVis);
title(sprintf('Encoding Features of the Denoising Autoencoder\n(sorted by L2 norm)'))
drawnow

% DISPLAY
fprintf('\nTesting Mean Squared Error for Autoencoder: %1.2f\n',xent1);
fprintf('\nTesting Mean Squared Error for Denoising Autoencoder: %1.2f\n',xent2);