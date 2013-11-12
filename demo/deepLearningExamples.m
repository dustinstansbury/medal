function deepLearningExamples(type)
%  deepLearningExamples(demo)
%------------------------------------------------------------------------------
%  A set of demos for the MEDAL toolbox
%------------------------------------------------------------------------------
% <demo> is a string indicating the demonstration type. Options for <demo> are:
%
%    'rbm'          -- Train a binary-input RBM on the MNIST dataset and use
%                      it to classify test cases
%
%    'grbm'         -- Train a gaussian-input RBM on a toy Gaussian Mixture
%                      dataset and use it to classify test cases
%
%    'mlnn'         -- Train single and ulti-layer Neural Networks on the
%                      MNIST dataset and use them to classify test cases
%
%    'dbn'          -- Train a Deep Belief Network on MNIST and use the
%                      top layer to classify test cases.
%
%    'ae'           -- Train sparse and denoising autoencoders on MNIST digits
%
%    'dae'          -- Train a deep autoencoder on MNIST and finetune
%                      the network to classify test cases.
%
%    'crbm'         -- Train a Convolutional RBM on the MNIST dataset.
%
%    'mlcnn'        -- Train a mult-layer Convolutional Neural Network
%                      on MNIST and use it to classify test cases
%
%    'drbm'         -- Train a dynamic RBM on a toy spatiotemporal dataset
%
%    'mcrbm'        -- Train a Mean-Covariance RBM on a color image patches
%
%    'all'          -- Run all demos in sequence
%------------------------------------------------------------------------------
% DES    
% stan_s_bury@berkeley.edu

if notDefined('type'), type = 'all'; end

switch lower(type)

case {'binary rbm','rbm'}
% BERNOULLI-BERNOULLI RESTRICTED BOLZMANN MACHINES
clear all; close all; clc
fprintf('\nRunning Binary RBM demo for Classification (demoBinaryRBM_MNIST.m)\n')
demoBinaryRBM_MNIST
fprintf('\nRBM demo finished.\n')

case {'gaussian rbm','grbm'}
% GAUSSIAN-BERNOULLI RESTRICTED BOLZMANN MACHINE ON GMM DATASET
clear all; close all; clc
fprintf('\nRunning Gaussian-Bernoulli RBM demo (demoGaussianRBM_Classifer.m)\n')
demoGaussianRBM_GMM
fprintf('\nGaussian RBM demo finished.\n')

case {'mlnn'}
% MULTI-LAYER NEURAL NETWORK
clear all; close all; clc
fprintf('\nRunning Multi-layer Neural Net demo (demoMLNN.m)\n')
demoMLNN
fprintf('\nNeural Net demo finished\n')

case {'dbn'}
% DEEP BELIEF NETWORK FOR CLASSIFICATION
clear all; close all; clc
fprintf('\nRunning DBN for Classification demo (demoDBN.m)\n')
demoDBN
fprintf('\nDBN demo finished\n')

case {'autoencoder','ae'}
% DENOISING AUTOENCODER
clear all; close all; clc
fprintf('\nRunning Denoising and Sparse Autoencoder demo (demoAE.m)\n')
demoAE;
fprintf('\nAutoencoder demo finished.\n')

case {'deep autoencoder','dae'}
% DENOISING AUTOENCODER
clear all; close all; clc
fprintf('\nRunning Deep Autoencoder demo (demoDAE.m)\n')
demoDAE;
fprintf('\nDeep Autoencoder demo finished.\n')

case {'crbm','convolutional rbm'}
% CONVOLUTIONAL RBM
clear all; close all; clc
fprintf('\nRunning Convolutional RBM demo (demoBinaryCRBM)\n')
demoBinaryCRBM;
fprintf('\nConvolutional RBM demo finished.\n')


case {'mlcnn','cnn','convolutional neural net'}
% CONVOLUTIONAL NEURAL NETWORK
clear all; close all; clc
fprintf('\nRunning Convolutional Neural Net demo (demoMLCNN.m)\n')
demoMLCNN
fprintf('\nConvolutional Neural Net demo finished.\n')

case {'drbm','dynamic rbm'}
% DYNAMIC/CONDITIONAL RBM
clear all; close all; clc
fprintf('\nRunning Dynamic RBM demo (demoDRBM.m)\n')
demoDRBM
fprintf('\nDynamic RBM Demo finished.\n')

case {'mcrbm','mean covariance rbm'}
% MEAN-COVARIANCE RBM
clear all; close all; clc
fprintf('\nRunning Mean-Covariance RBM demo (demoMCRBM.m)\n')
demoMCRBM
fprintf('\nMean-Covariance RBM Demofinished.\n')

case 'all'
demos = {'rbm','grbm','mlnn','dbn','ae','dae','crbm','mlcnn','drbm','mcrbm'};
	for iD = 1:numel(demos)
		deepLearningExamples(demos{iD});
		fprintf('\nPress any key to resume');
		pause
	end
	clc; close all; drawnow; clear all; 
end
