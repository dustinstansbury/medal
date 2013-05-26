fprintf('\nHere we train an RBM with Continuous inputs (2D Gaussian Mixture).\n');



load('gaussianData.mat');
trainData = data; clear data;
testData = testdata; clear testdata;

arch = struct('size', [2,25], 'classifier',true, 'inputType','gaussian');

opts = {'eta',.01, ...
		'batchSz',64, ...
		'nEpoch',50, ...
		'sparse',0.002, ...
		'displayEvery',10};
%  		'visFun',@visGaussianClassLearning};
arch.opts = opts;

clear r;
r = rbm(arch);
r = r.train(trainData,labels);
[pred,classError,misClass] = r.classify(testData,testlabels);

clf;
scatter(testData(:,1),testData(:,2),[],pred,'.'); colormap lines(3)
hold on
m = plot(testData(misClass,1),testData(misClass,2),'ko');
hold off;
title(sprintf('Predicted classes -- Error=%1.2f %%',classError*100));
legend(m,'Misclassified');
