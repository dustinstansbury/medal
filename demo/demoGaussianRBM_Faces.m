fprintf('\nHere we train an RBM with Continuous inputs (Faces Dataset).\n');

load('facesDataGray.mat');

data = bsxfun(@minus,data,mean(data));
data = bsxfun(@rdivide,data,std(data));

nHid = 1000;
arch = struct('size', [361,nHid], 'inputType','gaussian');

opts = {'lRate',.001, ...
		'batchSz',50, ...
		'nEpoch',2000, ...
		'wPenalty', 0.002, ...
		'sparsity',0.01, ...
		'sparseFactor', 5, ...
		'displayEvery',100, ...
		'visFun',@visGaussianRBMLearning};
arch.opts = opts;

clear r;
r = rbm(arch)
r = r.train(data);

