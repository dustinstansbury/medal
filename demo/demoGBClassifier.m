
load('gaussianData.mat');
trainData = data; clear data;
testData = testdata; clear testdata;
%------------------------------------------
%  trainData = bsxfun(@minus,trainData,mean(trainData));
%  testData = bsxfun(@minus,testData,mean(testData));
%  
%  trainData = bsxfun(@rdivide,trainData,std(trainData));
%  testData = bsxfun(@rdivide,testData,std(testData));

args = {'type','GB', ...
		'eta',.01, ...
		'batchSz',64, ...
		'nEpoch',200, ...
		'nHid',25, ...
		'learnSigma2',1 ...
		'sampleVis',1, ...
		'sparse',0.002, ...
		'visFun',@visGBClassLearn};

clear r;  r = rbmClassifier(args,trainData,labels);r = r.train;

figure(2);
subplot(121);
plot(r.e); axis square; title(sprintf('Reconstruction error \nCD[%d]',r.nGibbs)); xlabel('Iteration #')
set(gca,'fontsize',8);

subplot(122);
Wgb = r.vis(); title('Learned Feature Weights');
set(gca,'clim',[min(r.W(:)),max(r.W(:))]); axis image

% PREDICT CLASSES OF HOLD-OUT SET
[pred,classError,misClass]=r.predict(testData,testlabels);

figure(3);
myScatter(testData,[],[],testlabels);
hold on;
myScatter(testData(misClass,:));
title(sprintf('Missclassifications in red (rate = %1.2f%%)',100*classError));