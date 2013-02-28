fprintf('\nHere we train a Convolutional Multi-layered Neural Network\non the MNIST dataset.\n\n')

load 'mnistSmall.mat';

trainData = reshape(trainData',28,28,1,10000);
trainLabels = double(trainLabels');

testData = reshape(testData',28,28,1,2000);
testLabels = double(testLabels');


dataSize = [28,28,1];  % [nY x nX x nChannels]

arch = {struct('type','input','dataSize',dataSize), ...
        struct('type','conv','filterSize',[9 9], 'nFM', 6), ...
        struct('type','subsample','stride',[2 2]), ...
        struct('type','conv','filterSize',[5 5], 'nFM',12), ...
        struct('type','subsample','stride',2), ...
        struct('type','output', 'nOut', 10)};

n = mlcnn(arch);
n.batchSize = 100;
n.nXVal = 0;  % ADJUST FOR CROSSVALIDATION
n.costFun = 'xent';
n.nEpoch = 5;

n = n.train(trainData,trainLabels);
clear trainData trainLabels;

classErr = n.test(testData,testLabels,'classerr');
close all

figure('Name','Learned Layer Filters');
fprintf('\nDisplaying Layer Filters...\n');
visMLCNNFilters(n,[-.4 .4]) 

figure('Name','Layer Feature Maps');
fprintf('\nDisplaying Feature Maps...\n');
visMLCNNFeatureMaps(n); colormap gray



