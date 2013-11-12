function [] = visGaussianRBMLearning(RBM);
%-----------------------------------------
% [] = visGaussianRBMLearning(RBM);
%-----------------------------------------
% DES

figure(99);
clf;
set(gcf,'name','Learning Gaussian RBM');

[nObs,nDim] = size(RBM.W);
nVis = min(64,nDim);

wLims = [min(RBM.W(:)),max(RBM.W(:))];

subplottight(3,3,1,.15);
visWeights((RBM.auxVars.batchX'));
title('Batch Data');

subplottight(3,3,2,.15);
visWeights((RBM.pVis'));
title('Fantasies');


subplottight(3,3,3,.15);
visWeights(RBM.dW(:,1:nVis),0,[]);
title ('Weight Gradients');


subplottight(3,3,4,.15);
visWeights((RBM.b(1:nVis)'));
title('Visible Bias');


subplottight(3,3,5,.15);
hist(RBM.aHid(:));
title(sprintf('E[hid]=%1.2f\nTarget Sparsity =%0.4f',mean(RBM.aHid(:)),RBM.sparsity))


subplottight(3,3,6,.15);
visWeights((RBM.W(:,1:nVis)),0,[]);
title('Connection Weights');


subplottight(3,3,7,.15);
plot(RBM.auxVars.error);
title('Reconstruction errors');

subplottight(3,3,8,.15);
hist(RBM.W(:));
title('Connection Weights');

subplottight(3,3,9,.15);
semilogy(RBM.auxVars.lRate);
title('Learning Rate');

drawnow
