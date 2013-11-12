function [] = visBinaryRBMLearning(RBM);
%-----------------------------------------
%  [] = visBinaryRBMLearning(RBM);
%-----------------------------------------
% DES
figure(99)
set(gcf,'name','Learning Binary RBM');


nVis = floor(sqrt(size(RBM.W,1))).^2;

subplottight(3,3,1,.15);
visWeights(RBM.auxVars.batchX',0);
title('Batch Data');

subplottight(3,3,2,.15);
visWeights(RBM.pVis(:,1:nVis)',0);
title('Reconstructions');

subplottight(3,3,3,.15);
visWeights(RBM.dW(1:nVis,:),1,[]);
title ('Weight Gradients');


subplottight(3,3,4,.15);
visWeights(RBM.b(1:nVis),1);
title('Visible Bias');

subplottight(3,3,5,.15);
hist(RBM.W(:),20);
title('Connection Weights');

subplottight(3,3,6,.15);
visWeights(RBM.W(1:nVis,:),1,[]);
title('Connection Weights');

subplottight(3,3,7,.15);
plot(RBM.auxVars.error);
title('Reconstruction errors');

subplottight(3,3,8,.15);
hist(RBM.aHid(:));
title(sprintf('E[hid]=%1.2f\nTarget Sparsity =%0.4f',mean(RBM.aHid(:)),RBM.sparsity))

subplottight(3,3,9,.15);
plot(RBM.auxVars.lRate);
title('Learning Rate');

drawnow
