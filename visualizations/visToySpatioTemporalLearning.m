function [] = visToySpatioTemporalLearning(RBM);
%-----------------------------------------
%  isToySpatioTemporalLearning(RBM);
%-----------------------------------------
% DES
figure(99);
clf

set(gcf,'name','Learning Dynamic RBM');

subplottight(3,3,1,.15);

nVis = 1;
batchX = RBM.auxVars.batchX(1:nVis,:,:,:);
[nO,nV,nT] = size(batchX);
nV = sqrt(nV);
batchIm = zeros(nO*nV,nV*nT);
for iO = 1:nO
	for jT = 1:nT
		rIdx = (iO-1)*nV+1:iO*nV;
		cIdx = (jT-1)*nV+1:nV*jT;
		batchIm(rIdx,cIdx) = reshape(squeeze(batchX(iO,:,jT)),[nV,nV]);
	end
end

imagesc(batchIm); colormap gray; axis image
title('Sample Time Series');

subplottight(3,3,2,.15);
visWeights(RBM.pVis(1:nVis,:)');
title('Sample Reconstructions');

nVis = min(size(RBM.W,1),100);
subplottight(3,3,3,.1);
visWeights(RBM.dW(1:nVis,:)',1,1);
title ('Weight Gradients');

subplottight(3,3,4,.15);
visWeights(RBM.b(1:nVis),1);
title('Visible Bias');

subplottight(3,3,5,.25);
hist(RBM.W(:),20); axis square
title('Connection Weights');

subplottight(3,3,6,.15);
visWeights(RBM.W(1:nVis,:)',1);
title('Connection Weights');

subplottight(3,3,7,.15);
plot(RBM.auxVars.error); axis square
title('Reconstruction errors');

subplottight(3,3,8,.15);
hist(RBM.aHid(:),30); axis square
title(sprintf('E[hid]=%1.2f\nTarget Sparsity =%0.4f',mean(RBM.aHid(:)),RBM.sparsity))

subplottight(3,3,9,.15);
plot(RBM.auxVars.lRate); axis square
title('Learning Rate');

drawnow
