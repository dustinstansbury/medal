function [] = visRGBPatchearning(RBM,iE,jB);
%-----------------------------------------
%  [] = visPatchLearning(RBM,jB,iE,rotFlag);
%-----------------------------------------
% DES
figure(1)
if RBM.useGPU
	RBM = gpuGather(RBM);
end

if notDefined('iE'),iE = numel(RBM.e); end

if isfield(RBM.auxVars,'invXForm');
	invXForm = RBM.auxVars.invXForm;
else
	invXForm = eye(size(RBM.W,1));
end

nVis = min(64,floor(sqrt(size(RBM.W/3,2))).^2);

subplot(331);
visPatchesRGB(RBM.X(RBM.batchIdx{jB},:)',invXForm);
title('Batch Data');

subplot(332);
visPatchesRGB(RBM.pVis',invXForm);
title('Fantasies');


subplot(333);
visPatchesRGB(RBM.dW(:,1:nVis),invXForm);
title ('Weight Gradients');

subplot(334);
hist(RBM.b);
title('Visible Bias');

subplot(335);
hist(RBM.aHid(:));
title(sprintf('E[hid]=%1.2f\nTarget Sparsity =%0.4f',mean(RBM.aHid(:)),RBM.sparsity))

subplot(336);
visPatchesRGB(RBM.W(:,1:nVis),invXForm);
title('Basis/Connection Weights');


subplot(337);
plot(RBM.log.err(1:iE));
title('Reconstruction errors');

subplot(338)
hist(RBM.W(:));
title('Connection Weights');

subplot(339);
semilogy(RBM.log.eta(1:iE));
title('Learning Rate');

drawnow
