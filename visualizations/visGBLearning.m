function [] = visGaussianRBMLearning(RBM,iE,jB);
%-----------------------------------------
%  [] = visBBMLearning(RBM,jB,iE,rotFlag);
%-----------------------------------------
% DES

if notDefined('iE')
	iE = numel(RBM.e);
end

nVis = floor(sqrt(size(RBM.W,1))).^2;

wLims = [min(RBM.W(:)),max(RBM.W(:))];

subplot(331);
visWeights((RBM.X(RBM.batchIdx{jB},:)'));
title('Batch Data');

subplot(332);
visWeights((RBM.pVis(:,1:nVis)'));
title('Fantasies');


subplot(333);
visWeights(RBM.dW(1:nVis,:),0,[]);
title ('Weight Gradients');


subplot(334);
visWeights((RBM.b(1:nVis)'));
title('Visible Bias');


subplot(335)
visWeights((RBM.sigma2(:,1:nVis)'));
title('Visible Variance');

subplot(336);
visWeights((RBM.W(1:nVis,:)),0,[]);
title('Connection Weights');


subplot(337);
plot(RBM.e(1:iE));
title('Reconstruction errors');

subplot(338)
hist(RBM.W(:));
title('Connection Weights');

subplot(339);
semilogy(RBM.a(1:iE));
title('Learning Rate');

drawnow
