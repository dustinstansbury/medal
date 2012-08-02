function [] = visBGLearning(RBM,iE,jB);
%-----------------------------------------
%  [] = visBBMLearning(RBM,jB,iE,rotFlag);
%-----------------------------------------
% DES

if notDefined('iE')
	iE = numel(RBM.e);
end

nVis = floor(sqrt(size(RBM.W,1))).^2;


subplot(331);
visWeights(RBM.X(RBM.batchIdx{jB},:)',1);
title('Batch Data');

subplot(332);
visWeights(RBM.pVis(:,1:nVis)',1);
title('Fantasies');


subplot(333);
visWeights(RBM.dW(1:nVis,:),1,[]);
title ('Weight Gradients');


subplot(334);
visWeights(RBM.b(1:nVis));
title('Visible Bias');


subplot(335)
visWeights(RBM.sigma2);
title('Hidden Variance');

subplot(336);
visWeights(RBM.W(1:nVis,:),1,[]);
title('Connection Weights');


subplot(337);
plot(RBM.e(1:iE));
title('Reconstruction errors');

subplot(339);
semilogy(RBM.a(1:iE));
title('Learning Rate');

drawnow
