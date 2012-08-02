function [] = visRBMLearning(RBM,iE,jB,rotFlag);
%-----------------------------------------
%  [] = visRBMLearning(RBM,jB,iE,rotFlag);
%-----------------------------------------
% DES

if notDefined('iE')
	iE = numel(RBM.e);
end

nVis = floor(sqrt(size(RBM.W,1))).^2;

wLims = [min(RBM.W(:)),max(RBM.W(:))];
subplot(321);
visWeights(RBM.W(1:nVis,:),wLims,rotFlag);
title('Connection Weights');

subplot(322);
if strcmp(RBM.type,'GB')
	visWeights(RBM.b(1:nVis)',[],rotFlag);
else
	visWeights(RBM.b(1:nVis),[],rotFlag);
end
title('Visible Bias');

subplot(323);
visWeights(RBM.X(RBM.batchIdx{jB},:)',[],rotFlag);
title('Batch Data');

subplot(324);
visWeights(RBM.pVis(:,1:nVis)',[min(RBM.pVis(:)), max(RBM.pVis(:))],rotFlag);
title('Reconstructed Data');

subplot(325);
plot(RBM.e(1:iE));
title('Reconstruction errors');

subplot(326);
semilogy(RBM.a(1:iE));
title('Weight Decay Rate');
%  subplot(326);
%  visWeights(RBM.dW(1:nVis,:));
%  title ('Weight Gradients');
