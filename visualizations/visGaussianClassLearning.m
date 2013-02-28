function [] = visGaussianClassLearning(RBM);
%-----------------------------------------
%  [] =  visGaussianClassLearning(RBM);
%-----------------------------------------
% DES

nVis = floor(sqrt(size(RBM.W,1))).^2;

subplot(331);
scatter(RBM.auxVars.batchX(:,1),RBM.auxVars.batchX(:,2),'r.');

title('Batch Data');
xlim([-10, 10]); ylim([-20, 0]);
%  xlim([-5, 5]); ylim([-5, 5]);

subplot(332);
scatter(RBM.pVis(:,1),RBM.pVis(:,2),'b.');
title('Fantasies');
%  xlim([-5, 5]); ylim([-5, 5]);
xlim([-10, 10]); ylim([-20, 0]);


subplot(333);
visWeights(flipud(RBM.dW(1:nVis,:)));
title ('Weight Gradients');


subplot(334);
bar(RBM.b);
title('Visible Bias');

%
%  subplot(335)
%  bar((RBM.sigma2));
%  title('Visible Variance');

subplot(336);
visWeights(flipud(RBM.W(1:nVis,:)));
title('Connection Weights');

subplot(337);
plot(RBM.auxVars.error);
title('Reconstruction errors');

subplot(339);
semilogy(RBM.auxVars.lRate);
title('Learning Rate');

drawnow
