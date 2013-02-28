function [] = visBinaryCRBMLearning(RBM);
%------------------------------------------------------------------
%  visBinaryCRBMLearning(RBM);
%------------------------------------------------------------------
% DES


figure(99)
colormap gray
set(99,'name','Learning Convolutional RBM (Binary Visible)');
subplot(221);
imagesc(RBM.auxVars.batchX'); axis image; axis off;
title('Visible Data');

subplot(222);
imagesc(RBM.eVis'); axis image; axis off;
title('Reconstruction');

[c,r,k]=size(RBM.eHid);
subplot(223);
data = reshape(RBM.eHid,r*c,k);
visWeights(data,1);
title('Feature Maps');

[c,r,k]=size(RBM.W);
subplot(224);
data = reshape(RBM.W,r*c,k);
visWeights(data,1);
title('Learned Filters');
drawnow

figure(98)
colormap gray;
set(98,'name','Learning Convolutional RBM (Binary Visible)');
subplot(221);
bar(1:RBM.nFM,RBM.c);  axis square
xlim([1 RBM.nFM]);axis square;
xlabel('Feature Index')
title('Hidden Biases')

subplot(222);
data = squeeze(mean(mean(RBM.eHid0)));
bar(1:RBM.nFM,data);
xlim([1 RBM.nFM]);axis square;
xlabel('Feature Index')
title(sprintf('Mean Hidden Activation\nactual=%g \n target = %g\n',mean(data),RBM.sparsity))
drawnow

subplot(223);
data = reshape(RBM.dW,r*c,k);
visWeights(data); axis image; axis off
title('Weight Gradients')

subplot(224);
bar(1:RBM.nFM,RBM.auxVars.dcSparse); axis square;
xlim([1 RBM.nFM]);axis square;
xlabel('Hidden Feature Index')
title('Sparsenss Offset')
%  title('Hidden Bias Gradients')

