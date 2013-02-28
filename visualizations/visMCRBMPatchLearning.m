function [] = visMCRBMPatchLearning(RBM);

t = RBM.auxVars.t;

nShow = 225;
nShowC = min(nShow,size(RBM.C,2));
nShowM = min(nShow,size(RBM.W,2));

margin = .08;
figure(99); clf
subplottight(3,3,1,margin); visPatchesRGB(t.data,RBM.auxVars.invXForm);
title('Batch Data')

subplottight(3,3,2,margin); visPatchesRGB(t.negData,RBM.auxVars.invXForm);
title('HMC Samples')

subplottight(3,3,3,margin); visPatchesRGB(RBM.bV,RBM.auxVars.invXForm);
title('Visible Bias');

%  plot(RBM.log.err(1:iE-1)); xlabel('Epoch')
%  title('Reconstruction errors');


subplottight(3,3,4,margin); visPatchesRGB(RBM.C(:,1:nShowC),RBM.auxVars.invXForm);
title('C');

if RBM.modelMeans
	subplottight(3,3,5,margin);
	visPatchesRGB(RBM.W(:,1:nShowM),RBM.auxVars.invXForm);
	title('W')
end
subplottight(3,3,6,margin); imagesc(RBM.P); colormap hot; axis image; colorbar
%  subplottight(3,3,6,margin); plot(diag(RBM.P)); 
title('P')


subplottight(3,3,7,margin); visPatchesRGB(RBM.dC(:,1:nShowC),RBM.auxVars.invXForm);
title('dC')

if RBM.modelMeans
	subplottight(3,3,8,margin); visPatchesRGB(RBM.dW(:,1:nShowM),RBM.auxVars.invXForm);
	title('dW')
end
subplottight(3,3,9,margin); imagesc(RBM.dP); colormap jet; axis image; colorbar
title('dP')

drawnow