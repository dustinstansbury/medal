function [] = visMLNNLearning(MLNN);
%------------------------------------------------------------------
%  visMLNNLearning(MLNN);
%------------------------------------------------------------------
% DES


nHid = MLNN.nLayers - 2;
nRows = MLNN.nLayers;
nCols = 6;

figure(99)
clf
% INPUT LAYER VISUALIZATIONS


% VISUALIZE HIDDEN LAYERS 
for iL = 1:nHid
	subplottight(nRows,nCols,(nRows*nCols)-(iL+1)*nCols+1,.25);
	imagesc(MLNN.layers{iL+1}.act); colorbar
	title(sprintf('Hid %d act.',iL))
	
	subplottight(nRows,nCols,(nRows*nCols)-(iL+1)*nCols+2,.25);
	imagesc(MLNN.layers{iL+1}.W); colorbar
	title(sprintf('Hid %d W',iL))

	subplottight(nRows,nCols,(nRows*nCols)-(iL+1)*nCols+3,.25);
	imagesc(MLNN.layers{iL+1}.dW);colorbar
	title(sprintf('Hid %d dW',iL))
	
	subplottight(nRows,nCols,(nRows*nCols)-(iL+1)*nCols+4,.25);
	hist(MLNN.layers{iL+1}.W(:))
	title(sprintf('Hist Hid %d W',iL))
	
	subplottight(nRows,nCols,(nRows*nCols)-(iL+1)*nCols+5,.25);
	bar(MLNN.layers{iL+1}.b);
	xlim([0.5,numel(MLNN.layers{iL+1}.b)+.5])
	title(sprintf('Hid %d b',iL))

	subplottight(nRows,nCols,(nRows*nCols)-(iL+1)*nCols+6,.25);
	bar(MLNN.layers{iL+1}.db);
	xlim([0.5,numel(MLNN.layers{iL+1}.b)+.5])
	title(sprintf('Hid %d db',iL)); 
end


% VISUALIZE INPUT TLAYERS
subplottight(nRows,nCols,(nRows-1)*nCols+1,.25);
imagesc(MLNN.layers{1}.act); colorbar
title('Input act')

subplottight(nRows,nCols,(nRows-1)*nCols+2,.25);
imagesc(MLNN.layers{1}.W); colorbar
title('Input W')

subplottight(nRows,nCols,(nRows-1)*nCols+3,.25);
imagesc(MLNN.layers{1}.dW); colorbar
title('Input dW')

subplottight(nRows,nCols,(nRows-1)*nCols+4,.25);
hist(MLNN.layers{1}.W(:));
title('Hist Input W')

subplottight(nRows,nCols,(nRows-1)*nCols+5,.25);
bar(MLNN.layers{1}.b);
xlim([0.5,numel(MLNN.layers{1}.db)+.5])
title('Input b')

subplottight(nRows,nCols,(nRows-1)*nCols+6,.25);
bar(MLNN.layers{1}.db);
xlim([0.5,numel(MLNN.layers{1}.db)+.5])
title('Input db')


% VISUALIZE OUTPUT LAYERS

subplottight(nRows,4,1,.25);
hist(MLNN.layers{end}.act(:));
title('Hist Network Output');

subplottight(nRows,4,2,.25);
imagesc(MLNN.layers{end}.act); colorbar
title('Network Output');

subplottight(nRows,4,3,.25);
imagesc(MLNN.auxVars.targets); colorbar
title('Targets');

subplottight(nRows,4,4,.25);
hist(MLNN.auxVars.targets(:)); 
title('Hist Targets');

drawnow