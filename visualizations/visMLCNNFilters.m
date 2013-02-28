function im = visMLCNNFilters(net,cLim)
%  im = visMLCNNFilters(net)
%------------------------------------------------------------------------------
% Display the feature maps for each layer in a convolutional neural network
% object. Assumes that the feature maps in <net> are activated from a call to
% .fProp.
%------------------------------------------------------------------------------
% NEED TO INCLUDE SUBPLOTTIGHT IN UTILS!!

clf
if notDefined('cLim')
	cLim = 1;
end

nVisLayers = floor((net.nLayers - 2)/2);
layerCnt = 1;
for lL = 2:net.nLayers-1
	if strcmp(net.layers{lL}.type,'conv')
		filtSize = size(net.layers{lL}.filter);
		layerIm = reshape(net.layers{lL}.filter,[filtSize(1)*filtSize(2),prod(filtSize(3:end))]);
		subplottight(1,nVisLayers,layerCnt,.15);
		visWeights(layerIm,0,cLim);
		layerCnt = layerCnt + 1;
		title(sprintf('Conv. Layer %d',lL));
	end
end