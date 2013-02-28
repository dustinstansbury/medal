function im = visMLCNNFeatureMaps(net,cLim)
%  im = visMLCNNFeatureMaps(net,cLim)
%------------------------------------------------------------------------------
% Display the feature maps for each layer in a convolutional neural network
% object. Assumes that the feature maps in <net> are activated from a call to
% .fProp.
%------------------------------------------------------------------------------
% DES

layerCnt = net.nLayers - 1;
nVisLayers = layerCnt;
subplottight(nVisLayers,1,layerCnt,.15);
imagesc(net.layers{1}.fm(:,:,:,1)); axis image;
set(gca,'Xtick',[]);
set(gca,'Ytick',[]);
ylabel('Input')
layerCnt = layerCnt - 1;
for lL = 2:net.nLayers-1
	fmSize = size(net.layers{lL}.fm);
	layerIm = reshape(net.layers{lL}.fm(:,:,:,1),[fmSize(1),prod(fmSize(2:3))]);
	subplottight(nVisLayers,1,layerCnt,.15);
	imagesc(layerIm); axis image;
	if ~notDefined('cLim')
		set(gca,'clim',cLim);
	end
	hold on;
	for jM = 1:net.layers{lL}.nFM;
		x0 = ((jM-1)*fmSize(2));
		y0 = 1 ; %fmSize(1);
		r = rectangle('Position',[x0+.5,y0-.5,fmSize(2),fmSize(1)]);
		set(r,'EdgeColor','r','Linewidth',2)
	end
	hold off;
	set(gca,'Xtick',[]);
	set(gca,'Ytick',[]);
	ylabel(sprintf('%s',net.layers{lL}.type))
	layerCnt = layerCnt - 1;
end