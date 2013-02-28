function visNNLearning(net,fig,transWeights);
%  visNNLearning(net);

layers = net.layers;

if notDefined('fig')
	fig = 99;
end

figure(fig);
clf

nLayers = numel(layers);

if notDefined('transWeights')
	transWeights = zeros(nLayers,1);
end

for iL = 1:nLayers
	subplot(3,nLayers,iL);
	visWeights(layers{iL}.W,transWeights(iL));
	title(sprintf('Layer %i Weights',iL));
	axis square
end

for iL = nLayers+1:2*nLayers
	subplot(3,nLayers,iL);
	visWeights(layers{iL-nLayers}.dW,transWeights(iL-nLayers));
	title(sprintf('Layer %i Gradients',iL-nLayers));
	axis square
end

for iL = 2*nLayers+1:3*nLayers
	subplot(3,nLayers,iL);
	hist(layers{iL-2*nLayers}.output(:),20);
	title(sprintf('Layer %i -- %s output',iL-2*nLayers,layers{iL-2*nLayers}.actFun));
	axis square
end
drawnow