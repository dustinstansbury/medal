function visNNLearningAuxInput(net);
%  visNNLearningAuxInput(net);

layers = net.layers;

if notDefined('fig')
	fig = 99;
end

nLayers = numel(layers);

if notDefined('transWeights')
	transWeights = zeros(nLayers,1);
end



input = layers{1}.W;
input = net.auxInput*input;
figure(99); visWeights(input(:,1:10:end));
drawnow

figure(98);
for iL = 1:nLayers
	subplot(nLayers,1,iL);
	hist(layers{iL}.output(:),20);
	colormap hot
	title(sprintf('Layer %d outputs',iL));
end
%  
%  for iL = 1:nLayers
%  	subplot(3,nLayers,iL);
%  	if iL == 1
%  		
%  		input = layers{iL}.W;
%  		input = net.auxInput*input;
%  		figure(99); visWeights(input);
%  		figure(98);
%  	else
%  		input = layers{iL}.W;
%  	end
%  	visWeights(input);
%  	title(sprintf('Layer %i Weights',iL));
%  	axis square
%  end
%  
%  for iL = nLayers+1:2*nLayers
%  	subplot(3,nLayers,iL);
%  	visWeights(layers{iL-nLayers}.dW);
%  	title(sprintf('Layer %i Gradients',iL-nLayers));
%  	axis square
%  end
%  
%  for iL = 2*nLayers+1:3*nLayers
%  	subplot(3,nLayers,iL);
%  	hist(layers{iL-2*nLayers}.output(:),20);
%  	title(sprintf('Layer %i -- %s output',iL-2*nLayers,layers{iL-2*nLayers}.actFun));
%  	axis square
%  end
drawnow