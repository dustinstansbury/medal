classdef cnn

properties
	X
	nObs
	layers
	visFun
	nEpoch
	nFeats = 12;
	featSize = [7 7];
	poolSize = [2 2];
	eta = 0.1;
	displayEvery = 100;
	checkConverged = 0;
	anneal = 0;
	sparsity = 0.03;
	momentum = 0.9;
	wPenalty = .01;
	saveFold = './convNNSave';
	chkConverge = 0;
	log = struct();
	verbose = 1;
	varyEta = 0;
	saveEvery = 500;
	dataPoint = 0;
	dcSparse=0;
	
end % END PROPERTIES

methods

	function self = forwardProp(self,X)
		for iL = 1:self.nLayers
			switch self.layers(iL).type
			case 'imput'
				self = self.fPropInput();
			case 'conv'
				self. = self.fPropConv();
			case 'pool'
				self = self.fPropPool();
			case 'sample'
				self = self.fPropSample();			
			end
		end
	end

	function self = backProp(self,layer,J,dJ)

	end

	function [gradW, gradb] = gradient(self,layer)

	end

	function self = sampleUp(self,layer)

	end

	function self = sampleDown(self)

	end

	function self = fPropInput(self,iL)

	end

	function out = LCN(self,in);
	% LOCAL CONTRAST NORMALIZATION
	
	end

	function self = init(self)

	end

	function out = ff(in);
		out = flipud(fliplr(in));
	end
	
end

end