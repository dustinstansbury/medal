classdef dbn
% Deep Belief Network object.
%-----------------------------------------------------------------------------
% Initialize, train, and fine-tune a deep belief network.
% Note this function calls the rbm.m object to form layers.
%-----------------------------------------------------------------------------
% Dustin E. Stansbury
% stan_s_bury@berkeley.edu

properties
	class = 'dbn';			% OBJECT CLASS
	arch;					% ARCHITECTURE/OPTIONS
	inputType='binary';		% DEFAULT INPUT TYPE
	classifier=false;		% CLASSIFIER LAYER AT TOP
	nLayers;				% TOTAL # OF LAYERS (INCLUDING INPUT)
	nRBMLayers;				% # OF RBM LAYERS
	rbmLayers;				% RBM LAYERS
	verbose = 1;			% DISPLAY OUTPUT
	saveDir
end

methods
	function self = dbn(arch)
	% d = dbn(arch)
	%--------------------------------------------------------------------------
	% DBN constructor
	%--------------------------------------------------------------------------
	% INPUT:
	%  <arch>:  - a set of arguments defining the DBN architecture.
	%
	% OUTPUT:
	%     <d>:  - an DBN model object.
	%--------------------------------------------------------------------------
		self = self.init(arch);
	end

	function print(self)
	% print()
	%--------------------------------------------------------------------------
	% Display the properties and methods for the DBN object class.
	%--------------------------------------------------------------------------
		properties(self)
		methods(self)
	end

	function self = train(self,X,targets);
	% self = train(X,[targets])
	%--------------------------------------------------------------------------
	% Perform greedy layer-wise training of of a DBN
	%--------------------------------------------------------------------------
	% INPUT:
	%        <X>:  - to-be modeled data |X| = [#Obs x #Vis]
	%  <targets>:  - category targets. can be either [#Obs x #Class] as 1 of
	%                K representation or [#Obs x 1], where each entry is a
	%                numerical category label. (optional)
	% OUTPUT:
	%     <self>:  - trained DBN object.
	%--------------------------------------------------------------------------
		for iL = 1:self.nRBMLayers
			if self.verbose,
				self.printProgress('layerTrain',iL);
			end

			% TRAIN AN RBM LAYER
			if iL == self.nRBMLayers & self.classifier
				self.rbmLayers{iL} = self.rbmLayers{iL}.train(X,targets);
			else
				
				self.rbmLayers{iL} = self.rbmLayers{iL}.train(X,[]);
			end

			% PREPROCESS INPUT FOR NEXT LAYER
			X = self.rbmLayers{iL}.hidExpect(X);
		end
		if ~isempty(self.saveDir),
			d = self;
			save(fullfile(self.saveDir,'dbn.mat'),'d','-v7.3');
		end
	end

	function net = fineTune(self,X,targets,ftArch);
	%  net = fineTune(X,targets,ftArch);
	%--------------------------------------------------------------------------
	% Initialize a multi-layer neural network with trained DBN features and
	% fine tune for a task using backprop (stochastic gradient descent)
	%--------------------------------------------------------------------------
	% INPUT:
	%        <X>:  - to-be modeled data |X| = [#Obs x #Vis]
	%  <targets>:  - target variables.
	%   <ftArch>:  - architecture for the corresponding neural network (see
	%                mlnn.m)
	%
	% OUTPUT:
	%      <net>:  - a trained multi-layer neural network
	%-------------------------------------------------------------------------- 
		[nObs,nOut] = size(targets);
		if self.verbose
			self.printProgress('fineTune')
		end
		
		ftArch.size = [self.arch.size,nOut];

		net = mlnn(ftArch);
		% INITIALIZE MLNN WITH DBN WEIGHTS
		for iL = 1:numel(self.arch.size)-1
			net.layers{iL}.W = self.rbmLayers{iL}.W';
			net.layers{iL}.b = self.rbmLayers{iL}.c';
		end
		% FINE TUNE
		net = net.train(X,targets);
	end

	function [outAct,outProb,layerOut] = fProp(self,X,targets);
	% [outAct,outProb] = fProp(X,[targets]);
	%-------------------------------------------------------------------------- 
	% Forward propagate signal up through the layers of the dbn.
	%--------------------------------------------------------------------------
	% OUTPUT:
	%  <outAct>:  - activation of the top layer/output hidden units
	% <outProb>:  - probabilities of top layer hidden units
	%--------------------------------------------------------------------------
		if notDefined('targets'),
			targets = 0;
		end
		
		for iL = 1:self.nRBMLayers-1
			X = self.rbmLayers{iL}.hidExpect(X);
		end
		layerOut = self.rbmLayers{end}.hidGivVis(X,targets,1);
		outAct = layerOut.aHid;
		outProb = layerOut.pHid;
	end

	function [pred,error,misClass] = classify(self,X,targets)
	%  [pred,error,misClass] = classify(X,[targets])
	%--------------------------------------------------------------------------
	% Forward propagate data <X> through the layers of the DBN and classify.
	% If <targets> provided, also returns the classification error <error>,
	% and the indices of the misclassified inputs <misClass>.
	%--------------------------------------------------------------------------
	
	
		if notDefined('targets')
			targets = []; 
		else
			if isvector(targets);
				targets = self.oneOfK(targets);
			end
		end
		
		% PROPAGATE DATA SIGNALS UP
		for iL = 1:self.nRBMLayers-1
			X = self.rbmLayers{iL}.hidExpect(X);
		end
		
		% CALCULATE TOP LAYER ACTIVATIONS
		nClasses = self.rbmLayers{end}.nClasses;
		tmp = self.rbmLayers{end}.hidGivVis(X,zeros(size(X,1),nClasses),1);
		
		% CALCULATE CLASS PROBABILITY
		pClass = tmp.softMax(bsxfun(@plus,tmp.aHid*tmp.classW',tmp.d));
		
		% WINNER-TAKE-ALL CLASSIFICATION
		[~, pred] = max(pClass,[],2);

		if ~notDefined('targets') && nargout > 1
			[~,targets] = max(targets,[],2);

			% CALCULATE MISSCLASSIFICATION RATE
			misClass = find(pred~=targets);
			error = numel(misClass)/size(X,1);
		end
	end

	function self = init(self,arch)
	% dbn = init(arch)
	%--------------------------------------------------------------------------
	% Initialize deep belief network, based on architecture structure <arch>.
	%--------------------------------------------------------------------------
	% INPUT:
	% <arch>:  - Is an archtiecture struct with the fields:
	%            .shape    --> [#Vis #Hid1, ..., #HidN, #out]
	%            .lRate    --> the learning rate for each layer
	%            .sparsity --> the sparsity target for hidden units
	%            .dropout  --> the hidden unit dropout rate for each rbm layer
	%--------------------------------------------------------------------------
		arch = self.ensureArchetecture(arch);
		
		% GLOBAL OPTIONS
		if isfield(arch,'opts');
			opts = arch.opts;
			fn = fieldnames(self);
			for iA = 1:2:numel(opts)
				if ~isstr(opts{iA})
					error('<opts> must be a cell array string-value pairs.');
				elseif sum(strcmp(fn,opts{iA}))
					self.(opts{iA})=opts{iA+1};
				end
			end
		end
		
		self.arch = arch;
		self.nLayers = numel(arch.size);
		self.nRBMLayers = self.nLayers - 1;

		% INITIALIZE RBM LAYERS
		for iL = 2:self.nLayers
			layerArch.size = [arch.size(iL-1),arch.size(iL)];
			if iL == 2
				layerArch.inputType = self.inputType;
			else
				layerArch.inputType = 'binary';
			end

			self.rbmLayers{iL-1} = rbm(layerArch);
			self.rbmLayers{iL-1}.lRate = arch.lRate(iL-1);
			self.rbmLayers{iL-1}.nEpoch = arch.nEpoch(iL-1);
			self.rbmLayers{iL-1}.sparsity = arch.sparsity(iL-1);
			self.rbmLayers{iL-1}.dropout = arch.dropout(iL-1);
			
			if self.classifier & (iL == self.nLayers)
				self.rbmLayers{end}.classifier = true;
			end
		end
	end

	function arch = ensureArchetecture(self,arch)
	% arch = ensureArchitecture(arch)
	%--------------------------------------------------------------------------
	% Preprocess the provided architecture structure <arch>.
	%--------------------------------------------------------------------------
	% INPUT:
	% <arch>:  - is either a [2 x 1] vector giving the [#Vis x #Hid], in which
	%            case we use the default model parameters, or it is a strucure
	%            with any of the fields:
	%             .size                --> Network size; [#Vis x # Hid]
	%             .nEpoch(optional)    --> # of training epochs for each layer
	%             .lRate (optional)    --> learing rate for each layer
	%             .sparsity (optional) --> sparsity rate for each layer
	%             .dropout (optional)  --> dropout rate for each layer
	%--------------------------------------------------------------------------
		if ~isstruct(arch),arch.size = arch; end

		nLayers = numel(arch.size);
		
		% CHECK # EPOCHS
		if ~isfield(arch,'nEpoch')
			arch.lRate = repmat(100,1,nLayers-1);
		elseif numel(arch.nEpoch) == 1
			arch.nEpoch = repmat(arch.nEpoch,1,nLayers-1);
		end
		
		% CHECK LEARNING RATES
		if ~isfield(arch,'lRate')
			arch.lRate = repmat(.01,1,nLayers-1);
		elseif numel(arch.lRate) == 1
			arch.lRate = repmat(arch.lRate,1,nLayers-1);
		end

		% CHECK SPARSITY TARGET RATES
		if ~isfield(arch,'sparsity')
			arch.sparsity = repmat(0,1,nLayers-1);
		elseif numel(arch.sparsity) == 1
			arch.sparsity = repmat(arch.sparsity,1,nLayers-1);
		end

		% CHECK DROPOUT 
		if ~isfield(arch,'dropout')
			arch.dropout = repmat(0,1,nLayers-1);
		elseif numel(arch.dropout) == 1
			arch.dropout = repmat(arch.dropout,1,nLayers-1);
		end
	end
	
	function printProgress(self,type,aux)
	% VERBOSE
		switch type
			case 'layerTrain'
				fprintf('\n\nTraining RBM Layer: %i / %i\n',aux(1),self.nLayers-1);
			case 'fineTune'
				fprintf('\n\nFine-tuning using backprop:\n')
			case 'save'
				fprintf('\nSaving...\n\n');
		end
	end
end

end