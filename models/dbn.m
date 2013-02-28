classdef dbn
% Deep Belief Network object.
%-----------------------------------------------------------------------------
% Initialize, train, and fine-tune a deep belief network.
%
% Supports hidden unit dropout in each layer
%
% Note this function calls the rbm.m object to form layers.
%-----------------------------------------------------------------------------
% DES
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
end

methods
	function self = dbn(arch)
		self = self.init(arch);
	end

	function print(self)
		properties(self)
		methods(self)
	end

	function self = train(self,X,targets);
	% PRETRAIN A DEEP BELIEF NETWORK
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
	end

	function net = fineTune(self,X,targets,ftArch);
	% FINE TUNE AUTOENCODER FEATURES FOR A TASK
	% USING BACKPROP (STOCHASTIC GRADIENT DESCENT)
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
	% PROPAGATE SIGNAL UP THROUGH THE LAYERS OF THE DBN

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

	function [error,misClass,pred] = classify(self,X,targets)
	% CLASSIFY (DEVO)

		if isvector(targets);
			targets = self.oneOfK(targets);
		end

		[nObs,nClass] = size(targets);
		
		% CALCULATE TOP LAYER ACTIVATIONS
		[~,~,tmp] = self.fProp(X,targets);

		% CALCULATE CLASS PROBABILITY
		pClass = tmp.softMax(bsxfun(@plus,tmp.aHid*tmp.classW',tmp.d));
		
		% WINNER-TAKE-ALL CLASSIFICATION
		[~, pred] = max(pClass,[],2);

		[~,targets] = max(targets,[],2);

		% CALCULATE MISSCLASSIFICATION RATE
		misClass = find(pred~=targets);
		error = numel(misClass)/nObs;
	end

	function self = init(self,arch)
	% INITIALIZE DEEP AUTOENCODER
	% <arch> IS AN ARCHTIECTURE STRUCT WITH THE FIELDS:
	%	.shape -- [#INPUT #HID1, ..., #HIDN, #OUT]
	%	.lRate -- THE LEARNING RATE FOR EACH AE LAYER
	%	.sparsity -- THE SPARSITY TARGET FOR HIDDEN UNITS
	%	.dropout -- THE HIDDEN UNIT DROPOUT RATE FOR EACH RBM LAYER

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
	% PREPROCESS INITIALIZATION INPUT
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