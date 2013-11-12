classdef dae
% Deep Autoencoder object.
%-----------------------------------------------------------------------------
% Initialize, train, and/or fine-tune a multilayer autencoder.
%
% Supports denoising autoencoders and hidden unit dropout.
%
% Note this implementation calls the mlnn.m object to form autoencoder layers.
%-----------------------------------------------------------------------------
% DES

properties
	class = 'dae';			%  OBJECT NAME
	arch;					% ARCHITECTURE/OPTIONS
	nLayers;				% # OF AUTOENCODERS
	aLayers;				% AUTOENCODER/NN LAYERS
	verbose = 1;			% DISPLAY OUTPUT
	saveDir;
end

methods
	function self = dae(arch)
	% net = dae(arch)
	%--------------------------------------------------------------------------
	%dae constructor method. Initilizes a mlnn object, <net> given a user-
	%provided architecture, <arch>.
	%--------------------------------------------------------------------------
		self = self.init(arch);
	end

	function print(self)
	%print()
	%--------------------------------------------------------------------------
	%Print properties and methods for dae object.
	%--------------------------------------------------------------------------
	
		properties(self)
		methods(self)
	end

	function self = init(self,arch)
	% net = init(arch)
	%--------------------------------------------------------------------------
	% Initialize deep autoencoder with a user-defined architecture <arch>.
	%--------------------------------------------------------------------------
	% <arch> IS AN ARCHTIECTURE STRUCT WITH THE FIELDS:
	%	.size -- [#INPUT #HID1, ..., #HIDN, #OUT]
	%	.costFun -- COST FUNCTION FOR FIRST LAYER LEARNING
	%	.lRate -- THE LEARNING RATE FOR EACH AE LAYER
	%	.denoising -- THE INPUT CORRUPTION NOISE PERCENTAGE ()
	%	.sparsity -- THE SPARSITY TARGET FOR HIDDEN UNITS
	%	.dropout -- THE HIDDEN UNIT DROPOUT RATE FOR EACH AE LAYER
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
		
		% INITIALIZE AUTOENCODER LAYERS
		for iL = 2:self.nLayers
			layerArch.size = [arch.size(iL-1),arch.size(iL),arch.size(iL-1)];

			if iL == 2
				layerArch.actFun = {'sigmoid',arch.mapping};
			else
				layerArch.actFun = {'sigmoid'};
				arch.costFun = 'xent';
			end
			
			layerArch.lRate = arch.lRate(iL-1);
			self.aLayers{iL-1} = mlnn(layerArch);
			self.aLayers{iL-1}.nEpoch = arch.nEpoch(iL-1);
			self.aLayers{iL-1}.costFun = arch.costFun;
			self.aLayers{iL-1}.denoise = arch.denoise(iL-1);
			self.aLayers{iL-1}.sparsity = arch.sparsity(iL-1);
			self.aLayers{iL-1}.dropout = arch.dropout(iL-1);
		end
	end

	function self = train(self,data);
	%net = train(data);
	%--------------------------------------------------------------------------
	% Train a deep autoencoder.
	%--------------------------------------------------------------------------
		for iL = 1:numel(self.aLayers)
			if self.verbose,
				self.printProgress('layerTrain',iL);
			end
			% TRAIN AN AUTOENCODER LAYER
			self.aLayers{iL} = self.aLayers{iL}.train(data,data);

			% PREPROCESS INPUT FOR NEXT LAYER
			tmpNet = self.aLayers{iL}.fProp(data,data);
			data = tmpNet.layers{end-1}.act;
		end
		if ~isempty(self.saveDir),
			d = self;
			save(fullfile(self.saveDir,'dae.mat'),'d','-v7.3');
		end
	end

	function net = fineTune(self,data,targets,ftArch);
	% net = fineTune(data,targets,ftArch);
	%--------------------------------------------------------------------------
	% Fine tune autoencoder features for a task using backprop (stochastic gradient 
	% descent)
	%--------------------------------------------------------------------------
		[nObs,nOut] = size(targets);
		if self.verbose
			self.printProgress('fineTune')
		end

		ftArch.size = [self.arch.size,nOut];
		
		net = mlnn(ftArch);
		% INITIALIZE MLNN WITH AE WEIGHTS
		for iL = 1:numel(self.arch.size)-1
			net.layers{iL}.W = self.aLayers{iL}.layers{1}.W;
			net.layers{iL}.b = self.aLayers{iL}.layers{1}.b;
		end

		% FINE TUNE
		net = net.train(data,targets);
	end

	function arch = ensureArchetecture(self,arch)
	%arch = ensureArchitecture(arch)
	%--------------------------------------------------------------------------
	%Utility function to reprocess a supplied architecture, <arch>
	%--------------------------------------------------------------------------
	
		if ~isstruct(arch),arch.size = arch; end

		nLayers = numel(arch.size);

		% CHECK FIRST LAYER MAPPING
		if ~isfield(arch,'mapping')
			mapping0 = 'sigmoid';% ASSUME BINARY INPUT BY DEFAULT
			arch.mapping = mapping0;
		end

		% CHECK COST FUNCTION
		if ~isfield(arch,'costFun')
			costFun0 = 'xent';% ASSUME BINARY INPUT BY DEFAULT
			arch.costFun = costFun0; 
		end

		% CHECK LEARNING RATES
		if ~isfield(arch,'nEpoch')
			nEpoch0 = 10; % DEFAULT # TIMES TO SEE DATA
			arch.lRate = repmat(nEpoch0,1,nLayers-1);
		elseif numel(arch.nEpoch) == 1
			arch.nEpoch = repmat(arch.nEpoch,1,nLayers-1);
		end
		
		% CHECK LEARNING RATES
		if ~isfield(arch,'lRate')
			lRate0 = .1; % DEFAULT LEARNING RATE
			arch.lRate = repmat(lRate0,1,nLayers-1);
		elseif numel(arch.lRate) == 1
			arch.lRate = repmat(arch.lRate,1,nLayers-1);
		end

		% CHECK DENOISING RATES
		if ~isfield(arch,'denoise')
			denoise0 = 0; % NO DENOISING AE BY DEFAULT
			arch.denoise = repmat(denoise0,1,nLayers-1);
		elseif numel(arch.denoise) == 1
			arch.denoise = repmat(arch.denoise,1,nLayers-1);
		end

		% CHECK SPARSITY TARGET RATES
		if ~isfield(arch,'sparsity')
			sparsity0 = 0; % NO SPARSITY TARGET BY DEFAULT
			arch.sparsity = repmat(sparsity0,1,nLayers-1);
		elseif numel(arch.sparsity) == 1
			arch.sparsity = repmat(arch.sparsity,1,nLayers-1);
		end

		% CHECK DROPOUT 
		if ~isfield(arch,'dropout')
			dropout0 = 0; % NO DROPOUT BY DEFAULT
			arch.dropout = repmat(dropout0,1,nLayers-1);
		elseif numel(arch.dropout) == 1
			arch.dropout = repmat(arch.dropout,1,nLayers-1);
		end
	end
	function printProgress(self,type,aux)
	%printProgress(type)
	%--------------------------------------------------------------------------
	% Verbose utility function. <type> is the type of message to print. <aux> 
	% are optional arguments.
	%--------------------------------------------------------------------------
		switch type
			case 'layerTrain'
				fprintf('\n\nTraining Layer: %i / %i\n',aux(1),self.nLayers-1);
			case 'fineTune'
				fprintf('\n\nFine-tuning using backprop:\n')
			case 'save'
				fprintf('\nSaving...\n\n');
		end
	end
end % END METHODS
end % END CLASSDEF