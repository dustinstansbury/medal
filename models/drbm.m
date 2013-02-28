classdef drbm
% Dynamic/Conditional Restrictied Boltzmann Machine model object
%------------------------------------------------------------------------------
% Initialize and train a dynamce RBM model.
%
% Supports binary and Gaussian input variables.
%
% Model regularizers include L2 weight decay and hidden unit target sparsity
%
% Adapted from code provided by Graham Taylor, Geoff Hinton, and Sam Roweis,
% available at:
% http://www.cs.toronto.edu/~gwtaylor/publications/nips2006mhmublv
%------------------------------------------------------------------------------
% DES

properties
	class = 'drbm'; 		% DYNAMIC RBM MODEL CLASS

	% ARCHITECTURE
	inputType = 'binary';	% TYPE OF VISIBLES
	nVis 					% # OF VISIBLE UNITS (DIMENSIONS)
	nHid 					% # OF HIDDEN UNITS
	nT						% ORDER OF MODEL (# FRAMES BACK)

	% MODEL PARAMETERS
	W						% CONNECTION WEIGHTS
	dW 						% LEANING INCREMENT FOR CONN. WEIGHTS
	A 						% AR WEIGHTS (ON VISIBLES)
	dA 						% LEARNING INCREMENT ON AR WEIGHTS
	B 						% PAST-TO-HIDDEN WEIGHTS
	dB 						% LEARNING INCREMENT OF P-T-H WEIGHTS
	b 						% STATIC VISIBLE UNIT BIASES
	db						% LEARNING INCREMENT FOR VIS. BIAS
	bStar 					% DYNAMIC VISIBLE BIASES
	c 						% STATIC HIDDEN UNIT BIASES
	dc 						% LEARNING INCREMENT FOR HID. BIAS
	cStar					% DYNAMIC HIDDEN BIASES

	% MODEL STATES
	aVis 					% VISIBLE LAYER ACTIVATIONS
	pVis 					% VISIBLE LAYER PROBS
	aHid 					% HIDDEN LAYER ACTIVATION
	pHid 					% HIDDEN LAYER PROBS
	pHid0 					% INITIAL HIDDEN LAYER PROBS
	aHid0 					% INITIAL HIDDEN LAYER ACTIVATION

	% (DEFAULT) LEARNING PARAMETERS
	lRate = 1e-3;			% LEARNING RATE
	sampleVis = 0;			% SAMPLE THE VISIBLE UNITS
	sampleHid = 1;			% SAMPLE HIDDEN UNITS 
	momentum = 0.2;			% MOMENTUM TERM FOR WEIGHT ESTIMATION
	nEpoch = 1000;			% # OF FULL PASSES THROUGH TRIANING DATA
	wPenalty = 0.0002;		% CURRENT WEIGHT PENALTY
	sparsity = 0.01;		% SPARSENESS TARGET
	sparseGain=5;			% GAIN ON LEARNING RATE FOR SPARSITY
	batchSz = 100;			% # OF TRAINING POINTS PER BATCH
	nGibbs = 1;				% CONTRASTIVE DIVERGENCE
	beginWeightDecay=Inf;	% # OF EPOCHS TO BEGIN WEIGHT DECAY
	beginAnneal=Inf;		% # OF EPOCHS TO BEGIN SIM. ANNEALING

	% OTHER
	batchIdx 				% MINIBATCH INDICES INTO TRAINING DATA
	log 					% ERROR AND LEARNING RATE LOGS
	verbose = 1;			% DISPLAY PROGRESS
	saveEvery = 0;			% # OF EPOCHS TO SAVE INTERMEDIATE MODELS
	displayEvery=500;		% # WEIGHT UPDATES TO VISUALIZE LEARNING
	visFun 					% USER-DEFINED FUNCTION ('@myFun')
	auxVars 	 			% AUXILLARY VARIABLES, JUST IN CASE
	useGPU = 0; 			% USE CUDA, IF AVAILABLE
	gpuDevice 				% GPU DEVICE STRUCTURE
	saveFold='./drbmSave';	% # DEFAULT SAVE FOLDDER
		
end % END PROPERTIES

methods
	% CONSTRUCTOR
	function self = drbm(arch)
		self = self.init(arch);
	end

	function [] = print(self)
		properties(self)
		methods(self)
	end
	
	function self = train(self,X)
	% MAIN
		wPenalty = self.wPenalty;
		[~,nFrames] = size(X);
		
		self.batchIdx = self.createBatches(nFrames);

		if self.useGPU
			self = gpuDistribute(self);
			X = gpuArra(X);
		end
		
		dCount = 1;
		mCount = 1;
		iE = 1;
		while 1
			sumErr = 0;

			% BEGIN SIMULATED ANNEALING?
			if iE >= self.beginAnneal
				self.lRate = max(1e-10,self.lRate/max(1,iE/self.beginAnneal));
			end
			
			% BEGIN WEIGHT DECAY?
			if iE >= self.beginWeightDecay
				self.wPenalty = wPenalty;
			else
				self.wPenalty = 0;
			end

			% LOOP OVER BATCHES 
			for jB = 1:numel(self.batchIdx)

				% COMPOSE DELAYED VERSIONS OF BATCH DATA
				nObs = numel(self.batchIdx{jB});
				batchX = zeros(nObs,self.nVis,self.nT+1);
				
				batchX(:,:,1) = X(self.batchIdx{jB},:);
				for iT = 1:self.nT
					batchX(:,:,iT+1) = X(self.batchIdx{jB}-iT,:);
				end

				self = self.composeBiases(batchX);
				self = self.runGibbs(batchX(:,:,1),self.nGibbs);
				self = self.updateParams(batchX);
				batchErr = self.batchError(batchX(:,:,1));
				sumErr = sumErr + batchErr;
				
				if ~isempty(self.visFun) & ~mod(dCount,self.displayEvery);
					self.auxVars.batchX = squeeze(batchX	);
					self.auxVars.error = self.log.err(1:max((iE-1),1));
					self.auxVars.lRate = self.log.lRate(1:max((iE-1),1));
					self.visLearning;
				end
				dCount = dCount+1;
			end

			% SPARSITY (ONLY WORKS ON STATIC BIASES)
			if self.sparsity
				dcSparse = -self.lRate*self.sparseGain*(mean(self.pHid,2)-self.sparsity);
				self.c = self.c + dcSparse;
			end
			
			self.log.err(iE) = sumErr;
			self.log.lRate(iE) = self.lRate;
			
			if self.verbose, self.printProgress(sumErr,iE,jB); end

			% SAVE IF NECESSARY
			if ~mod(iE, self.saveEvery), self.save; end
			if iE >= self.nEpoch, break; end
			iE = iE + 1;
		end
				
		% PULL DATA FROM GPU, IF NECESSARY
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
			self.gpuDevice = [];
		end
		fprintf('\n');
	end

	function self = runGibbs(self,X,nGibbs)
	% MAIN GIBBS SAMPLER
		nObs = size(X,1);
		for iC = 1:nGibbs
			% GO UP
			if iC == 1
				self = self.hidGivVis(X,1);
				self.pHid0 = self.pHid; % LOG STATES FOR GRADIENT 
				self.aHid0 = self.aHid; % CALCULATIONS
			else
				self = self.hidGivVis(X,0);
			end
			% GO DOWN
			self = self.visGivHid(self.aHid,self.sampleVis);
			X = self.aVis;
		end
		% FINISH
		self = self.hidGivVis(self.aVis,0);
	end

	function self = hidGivVis(self,X,sampleHid)
	% p(H|V), SAMPLE H, IF NEEDED
		self.pHid = self.sigmoid(bsxfun(@plus,self.W*X',self.cStar));
		
		if sampleHid
			self.aHid = single(self.pHid>rand(size(self.pHid)));
		else
			self.aHid = self.pHid;
		end
	end
	
	function self = visGivHid(self,aHid,sampleVis)
	% p(V|H)
		nObs = size(aHid,1);
		switch self.inputType
		case 'binary'
			self.pVis = self.sigmoid(bsxfun(@plus,aHid'*self.W,self.bStar'));
			if sampleVis
				self.aVis = self.pVis>rand(size(self.pVis));
			else
				self.aVis = self.pVis;
			end
		case 'gaussian'
			mu = bsxfun(@plus,aHid'*self.W,self.bStar');
			self.pVis = self.drawNormal(mu);
			if sampleVis
				self.aVis = self.pVis;
			else
				self.aVis = mu;
			end
		end
	end

	function self = composeBiases(self,X)
	% COMPOSE DYNAMIC BIASES
		nObs = size(X,1);
			
		% VISIBLE DYNAMIC BIASES (AUTOREGRESSIVE)
		self.bStar = zeros(self.nVis,nObs);
		
		for iT = 1:self.nT
			self.bStar = self.bStar + self.A(:,:,iT)*X(:,:,iT+1)';
		end
		% ADD STATIC VISIBLE BIASES
		self.bStar = bsxfun(@plus,self.bStar,self.b); % Eq 4.7

		% HIDDEN DYNAMIC BIASES
		self.cStar = zeros(self.nHid,nObs);
		for iT = 1:self.nT
			self.cStar = self.cStar + self.B(:,:,iT)*X(:,:,iT+1)';
		end
		
		% ADD STATIC HIDDEN BIASES
		self.cStar = bsxfun(@plus,self.cStar,self.c); % Eq 4.4
	end
	
	
	function self = updateParams(self,X);
	% LEARNING RULES
		nObs = size(X,1);
		
		% UNDIRECTED CONNECTION WEIGHTS
		dW = (self.pHid0*X(:,:,1) - self.pHid*self.aVis)/nObs;
		self.dW = self.momentum*self.dW + (1-self.momentum)*dW - ...
				  self.wPenalty*self.W;
		self.W = self.W + self.lRate*self.dW;

		% DIRECTED CONNECTION WEIGHTS
		for iT = 1:self.nT
			% AUTOREGRESSIVE
			dAPos = bsxfun(@minus,X(:,:,1)',self.bStar)*X(:,:,iT+1);
			dANeg = bsxfun(@minus,self.aVis',self.bStar)*X(:,:,iT+1);
			dA = (dAPos - dANeg)/nObs;

			self.dA(:,:,iT) = self.momentum*self.dA(:,:,iT) + ...
			                  (1-self.momentum)*dA - ...
			                  self.wPenalty*self.A(:,:,iT);

			self.A(:,:,iT) = self.A(:,:,iT) + ...
			                 self.lRate*self.dA(:,:,iT);

            % DYNAMIC HIDDEN
			dBPos = self.aHid0*X(:,:,iT+1);
			dBNeg = self.pHid*X(:,:,iT+1);
			dB = (dBPos - dBNeg)/nObs;

			self.dB(:,:,iT) = self.momentum*self.dB(:,:,iT) + ...
		                      (1-self.momentum)*dB - ...
		                      self.wPenalty*self.B(:,:,iT);

			self.B(:,:,iT) = self.B(:,:,iT) + ...
			                 self.lRate*self.dB(:,:,iT);
		end
		
		% STATIC BIASES
		db = (mean(X(:,:,1)) - mean(self.pVis))';
		self.db = self.momentum*self.db + self.lRate*db;
		self.b = self.b + self.db;

		dc = mean(self.pHid0,2) - mean(self.pHid,2);
		self.dc = self.momentum*self.dc + self.lRate*dc;
		self.c = self.c + self.dc;
	end

	function samples = sample(self,v0,nFrames,nGibbs);
	%% GENERATE A SEQUENCE OF <nFrames> FROM
	%% THE MODEL GIVEN INITIAL CONDITIONS v0
		if notDefined('nFrames'),nFrames = self.nT; end
		if notDefined('nGibbs'),nGibbs = 50; end
		
		[nDim,N] = size(v0);
		v0 = reshape(v0,[1,nDim,N]);
		v0 = cat(3,v0(:,:,1),v0);
		samples = zeros(self.nVis,nFrames);
		
		if self.useGPU
			samples = gpuArray(single(samples));
			self = gpuDistribute(self);
		end
		
		for iF = 1:nFrames
			if self.verbose,
				fprintf('\rGenerating Frame %d/%d',iF,nFrames)
			end
			self = self.composeBiases(v0);
			self = self.runGibbs(v0(:,:,1),nGibbs); % CHECK THIS...
			samples(:,iF) = self.aVis(:);
			% MAKE T --> T-1, COPY STATE T
			for ii = 1:2
				v0 = cat(3,self.aVis,v0);
			end
			v0(:,:,end) = [];
		end
	end

	% CALCULATE BATCH SUM OF SQUARED ERROR
	function err = batchError(self,X);
		err = sum(sum((X-self.aVis).^2));
	end

	function arch = ensureArchitecture(self,arch);
	% PREPROCESS PROVIDED ARCHITECTURE
		if ~isstruct(arch), arch.size = arch; end
		if ~isfield(arch,'size'),
			error('must provide an architecture size');
		end
		if ~isfield(arch,'inputType'),
			inputType0 = 'binary'; % ASSUME BINARY BY DEFAULT
			arch.inputType = inputType0;
		 end
		if ~isfield(arch,'nT'),
			nT0 = 3; % ASSUME T-3 MEMORY
			arch.nT = nT0;
		end
	end

	function self = init(self,arch)
		% ARCHITECTURE
		arch = self.ensureArchitecture(arch);
		self.nVis = arch.size(1);
		self.nHid = arch.size(2);
		self.inputType = arch.inputType;
		self.nT = arch.nT;
	
		% GLOBAL OPTIONS
		if isfield(arch,'opts')
			opts = arch.opts;
			fn = fieldnames(self);
			for iA = 1:2:numel(opts)
				if ~isstr(opts{iA})
					error('<.opts> must be a cell array string-value pairs.')
				elseif sum(strcmp(fn,opts{iA}))
					self.(opts{iA})=opts{iA+1};
				end
			end
			clear opts;
		end

		self.log.err = zeros(1,self.nEpoch);
		self.log.lRate = self.log.err;

%  		range = sqrt(6/(self.nVis*self.nT+self.nHid));
%  		self.W = 2*range*(rand(self.nHid,self.nVis) - .5);
		self.W = .01*(randn(self.nHid,self.nVis));
		self.dW = zeros(size(self.W));
		
		self.A = .01*randn(self.nVis,self.nVis,self.nT);
		self.dA = zeros(size(self.A));
		
		self.B = .01*randn(self.nHid,self.nVis,self.nT);
		self.dB = zeros(size(self.B));

		self.b = .01*randn(self.nVis,1);
		self.db = zeros(size(self.b));
		
		self.c = -1 + .01*randn(self.nHid,1);
		self.dc = zeros(size(self.c));
	end

	function batches = createBatches(self,nFrames)
	% CREATE BATCH INDICES THAT REFERENCE INTO SEQUENCE FRAMES
	% SPLIT FRAMES INTO CHUNKS (IGNORE FIRST #T FRAMES)
	
		chunkIdx = self.nT+1:(self.nT+1):nFrames;
		chunks = [];
		for ch = 1:(length(chunkIdx) - 1)
	        chunks{ch} = [chunkIdx(ch):chunkIdx(ch+1)-1];
        end
        
        nChunks = numel(chunks);
		chunksPerBatch = floor(self.batchSz/(self.nT+1));
        
        % SHUFFLE THE CHUNKS
        chunks = chunks(randperm(nChunks));
        
        % CREATE THE BATCHES
        batchIdx = 1:chunksPerBatch:nChunks;
        for iB = 1:numel(batchIdx)-1
			batches{iB} = [chunks{batchIdx(iB):batchIdx(iB+1)-1}];
        end
        batches{end+1} = [chunks{batchIdx(end):nChunks}];
	end

	function p  = sigmoid(self,X)
		if self.useGPU
			p = arrayfun(@(x)(1./(1 + exp(-x))),X);
		else
			p = 1./(1 + exp(-X));
		end
	end

	function p = drawNormal(self,mu);
		p = mvnrnd(mu,ones(1,self.nVis));
	end

	% VISUALIZATION
	function visLearning(self);
		if isempty(self.visFun)
			switch self.inputType
				case 'binary'
					visBBLearning(self,iE,jB);
				case 'gaussian'
					visGBLearning(self,iE,jB);
			end
		else
			self.visFun(self);
		end
	end

	% VERBOSE
	function [] = printProgress(self,sumErr,iE,jB)	
		if iE > 1
			if self.log.err(iE) > self.log.err(iE-1) & iE > 1
				indStr = '(UP)    ';
			else
				indStr = '(DOWN)  ';
			end
		else
			indStr = '';
		end
		fprintf('Epoch %d/%d --> Recon. error: %f %s\r', ...
		iE,self.nEpoch,sumErr,indStr);
	end

	function [F] = freeEnergy(self,X)
	%% CALCULATE FREE ENERGY OF VISIBLES
		F = [];
	end

	function success = converged(self,errors);
		success = (abs(mean(gradient(smooth(errors)))/max(self.e)) <= 0.0001);
		if success
			fprintf('\nCONVERGED.\n');
		end
	end

	function save(self)
		r = self;
		if ~exist(r.saveFold,'dir'),mkdir(r.saveFold);end
		save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',iE)),'r'); clear r;
	end
end % END METHODS
end % END CLASSDEF