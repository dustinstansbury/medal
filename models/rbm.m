classdef rbm
% Restricted Boltzmann Machine Model object:
%----------------------------------------------------------------------------
% Initialize and train an RBM energy-based model. Supports binary, Gaussian,
% and multinomial inputs (still in development).
%
% Supports binary and noisy rectified linear units (NReLU) in the hidden layer.
%
% Supports joint modeling of binary and multinomial variables for classification.
%
% Model regularizers include L2 and L1 (via subgradients) weight decay, hidden
% unit sparsity (binary units only), and hidden unit dropout.
%----------------------------------------------------------------------------
% Dustin E. Stansbury
% stan_s_bury@berkeley.edu


properties
	class = 'rbm'; 		% GENERAL CLASS OF MODEL
	inputType = 'binary';% TYPE OF RBM ('BB','GB')
	classifier = false; % TRAIN MULTINOMIAL UNITS FOR CLASSIFICATION IN PARALLEL
	nClasses;			% # OF OUTPUT CLASSES (.classifier = true)
	nObs;				% # OF TRAINING OBSERVATIONS
	nVis;				% # OF VISIBLE UNITS (DIMENSIONS)
	nHid = 100;			% # OF HIDDEN UNITS
	rlu = 0;			% USE RECTIFIED LINEAR UNITS
	W;					% CONNECTION WEIGHTS
	dW;					% GRADIENT FOR CONN. WEIGHTS
	b;					% VISIBLE UNIT BIASES
	db;					% GRADIENT FOR VIS. BIAS
	c;					% HIDDEN UNIT BIASES
	dc;					% GRADIENT FOR HID. BIAS
	classW=0			% CLASSFIER OUTPUT WEIGHTS
	dClassW; 			% CLASS. WEIGHT GRADIENTS
	d=0;				% CLASSIFIER BIAS
	dd;					% GRADIENT FOR classifier BIASES
	log;				% ERROR AND LEARNING RATE LOGS
	aVis;				% VISIBLE LAYER ACTIVATIONS
	aHid;				% HIDDEN LAYER ACTIVATION
	pHid;				% HIDDEN LAYER PROBS
	pHid0;				% INITIAL HIDDEN LAYER PROBS
	aHid0;				% INITIAL HIDDEN LAYER ACTIVATION
	eHid;				% EXPECTATION OF HIDDEN STATE (POST TRAINING)
	pClass;				% PROBABILITIES OF CLASSIFIER UNITS (.classifier = 1)
	aClass;				% ACTIVATION OF CLASSIFIER UNITS (.classifier = 1)
	docLen;				% DOCUMENT LENGTH (MULTINOMIAL INPUTS)	
	lRate = 0.1;		% DEFAULT LEARNING RATE
	batchIdx = [];		% BATCH INDICES INTO TRAINING DATA
	sampleVis = 0;		% SAMPLE THE VISIBLE UNITS
	sampleHid = 1;		% SAMPLE HIDDEN UNITS 
	momentum = 0;		% DEFAULT MOMENTUM TERM 
	nEpoch = 100;		% # OF FULL PASSES THROUGH TRIANING DATA
	wPenalty = 0;		% CURRENT WEIGHT PENALTY
	sparsity = 0;		% SPARSENESS FACTOR
	dropout = 0;		% HIDDEN UNIT DROPOUT
	doMask = 1;			% DROPOUT MASK
	topoMask = [];		% MASK FOR TOPOGRAPHY
	sparseGain=5;		% LEARNING RATE GAIN FOR SPARSITY
	batchSz = 100;		% # OF TRAINING POINTS PER BATCH
	nGibbs = 1;			% CONTRASTIVE DIVERGENCE (1)
	beginAnneal = Inf;	% # OF EPOCHS TO START SIMULATED ANNEALING
	beginWeightDecay=1% # OF EPOCHS TO START WEIGHT PENALTY
	verbose = 1;		% DISPLAY PROGRESS
	saveEvery = 0;		% # OF EPOCHS TO SAVE INTERMEDIATE MODELS
	displayEvery=500;	% # WEIGHT UPDATES TO VISUALIZE LEARNING
	visFun = [];		% USER-DEFINED FUNCTION ('@myFun')
	auxVars = []; 		% AUXILLARY VARIABLES, JUST IN CASE
	useGPU = 0; 		% USE CUDA, IF AVAILABLE
	gpuDevice;			% GPU DEVICE STRUCTURE
	interactive = 0;	% STORE A GLOBAL COPY FOR INTERACTIVE TRAINING
	saveFold
end % END PROPERTIES

methods

	function self = rbm(arch)
	% r = rbm(arch)
	%--------------------------------------------------------------------------
	% RBM constructor
	%--------------------------------------------------------------------------
	% INPUT:
	%  <arch>:  - a set of arguments defining the RBM architecture.
	%
	% OUTPUT:
	%     <r>:  - an RBM model object.
	%--------------------------------------------------------------------------
		if notDefined('arch')
		else
			self = self.init(arch);
		end
	end

	function print(self)
	% print()
	%--------------------------------------------------------------------------
	% Display the properties and methods for the RBM object class.
	%--------------------------------------------------------------------------
		properties(self)
		methods(self)
	end
	
	function self = train(self,X,targets)
	% self = train(X,[targets])
	%--------------------------------------------------------------------------
	% Train an RBM using Contrastive Divergence
	%--------------------------------------------------------------------------
	% INPUT:
	%        <X>:  - to-be modeled data |X| = [#Obs x #Vis]
	%  <targets>:  - category targets. can be either [#Obs x #Class] as 1 of
	%                K representation or [#Obs x 1], where each entry is a
	%                numerical category label. (optional)
	% OUTPUT:
	%     <self>:  - trained RBM object.
	%--------------------------------------------------------------------------
		lRate0 = self.lRate;
		if self.sparsity > 0, meanAct = 0; end
		if notDefined('targets')
			targets = 0;
		end

		self.nObs = size(X,1);

		% INITIALIZE CLASSIFIER UNITS
		if self.classifier || any(targets) 
			self.classifier = true;
			[self,targets] = self.initClassifer(targets);
		end
		
		self.log.err = zeros(1,self.nEpoch);
		self.log.lRate = self.log.err;
	
		wPenalty = self.wPenalty;
		dCount = 1;
		mCount = 1;
		
		self.batchIdx = self.createBatches(X);

		% SEND DATA TO GPU
		% USER WILL NEED TO IMPLEMENT gpuDistribute.m
		if self.useGPU
			self = gpuDistribute(self);
			X = gpuArray(X);
			targets = gpuArray(targets);
		end
		
		iE = 1;
		while 1
			sumErr = 0;

			% SIMULATED ANNEALING
			if iE > self.beginAnneal
				self.lRate = max(lRate0*((iE-self.beginAnneal)^(-.25)),1e-8);
			end
			
			% WEIGHT DECAY
			if iE < self.beginWeightDecay
				self.wPenalty = 0;
			else
				self.wPenalty = wPenalty;
			end

			% LOOP OVER BATCHES
			for jB = 1:numel(self.batchIdx)
				batchX = X(self.batchIdx{jB},:);
				if self.classifier
					batchTargets = targets(self.batchIdx{jB},:);
				else
					batchTargets = 0;
				end
				
				if strcmp(self.inputType,'multinomial')
					self.docLen = sum(batchX,2);
				end
				
				if self.dropout > 0 % SAMPLE DROPOUT MASK ONECE PER MINIBATCH
					self.doMask = rand(size(batchX,1),self.nHid)>self.dropout;
				else
					self.doMask = 1;
				end

				% SAMPLE FROM MODEL, CALCULATE GRADIENTS, UPDATE PARAMS
				self = self.runGibbs(batchX,batchTargets);
				self = self.updateParams(batchX,batchTargets);
				sumErr = self.accumErr(batchX,sumErr);
				
				if ~isempty(self.visFun) & ~mod(dCount,self.displayEvery)&iE>1;
					self.auxVars.batchX = batchX;
					self.auxVars.batchTargets = targets;
					self.auxVars.error = self.log.err(1:iE-1);
					self.auxVars.lRate = self.log.lRate(1:iE-1);
					self.visLearning;
				end
				dCount = dCount+1;
				% MOVING AVERAGE OF HIDDEN ACTIVATIONS FOR SPARSITY
				if self.sparsity & ~self.rlu
					meanAct = mean(self.aHid)*0.1 + 0.9*meanAct;
				end
			end

			% SPARSITY (BINARY HIDDENS ONLY)
			if self.sparsity & ~self.rlu
				dcSparse = -self.lRate*self.sparseGain*(meanAct-self.sparsity);
				self.c = self.c + dcSparse;
			end

 			% ENDUCE SPARSITY
 			% if self.sparsity & ~self.rlu
 			% 	dcSparse = -self.lRate*self.sparseGain*(mean(self.pHid)-self.sparsity);
 			% 	self.c = self.c + dcSparse;
 			% end

			% LOG ERRORS, ETC.
			self.log.err(iE) = sumErr/self.nObs;
			self.log.lRate(iE) = self.lRate;
			
			if self.verbose
				self.printProgress(sumErr,iE,jB);
			end

			% SAVE?
			if iE > 1
				if ~mod(iE, self.saveEvery) & ~isempty(self.saveFold)
					r = self;
					if ~exist(r.saveFold,'dir')
						mkdir(r.saveFold);
					end
					save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',iE)),'r'); clear r;
				end
			end

			% CLEAN UP IF USING INTERACTIVE TRAINING
			if iE >= self.nEpoch
				if self.interactive
					clear global r
				end
				break
			else % STORE GLOBAL COPY FOR INTERACTIVE TRAINING
				if self.interactive
					global r
					r = self;
				end
			end
			iE = iE + 1;
		end
		
		% PULL DATA FROM GPU, IF NEEDED
		% USER WILL NEED TO IMPLEMENT gpuGather.m
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
			self.gpuDevice = [];
		end
		fprintf('\n');
	end

	function self = runGibbs(self,X,targets)
	% r = runGibbs(X,targets)
	%--------------------------------------------------------------------------
	% Draw MCMC samples from the current model via Gibbs sampling.
	%--------------------------------------------------------------------------
	% INPUT:
	%       <X>:  - minibatch data.
	% <targets>:  - possible categorical targets. if no categorization,
	%               <targets> should be empty ([]).
	%
	% OUTPUT:
	%    <r>:  - RBM object with updated states
	%--------------------------------------------------------------------------
		nObs = size(X,1);
		iC = 1;
		% GO UP
		self = self.hidGivVis(X,targets,1);
		% LOG INITIAL STATES FOR GRADIENT CALCULATION
		self.pHid0 = self.pHid;
		self.aHid0 = self.aHid;
		
		while 1
			% GO DOWN
			self = self.visGivHid(self.aHid,self.sampleVis);
			X = self.aVis;

			% GO BACK UP
			if iC >= self.nGibbs
				self = self.hidGivVis(self.aVis,targets,0);
				break
			else
				self = self.hidGivVis(self.aVis,targets,1);
			end
			iC = iC + 1;
		end
	end

	function self = hidGivVis(self,X,targets,sampleHid)
	% r = hidGivVis(X,targets,[sampleHid])
	%--------------------------------------------------------------------------
	% Update hidden unit probabilities and states conditioned on the current
	% states of the visible units.
	%--------------------------------------------------------------------------
	% INPUT:
	%         <X>:  - batch data.
	%   <targets>:  - possible target variables.
	% <sampleHid>:  - flag indicating to sample the states of the hidden units.
	%
	% OUTPUT:
	%      <r>:  - RBM object with updated hidden unit probabilities/states.
	%--------------------------------------------------------------------------
		hidBias = self.c;
		if strcmp(self.inputType,'multinomial')
			% WEIGHT BIASES BY DOCUMENT LENGTH
			hidBias = bsxfun(@times,self.docLen,hidBias);
		end
		
		if self.classifier
			% JOINTLY MODEL MULTINOMIAL OVER INPUT CLASSES
			pHid = self.sigmoid(bsxfun(@plus,X*self.W + targets*self.classW ,hidBias));
		else
			pHid = self.sigmoid(bsxfun(@plus,X*self.W, hidBias));
		end
		
		if self.rlu % RECTIFIED LINEAR UNITS 
			self.aHid = max(0,pHid + randn(size(pHid)).*sqrt(self.sigmoid(pHid)));
		else
			if sampleHid
				self.aHid = single(pHid>rand(size(X,1),self.nHid));
			else
				self.aHid = pHid;
			end
		end
		self.pHid = pHid;

		% ENDUCE TOPOGRAPHY?
		if ~isempty(self.topoMask)
			for iH = 1:size(self.aHid,1)
				self.aHid(iH,:) = sum(bsxfun(@times,self.topoMask,self.aHid(iH,:)),2);
			end
		end
		% DEAL WITH DROPOUT
		self.aHid = self.aHid.*self.doMask;
	end
	
	function self = visGivHid(self,aHid,sampleVis)
	% r = hidGivVis(aHid,[sampleVis])
	%--------------------------------------------------------------------------
	% Update visible unit states conditioned on the current states of the hidden 
	% units.
	%--------------------------------------------------------------------------
	% INPUT:
	%      <aHid>:  - current hidden unit states (activations)
	% <sampleVis>:  - flag indicating to sample the states of the visible units.
	%
	% OUTPUT:
	%      <r>:  - RBM object with updated visible unit probabilities/states.
	%--------------------------------------------------------------------------
		nObs = size(aHid,1);
		switch self.inputType
			case 'binary'
				pVis = self.sigmoid(bsxfun(@plus,aHid*self.W',self.b));
				if sampleVis
					self.aVis = pVis>rand(nObs,self.nVis);
				else
					self.aVis = pVis;
				end

			case 'gaussian'
				mu = bsxfun(@plus,aHid*self.W',self.b);
				
				if sampleVis
					self.aVis = self.drawNormal(mu);					
				else
					self.aVis = mu;
				end
				
			case 'multinomial'
				pVis = self.softMax(bsxfun(@plus,aHid*self.W',self.b));
				self.aVis = zeros(size(pVis));
				for iO  = 1:nObs
					% DRAW D SEPARATE MULTINOMIALS FOR EACH INPUT
					self.aVis(iO,:) = mnrnd(self.docLen(iO),pVis(iO,:));
				end
		end

		if self.classifier
			self.pClass = self.softMax(bsxfun(@plus,self.aHid*self.classW',self.d));
			self.aClass = self.sampleClasses(self.pClass);
		end
	end
	
	function self = updateParams(self,X,targets);
	% self = updateParams(X,targets)
	%--------------------------------------------------------------------------
	% Update current model parameters based on the states of hidden and visible
	% units. Weight decay is applied here.
	%--------------------------------------------------------------------------
	% INPUT:
	%       <X>:  - minibatch data.
	% <targets>:  - possible categorical targets. if no categorization,
	%               <targets> should be empty ([]).
	%
	% OUTPUT:
	%    <self>:  - RBM object with updated states
	%--------------------------------------------------------------------------
		nObs = size(X,1);
		
		dW = (X'*self.pHid0 - self.aVis'*self.pHid)/nObs;
		self.dW = self.momentum*self.dW + ... % MOMENTUM
			          (1-self.momentum)*dW;         % NEW GRADIENT

		self.W = self.W + self.lRate*self.dW;

		% WEIGHT DECAY
		if self.wPenalty > 0 % L2 
			self.W = self.W - self.lRate*self.wPenalty*self.W;
		elseif self.wPenalty < 0 % L1 SUBGRADIENT AT 0
			self.W = self.W + self.lRate*self.wPenalty*sign(self.W);
		end

		db = mean(X) - mean(self.aVis);
		dc = mean(self.pHid0) - mean(self.pHid);

		self.db = self.momentum*self.db + self.lRate*db;
		self.b = self.b + self.db;

		self.dc = self.momentum*self.dc + self.lRate*dc;
		self.c = self.c + self.dc;

		if self.classifier
			% CLASSIFIER WEIGHTS
			dClassW=(targets'*self.pHid0 - self.aClass'*self.pHid)/nObs;
			self.dClassW=self.momentum*self.dClassW+self.lRate*(dClassW-self.wPenalty*self.classW);
			self.classW = self.classW + self.dClassW;
			
			% CLASSIFIER BIASES
			dd = (sum(targets) - sum(self.aClass))/nObs;
			self.dd = self.momentum*self.dd + self.lRate*dd;
			self.d = self.d + self.dd;
		end
	end

	function err = accumErr(self,X,err0);
	% err = updateParams(X,err0)
	%--------------------------------------------------------------------------
	% Add reconstruction error (squared difference) for the current batch to
	% the total error.
	%--------------------------------------------------------------------------
	% INPUT:
	%       <X>:  - minibatch data.
	%    <err0>:  - current resevoir of error
	%
	% OUTPUT:
	%     <err>:  - updated error resevoir
	%--------------------------------------------------------------------------
		err = sum(sum((X-self.aVis).^2));
		err = err + err0;
	end

	function self = init(self,arch)
	% r = init(arch)
	%--------------------------------------------------------------------------
	% Initialize an rbm object based on provided architecture, <arch>.
	%--------------------------------------------------------------------------
	% INPUT:
	%    <arch>:  - a structure of architecture options. Possible fields are:
	%               .size       --> [#Visible x #Hidden] units
	%               .inputType  --> string indicating input type (i.e. 'binary',
	%                              'gaussian','multinomial')
	%               .classifyer --> flag indicating whether to train a class-
	%                               ifier model in parallel.
	%               .opts       --> additional cell array of options, defined
	%                               in argument-value pairs.
	%
	% OUTPUT:
	%     <r>:  - an initialized RBM object.
	%--------------------------------------------------------------------------
		arch = self.ensureArchitecture(arch);
		
		self.nVis = arch.size(1);
		self.nHid = arch.size(2);
		self.inputType = arch.inputType;
		self.classifier = arch.classifier;

		% PARSE ANY ADDITIONAL OPTIONS
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
		
		% INIT. WEIGHTS (A' LA BENGIO)
		range = sqrt(6/(2*self.nVis));
		self.W = single(2*range*(rand(self.nVis,self.nHid)-.5));
		
		self.dW = zeros(size(self.W),'single');
		self.b = zeros(1,self.nVis,'single');
		self.db = zeros(size(self.b),'single');
		self.c = zeros(1,self.nHid,'single');
		self.dc = zeros(size(self.c),'single');
	end

	function arch = ensureArchitecture(self,arch)
	% arch = ensureArchitecture(self,arch)
	%--------------------------------------------------------------------------
	% Preprocess the provided architecture structure <arch>.
	%--------------------------------------------------------------------------
	% INPUT:
	% <arch>:  - is either a [2 x 1] vector giving the [#vis x # hid], in which
	%            case we use the default model parameters, or it is a strucure
	%            with the fields:
	%             .size                  --> Network size; [#Vis x # Hid]
	%             .inputType (optional)  --> ['binary'] or 'gaussian'
	%             .classifier (optional) --> true or [false]
	%             .opt (optional)        --> a cell array of {'parameter',paramValue}
	%                                        of global options
	%--------------------------------------------------------------------------
		% IF arch IS GIVEN AS A VECTOR 
		if ~isstruct(arch),arch.size = arch; end

		if ~isfield(arch,'size');
			error('need to supply the # of Visible and Hidden units')
		end
		
		if ~isfield(arch,'inputType');
			arch.inputType = self.inputType;
		end
		
		if ~isfield(arch,'classifier');
			arch.classifier = self.classifier;
		end
		
		if isfield(arch,'opts');
			opts = arch.opts;
			if ~iscell(opts) || mod(numel(opts),2)
				error('arch.opts must be a cell array of string-value pairs.')
			end
		end
	end

	function [self,targets] = initClassifer(self,targets);
	% [self,targets] = initClassifer(targets);
	%--------------------------------------------------------------------------
	% Initialize classifier units based on form of <targets>. Can also be used
	% to preprocess targets to be 1-of-K.
	%--------------------------------------------------------------------------

		% IF SUPPLIED TARGETS ARE A LIST OF LABELS
		if isvector(targets)
			targets = self.oneOfK(targets);
		end
		self.nClasses = size(targets,2);
		self.classW = single(0.1*randn(self.nClasses,self.nHid));
		self.dClassW = zeros(size(self.classW),'single');
		self.d = zeros(1,self.nClasses,'single');
		self.dd = self.d;
	end

	function batchIdx = createBatches(self,X)
	% batchIdx = createBatches(X)
	%--------------------------------------------------------------------------
	% Create minibatches based on dimensions of inputs <X>. Returns a cell
	% array with each entry giving the indices in to the rows of <X> for a
	% single batch.
	%--------------------------------------------------------------------------
		nObs = size(X,1);
		nBatches = ceil(nObs/self.batchSz);
		tmp = repmat(1:nBatches, 1, self.batchSz);
		tmp = tmp(1:nObs);
		randIdx=randperm(nObs);
		tmp = tmp(randIdx);
		for iB=1:nBatches
		    batchIdx{iB} = find(tmp==iB);
		end
	end

	function visLearning(self);
	% visLearning(self);
	%--------------------------------------------------------------------------
	% Deal with any visualiztions. Note, must set self.visFun to appropriate
	% visualization function handle.
	%--------------------------------------------------------------------------
		if ~isempty(self.visFun)
			try
				self.visFun(self);
			catch
				fprintf('\nVisualization failed...')
			end
		end
	end

	function printProgress(self,sumErr,iE,jB)
	% printProgress(sumErr,iE,jB)
	%--------------------------------------------------------------------------
	% Utility function to display std out.
	%--------------------------------------------------------------------------
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

	function E = hidExpect(self,X);
	% E = hidExpect(X);
	%--------------------------------------------------------------------------
	% Calculate hidden unit expectations for network input <X>
	%--------------------------------------------------------------------------
		switch self.inputType
		case 'multinomial'
			docLen = sum(X,2);
			E = self.sigmoid(bsxfun(@plus,X*self.W,bsxfun(@times,docLen,self.c)));
		otherwise
			if self.rlu
 				% E = max(0,bsxfun(@plus,X*self.W,self.c));
				E = max(0,X*self.W);
				E = bsxfun(@minus,E,mean(E));
				E = bsxfun(@rdivide,E,std(E));
			else
				E = self.sigmoid(bsxfun(@plus,X*self.W,self.c));
			end
		end
	end
	
	function samps = sample(self,X0,nSamples,nSteps)
	% samps = sample(X0,[nSamples],[nSteps])
	%--------------------------------------------------------------------------
	% Draw sample(s) from model using a Markov chain.
	%--------------------------------------------------------------------------
	% INPUT:
	%       <X0>:  - initial state from which to begin the Markov chain
	% <nSamples>:  - the number of individual samples to generate. default is
	%                to draw a single sample.
	%   <nSteps>:  - the number of transitions/steps to take in the Markov
	%                chain. default is to take 50 steps.
	%--------------------------------------------------------------------------
	
		self.sampleVis = 0;
		self.sampleHid = 0;
		if notDefined('nSamples');nSamples = 1;end
		if notDefined('nSteps'),nSteps = 50; end
		
		[nObs,nVis] = size(X0);
		samps = zeros(nObs,nVis,nSamples);

		if self.useGPU
			samps = gpuArray(single(samps));
			self = gpuDistribute(self);
		end
		
		for iS = 1:nSamples
			vis = X0;
			for iI = 1:nSteps
				hid = self.sigmoid(bsxfun(@plus,binornd(1,vis)* ...
									self.W,self.c));
				switch self.inputType
				case 'binary'
					vis = self.sigmoid(bsxfun(@plus,binornd(1,hid)* ...
					self.W',self.b));
				case 'gaussian'
					vis = bsxfun(@plus,hid*self.W',self.b);
					
				case 'multinomial'
					vis = bsxfun(@plus,hid*self.W',self.b)
					pVis = softMax(vis);
					% ENSURE PDF
					pVis = bsxfun(@rdivide,eVis,sum(eVis,2));
					vis = zeros(size(pVis));
					docLen = sum((find(pVis)),2);
					for iO  = 1:nObs
						% DRAW D SEPARATE MULTINOMIALS FOR EACH INPUT
						vis(iO,:) = mnrnd(docLen(iO),pVis(iO,:));
					end
				end
			end
			samps(:,:,iS) = vis;
		end
		
		if self.useGPU
			samps = gather(samps);
		end
	end

	function F = freeEnergy(self,X)
	% F = freeEnergy(X)
	%--------------------------------------------------------------------------
	% Calculate model free energy for an input <X>. Currently only available for
	% binary and gaussian inputs and binary hidden units.
	%--------------------------------------------------------------------------
	% INPUT:
	%  <X>:  - network input.
	%
	% OUTPUT:
	%  <F>:  - free energy, summing over all binary hidden units.
	%--------------------------------------------------------------------------
	 
		nSamps = size(X,1);
		upBound = 50; loBound = -50; % FOR POSSIBLE OVERFLOW
		switch self.inputType
			case 'binary'
				H = ones(nSamps,1)*self.c + X*self.W;
				H = max(min(H,upBound),loBound);
				% SAMPLE ENERGIES
				sampE = -X*self.b' - sum(-log(1./exp(H)),2);
				F = mean(sampE);
			case 'gaussian'
				H = ones(nSamps,1)*self.c + X*self.W;
				H = max(min(H,upBound),loBound);
				% SAMPLE ENERGIES
				sampE = bsxfun(@minus,X,self.b);
				sampE = sum(sampE.^2,2)/2;
				sampE = sampE - sum(-log(1./(1+exp(H))),2);
				F = mean(sampE);
			case 'multinomial'
				error('need to implement multinomial free energy')
		end
	end

	function targets = oneOfK(self,labels)
	% targets = oneOfK(labels)
	%--------------------------------------------------------------------------
	% Create a 1-of-k representation of a set of labels, where <labels> is a
	% vector of integer class labels.
	%--------------------------------------------------------------------------
		classes = unique(labels);
		targets = zeros(numel(labels),max(classes));
		for iL = 1:numel(classes)
			targets(labels==classes(iL),iL)=1;
		end
	end


	function p = sigmoid(self,X)
	% p = sigmoid(X)
	%--------------------------------------------------------------------------
	% Sigmoid activation function
	%--------------------------------------------------------------------------
		if self.useGPU
			p = arrayfun(@(x)(1./(1 + exp(-x))),X);
		else
			p = 1./(1 + exp(-X));
		end
	end

	function p = drawNormal(self,mu);
	% p = drawNormal(mu);
	%--------------------------------------------------------------------------
	% Draw samples from a multivariate normal  with mean <mu> and identity
	% covariance.
	%--------------------------------------------------------------------------
		% ASSUMES UNIT VARIANCE OF ALL VISIBLES
		% (I.E. YOU SHOULD STANDARDIZE INPUTS)
		p = mvnrnd(mu,ones(1,self.nVis));
	end

	function c = softMax(self,X)
	% c = softMax(X)
	%--------------------------------------------------------------------------
	% Siftmax activation function for input X. Returns a vector of category
	% probabilities.
	%--------------------------------------------------------------------------
		c = bsxfun(@rdivide,exp(X),sum(exp(X),2));
	end

	function classes = sampleClasses(self,pClass)
	%--------------------------------------------------------------------------
	% classes = sampleClasses(pClass)
	%--------------------------------------------------------------------------
	% Sample class labels <classes> given a set of class probabilities, <pClass>
	%--------------------------------------------------------------------------
		[nObs,nClass]=size(pClass);
		classes = zeros(size(pClass));
		% ENSURE NORMALIZED
		pClass = bsxfun(@rdivide,pClass,sum(pClass,2));
		for iC = 1:nObs
			probs = pClass(iC,:);
			samp = cumsum(probs)>rand;
			idx = min(find(max(samp)==samp));
			classSamp = zeros(size(probs));
			classSamp(idx) = 1;
			classes(iC,:) = classSamp;
		end
	end
	
	function [pred,error,misClass] = classify(self,X,targets)
	% [pred,error,misClass] = classify(X,[targets])
	%--------------------------------------------------------------------------
	% Classify inputs <X> and calculate misclassification and error when comp-
	% ared to <targets>.
	%--------------------------------------------------------------------------
	% INPUT:
	%        <X>:  - a set of inputs to the network.
	%  <targets>:  - (optional). a set of target classes. Can be a 1-of-K vector
	%                or can a vector with equal length to the number of inputs
	%                where each entry is an interger class label.
	% OUTPUT:
	%     <pred>:  - predicted class label.
	%    <error>:  - the classification error, based on provided <targets>
	% <misClass>:  - the indices of the inputs that were misclassified.
	%--------------------------------------------------------------------------
	
		if notDefined('targets'),
			targets = [];
		else
			if isvector(targets);
				targets = self.oneOfK(targets);
			end
		end

		nObs = size(X,1);

		% ACTIVATE HIDDEN UNITS USING INPUT ONLY
		pHid = self.sigmoid(bsxfun(@plus,X*self.W,self.c));

		% CALCULATE CLASS PROBABILITY
		pClass = self.softMax(bsxfun(@plus,pHid*self.classW',self.d));
		% WINNER-TAKE-ALL CLASSIFICATION
		[~, pred] = max(pClass,[],2);
		if ~notDefined('targets') && nargout > 1
			[~,targets] = max(targets,[],2);

			% CALCULATE MISSCLASSIFICATION RATE
			misClass = find(pred~=targets);
			error = numel(misClass)/nObs;
		end
	end
end % END METHODS
end % END CLASSDEF
