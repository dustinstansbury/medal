classdef rbm
% Restricted Boltzmann Machine Model object:
%----------------------------------------------------------------------------
% Initialize and train a an RBM energy-based model. Supports binary, Gaussian,
% and Replicated Softmax inputs.
%
% Supports joint modeling of multinomial variables for classification.
%
% Model Regularizers include L2 weight decay, hidden unit sparsity, and hidden
% unit dropout.
%----------------------------------------------------------------------------
% DES
% stan_s_bury@berkeley.edu

properties
	class = 'rbm'; 		% GENERAL CLASS OF MODEL
	inputType = 'binary';% TYPE OF RBM ('BB','GB')
	classifier = false;  % TRAIN MULTINOMIAL UNITS FOR CLASSIFICATION IN PARALLEL
	nClasses;			% # OF OUTPUT CLASSES (.classifier = true)
	nObs;				% # OF TRAINING OBSERVATIONS
	nVis;				% # OF VISIBLE UNITS (DIMENSIONS)
	nHid = 100;			% # OF HIDDEN UNITS
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
	pVis;				% VISIBLE LAYER PROBS
	aHid;				% HIDDEN LAYER ACTIVATION
	pHid;				% HIDDEN LAYER PROBS
	pHid0;				% INITIAL HIDDEN LAYER PROBS
	aHid0;				% INITIAL HIDDEN LAYER ACTIVATION
	eHid;				% EXPECTATION OF HIDDEN STATE (POST TRAINING)
	pClass;				% PROBABILITIES OF CLASSIFIER UNITS (.classifier = 1)
	aClass;				% ACTIVATION OF CLASSIFIER UNITS (.classifier = 1)
	lRate = 0.1;		% DEFAULT LEARNING RATE
	batchIdx = [];		% BATCH INDICES INTO TRAINING DATA
	sampleVis = 0;		% SAMPLE THE VISIBLE UNITS
	sampleHid = 1;		% SAMPLE HIDDEN UNITS 
	momentum = 0;		% DEFAULT MOMENTUM TERM 
	nEpoch = 100;		% # OF FULL PASSES THROUGH TRIANING DATA
	wPenalty = 0;		% CURRENT WEIGHT PENALTY
	sparsity = 0;		% SPARSENESS FACTOR
	dropout = 0;		% HIDDEN UNIT DROPOUT
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
	saveFold='./rbmSave';% # DEFAULT SAVE FOLDDER
		
end % END PROPERTIES

methods
	% CONSTRUCTOR
	function self = rbm(arch)
		if notDefined('arch')
		else
			self = self.init(arch);
		end
	end

	function [] = print(self)
		properties(self)
		methods(self)
	end
	
	function self = train(self,X,targets)
	% Train an RBM using Contrastive Divergence
	% <X> is [#Obs x #Vis]
	% <targets> (optional) is either [#Obs x #Class] as 1 of K representation
	%           or [#Obs x 1], where each entry is a numerical category label
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
				self.lRate = max(self.lRate/max(1,iE/self.beginAnneal),1e-8);
			end
			
			% WEIGHT DECAY?
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
			end

			% SPARSITY
			if self.sparsity
				dcSparse = -self.lRate*self.sparseGain*(mean(self.pHid)-self.sparsity);
				self.c = self.c + dcSparse;
			end
			
			self.log.err(iE) = sumErr;
			self.log.lRate(iE) = self.lRate;
			
			if self.verbose
				self.printProgress(sumErr,iE,jB);
			end
			
			if iE > 1
				if ~mod(iE, self.saveEvery)
					r = self;
					if ~exist(r.saveFold,'dir')
						mkdir(r.saveFold);
					end
					save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',iE)),'r'); clear r;
				end
			end
			if iE >= self.nEpoch
				break
			end
			iE = iE + 1;
		end
		% PULL DATA FROM GPU, IF NEEDED
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
			self.gpuDevice = [];
		end
		fprintf('\n');
	end

	function self = runGibbs(self,X,targets)
	% MAIN GIBBS SAMPLER
		nObs = size(X,1);
		iC = 1;
		while 1
			% GO UP
			if iC == 1
				self = self.hidGivVis(X,targets,1);
				% LOG INITIAL STATES FOR GRADIENT CALCULATION 
				self.pHid0 = self.pHid;
				self.aHid0 = self.aHid;
			else
				self = self.hidGivVis(X,targets,0);
			end
			% GO DOWN
			self = self.visGivHid(self.aHid,self.sampleVis);
			X = self.aVis;
		
			% FINISH 
			if iC >= self.nGibbs
				self = self.hidGivVis(self.aVis,targets,0);
				break
			end
			iC = iC + 1;
		end
	end

	function self = hidGivVis(self,X,targets,sampleHid)
	% p(H|V), SAMPLE H, IF NEEDED
		if self.classifier
			pHid = self.sigmoid(bsxfun(@plus,X*self.W + targets*self.classW ,self.c));
		else
			pHid = self.sigmoid(bsxfun(@plus,X*self.W, self.c));
		end

		if sampleHid
			self.aHid = single(pHid>rand(size(X,1),self.nHid));
		else
			self.aHid = pHid;
		end
		
		% DROP OUT HIDDEN UNITS RANDOMLY
		if self.dropout
			self.aHid = self.aHid.*(rand(size(self.aHid))>self.dropout);
		end
		self.pHid = pHid;
	end
	
	function self = visGivHid(self,aHid,sampleVis)
	% p(V|H)
		nObs = size(aHid,1);
		switch self.inputType
			case 'binary'
				pVis = self.sigmoid(bsxfun(@plus,aHid*self.W',self.b));
				if sampleVis
					self.aVis = pVis>rand(nObs,self.nVis);
				else
					self.aVis = pVis;
				end
				self.pVis = pVis;

			case 'gaussian'
				mu = bsxfun(@plus,aHid*self.W',self.b);
				self.pVis = self.drawNormal(mu);
				
				if sampleVis
					self.aVis = self.pVis;
				else
					self.aVis = mu;
				end
		end

		if self.classifier
			self.pClass = self.softMax(bsxfun(@plus,self.aHid*self.classW',self.d));
			self.aClass = self.sampleClasses(self.pClass);
		end
	end
	
	function self = updateParams(self,X,targets);
	% LEARNING RULES
		nObs = size(X,1);
		
		dW = (X'*self.pHid0 - self.aVis'*self.pHid)/nObs;
		self.dW=self.momentum*self.dW + ... % MOMENTUM
		        (1-self.momentum)*dW - ...  % NEW GRADIENT
		        self.wPenalty*self.W;         % WEIGHT DECAY
		self.W = self.W + self.lRate*self.dW;

		db = mean(X) - mean(self.pVis);
		self.db = self.momentum*self.db + self.lRate*db;
		self.b = self.b + self.db;

		dc = mean(self.pHid0) - mean(self.pHid);
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
	% ACCUMULATE SQUARED RECONSTRUCTION ERROR
		err = sum(sum((X-self.aVis).^2));
		err = err + err0;
	end

	function self = init(self,arch)
	% PARSE ARGUMENTS/OPTIONS
		arch = self.ensureArchitecture(arch);
		
		self.nVis = arch.size(1);
		self.nHid = arch.size(2);
		self.inputType = arch.inputType;
		self.classifier = arch.classifier;

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
		
		switch self.inputType
		case 'gaussian'
			self.W = (2/(self.nHid+self.nVis))*rand(self.nVis,self.nHid) - ...
			1/(self.nVis + self.nHid);
			
		case 'binary'
			self.W = 1/sqrt(self.nVis +  self.nHid)* ...
			2*(rand(self.nVis,self.nHid)-.5);
		end
		
		self.dW = zeros(size(self.W));
		self.b = zeros(1,self.nVis);
		self.db = zeros(size(self.b));
		self.c = zeros(1,self.nHid);
		self.dc = zeros(size(self.c));
	end

	function arch = ensureArchitecture(self,arch)
	% PREPROCESS A SUPPLIED ARCHITECTURE.
	% <arch> IS EITHER A [2 X 1] VECTOR GIVING THE [#Vis x # Hid],
	%        IN WHICH CASE WE USE THE DEFAULT MODEL PARAMETERS, OR
	%        IT IS A STRUCURE WITH THE FIELDS
	%             .size -- Network size; [#Vis x # Hid]
	%             .inputType (optional) -- ['binary'] or 'gaussian'
	%             .classifier (optional) -- true or [false]
	%             .opt (optional)   -- Cell array of {'parameter',paramValue} of
	%                                  global options
	
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
	% INITIALIZE CLASSIFIER UNITS, IF APPLICABLE

		% IF SUPPLIED TARGETS ARE A LIST OF LABELS
		if isvector(targets)
			targets = self.oneOfK(targets);
		end
		self.nClasses = size(targets,2);
		self.classW = 0.1*randn(self.nClasses,self.nHid);
		self.dClassW = zeros(size(self.classW));
		self.d = zeros(1,self.nClasses);
		self.dd = self.d;
	end

	function batchIdx = createBatches(self,X)
	% CREATE MINIBATCHES
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

	function p  = sigmoid(self,X)
	% SIGMOID ACTIVATION FUNCTION
		if self.useGPU
			p = arrayfun(@(x)(1./(1 + exp(-x))),X);
		else
			p = 1./(1 + exp(-X));
		end
	end

	function p = drawNormal(self,mu);
	% DRAW FROM A MULTIVARIATE NORMAL
	
		% ASSUMES UNIT VARIANCE OF ALL VISIBLES
		% (I.E. YOU SHOULD STANDARDIZE INPUTS)
		p = mvnrnd(mu,ones(1,self.nVis));
	end

	function visLearning(self,iE,jB);
	% VISUALIZATIONS
		
		if isempty(self.visFun)
			switch self.inputType
				case 'binary'
					visBinaryRBMLearning(self);
				case 'gaussian'
					visGaussianRBMLearning(self);
			end
		else
			self.visFun(self);
		end
	end

	function [] = printProgress(self,sumErr,iE,jB)
	% VERBOSE
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
	% CALCULATE HIDDEN UNIT EXPECTATIONS
		switch self.inputType
		case 'binary'
			E = self.sigmoid(bsxfun(@plus,X*self.W,self.c));
		case 'gaussian'
			E = bsxfun(@plus,X*self.W,self.c);
		end
	end
	
	function samps = sample(self,data,nSamples,nIters)
	% DRAW SAMPLE FROM MODEL USING GIBBS SAMPLING
	
		self.sampleVis = 0;
		self.sampleHid = 0;
		if notDefined('nSamples');nSamples = 1;end
		if notDefined('nIters'),nIters = 50;end
		
		[nObs,nVis] = size(data);
		samps = zeros(nObs,nVis,nSamples);

		if self.useGPU
			samps = gpuArray(single(samps));
			self = gpuDistribute(self);
		end
		
		for iS = 1:nSamples
			vis = data;
			for iI = 1:nIters
				hid = self.sigmoid(bsxfun(@plus,binornd(1,vis)* ...
									self.W,self.c));
				switch self.inputType
				case 'binary'
					vis = self.sigmoid(bsxfun(@plus,binornd(1,hid)* ...
					self.W',self.b));
				case 'gaussian'
					vis = bsxfun(@plus,hid*self.W',self.b);
				end
			end
			samps(:,:,iS) = vis;
		end
		
		if self.useGPU
			samps = gather(samps);
		end
	end

	function [F] = freeEnergy(self,X)
	% CALCULATE MODEL FREE ENERGY FOR AN INPUT X
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
		end
	end

	function targets = oneOfK(self,labels)
	% CREATE A 1-OF-K REPRESENTATION OF A SET OF LABELS
		classes = unique(labels);
		targets = zeros(numel(labels),max(classes));
		for iL = 1:numel(classes)
			targets(labels==classes(iL),iL)=1;
		end
	end

	function c = softMax(self,X)
	% SOFT MAX CLASSIFICATION FUNCTION
		c = bsxfun(@rdivide,exp(X),sum(exp(X),2));
	end

	function classes = sampleClasses(self,pClass)
	% SAMPLE CLASS LABELS GIVEN A SET OF PROBABILITIES
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
	% CLASSIFY
	
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