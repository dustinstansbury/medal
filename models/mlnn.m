classdef mlnn
% Multilayer Neural Network model object.
%------------------------------------------------------------------------------
% Initialize, train, and test a multilayer neural network.
%
% Supports multiple activation functions including linear, sigmoid, tanh,
% exponential, and soft rectification (softplus).
%
% Supported cost functions include mean squared error (mse), binary (xent)
% and multi-class (mcxent) cross-entropy, and exponential log likelihood
% (expll).
%
% Supports multiple regularization techniques including weight decay,
% hidden unit dropout, and early stopping. Can be used to train standard
% and denoising autoencoders .
%------------------------------------------------------------------------------
% DES
% stan_s_bury@berkeley.edu

% TO DO:
% - ADD DELAYINPUTS METHOD

properties
	class = 'mlnn';
	arch					% NETWORK ARCHITECTURE (STRUCT)
	nLayers;				% # OF UNIT LAYERS
	layers ;				% LAYER STRUCTS

	netOutput = [];			% CURRENT OUTPUT OF THE NETWORK
	netError;				% CURRENT NETWORK OUTPUT ERROR
	
	costFun = 'mse';		% COST FUNCTION
	xValCostFun = 'mse';	% COST FUNCTION FOR EARLY STOPPING
	J = [];					% CURRENT COST FUNCTION VALUE
	
	nEpoch = 10;			% TOTAL NUMBER OF TRAINING EPOCHS
	epoch = 1;				% CURRENT EPOCH

	inputDelays = 0;		% VECTOR OF INPUT TEMPORAL DELAYS (0=NO DELAY)
	batchSize = 20;			% SIZE OF MINIBATCHES
	trainBatches = [];		% BATCH INDICES
	xValBatches = [];		% XVAL. DATA INDICES
	
	trainCost = [];			% TRAINING ERROR HISTORY (ALL DATA)
	xValCost = [];			% XVALIDATION EROR HISTORY
	bestxValCost = Inf		% TRACK BEST NETWORK ERROR

	nXVal = 0;				% PORTION OF DATA HELD OUT FOR XVALIDATION
	stopEarly = 5;			% # OF EARLY STOP EPOCHS (XVAL ERROR INCREASES)
	stopEarlyCnt = 0;		% EARLY STOPPING CRITERION COUNTER
	testingNow 				% ARE WE TESTING THE NETWORK NOW?
	bestNet = [];			% STORAGE FOR BEST NETWORK
	
	wPenalty = 0.001;		% WEIGHT DECAY TERM
	momentum = .9;			% MOMENTUM
	normGrad = 1;
	maxNorm=Inf;			% RENORMALIZE WEIGHTS WHOSE NOR EXCEED maxNorm

	sparsity = 0;			% TARGET SPARSITY (SIGMOID ONLY)
	sparseGain = 1;			% GAIN ON LEARNING RATE FOR SPARSITY (SIGMOID ONLY)
	dropout = 0;			% PROPORTION OF HIDDEN UNIT DROPOUT 
	denoise = 0;			% PROPORTION OF VISIBLE UNIT DROPOUT 

	beginAnneal=Inf;		% # OF EPOCHS AFTER WHICH TO BEGIN SIM. ANNEALING
	beginWeightDecay=1;		% # OF EPOCHS AFTER WHICH TO BEGIN WEIGHT DECAY
	
	saveEvery = 1e100;		% SAVE PROGRESS EVERY # OF EPOCHS
	saveDir;
	displayEvery = 10;		% VISUALIZE EVERY # OF EPOCHS
	visFun;					% VISUALIZATION FUNCTION HANDLE
	trainTime = Inf;		% TRAINING DURATION
	verbose = 500;			% DISPLAY THIS # OF WEIGHT UPDATES
	auxVars = [];			% AUXILIARY VARIABLES (VISUALIZATION, ETC)
	interactive = 0;		% STORE A GLOBAL COPY FOR INTERACTIVE TRAINING
	useGPU = 0;				% USE THE GPU, IF AVAILABLE
	gpuDevice = [];			% STORAGE FOR GPU DEVICE
	checkGradients = 0;		% NUMERICAL GRADIENT CHECKING
end

methods
	function self = mlnn(arch)
	% net = mlnn(arch)
	%--------------------------------------------------------------------------
	%mlnn constructor method. Initilizes a mlnn object, <net> given a user-
	%provided architecture, <arch>.
	%--------------------------------------------------------------------------
		self = self.init(arch);
	end

	function print(self)
	%print()
	%--------------------------------------------------------------------------
	%Print properties and methods for mlnn object.
	%--------------------------------------------------------------------------
		properties(self)
		methods(self)
	end

	function self = init(self,arch)
	%net = init(arch)
	%--------------------------------------------------------------------------
	%Utility function to used intitialize a neural network given an architecture
	%<arch> is a struct with required fields:
	%	.size   --  [#input, #hid1, ..., #hidN, #output]; SIZE [1 x nUnits]
	%	.actFun -- SIZE [1 x (nUnits - 1)]
	%	.lRate -- SIZE [1 x (nUnits -1)]
	%
	% <arch> can also contain an options fields <.opt>, which is a cell array
	% of property-value pairs.
	%
	% Returns a mlnn object, <net>.
	%--------------------------------------------------------------------------
		arch = self.ensureArchitecture(arch);

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
			arch = rmfield(arch,'opts');
		end
		self.arch = arch;
		self.nLayers = numel(arch.size);
		self.layers{1}.type = 'input';
		% INITIALIZE LAYER WEIGHTS AND BIASES
	    for lL = 2:self.nLayers
		    self.layers{lL}.type = 'hidden';
		    % LEARNING RATES
			self.layers{lL-1}.lRate = arch.lRate(lL-1);

			% WEIGHTS AND WEIGHT MOMENTA (A' LA BENGIO)
			range = sqrt(6/((arch.size(lL) + arch.size(lL - 1))));
			self.layers{lL - 1}.W = (rand(arch.size(lL),arch.size(lL-1))-.5)*2*range;
			self.layers{lL - 1}.pW = zeros(size(self.layers{lL-1}.W));

			% BIASES AND BIAS MOMENTA
			self.layers{lL - 1}.b = zeros(arch.size(lL), 1);
			self.layers{lL - 1}.pb = zeros(size(self.layers{lL - 1}.b));

			% ACTIVATION FUNCTION
			self.layers{lL}.actFun = arch.actFun{lL};
			
			% MEAN ACTIVATIONS (FOR SPARSITY)
			self.layers{lL}.meanAct = zeros(1, arch.size(lL));
		end
		self.layers{end}.type = 'output';

		% DISTRIBUTE VALUES TO THE GPU
		if self.useGPU
			self = gpuDistribute(self);
		end
	end

	function self = train(self, data, targets, batches)
	%net = train(data, targets, [batches])
	%--------------------------------------------------------------------------
	% Train a neural net using stochastic gradient descent. |data| = [#Obs x #Input],
	% |targets| = [#Obs x #Output]. <batches> is an optional set of user-suppled
	%  batch indices; can either be a struct with the fields .trainBatches and/or
	% .xValBatches, or a cell array array of batch indices (which wll be split
	% into training and cross-validation sets accordingly). Note that user-supplied
	% indices can contain input delays, where delays are along columns.
	%--------------------------------------------------------------------------
	
		% PRELIMS
		if notDefined('batches')
			self = self.makeBatches(data);
		else %IN CASE BATCHES SUPPLIED
			if iscell(batches)
				self = self.processBatches(batches);
			elseif isstruct(batches)
				self.trainBatches = batches.trainBatches;
				self.xValBatches = batches.xValBatches;
			end
		end

		self.trainCost = zeros(self.nEpoch,1);
		self.xValCost = zeros(self.nEpoch,1);
		nBatches = numel(self.trainBatches);
		tic; cnt = 1;
		
		if self.checkGradients
			checkNNGradients(self,data,targets);
		end

		wPenalty = self.wPenalty;
		% MAIN
	    while 1
			if self.verbose, self.printProgress('epoch'); end
			batchCost = zeros(nBatches,1);
			if self.epoch > self.beginAnneal
				for lL = 1:self.nLayers - 1
					self.layers{lL}.lRate = max(1e-10, ...
									        self.layers{lL}.lRate/ ...
									        max(1,self.epoch/self.beginAnneal));
				end
			end

			if self.epoch >= self.beginWeightDecay
				self.wPenalty = wPenalty;
			else
				self.wPenalty = 0;
			end

			for iB = 1:nBatches
				% GET BATCH DATA
				batchIdx = self.trainBatches{iB};
				netInput = self.getBatchInput(data,batchIdx);				
				netTargets = self.getBatchTargets(targets,batchIdx);

				% ADD BINARY NOISE TO INPUT (DENOISING AE)
				if self.denoise > 0
				    netInput = netInput.*(rand(size(netInput))>self.denoise);
				end

				% BACKPROP MAIN
				self = self.fProp(netInput, netTargets);
				self = self.bProp;
				self = self.updateParams;

				% ASSESS BATCH LOSS
				batchCost(iB) = self.J;

				% DISPLAY?
				if ~mod(cnt, self.displayEvery);
					self.auxVars.targets = netTargets;
					self.auxVars.W = self.layers{1}.W;
					self.visLearning;
				end
				cnt = cnt + 1;
			end
	        
	        % MEAN LOSS OVER ALL TRAINING POINTS
			self.trainCost(self.epoch) = mean(batchCost);
			
	        % CROSS-VALIDATION
			if ~isempty(self.xValBatches)
				self = self.crossValidate(data,targets);
				self = self.assessNet;
				if self.verbose, self.printProgress('xValCost');end
			end
			
			% SAVE CURRENT NETWORK
			if ~mod(self.epoch,self.saveEvery) & ~isempty(self.saveDir)
				self.save; 
			end
	        if self.verbose, self.printProgress('trainCost');end
	        
			% DONE?
			if (self.epoch >= self.nEpoch) || ...
				(self.stopEarlyCnt >= self.stopEarly),
				if ~isempty(self.bestNet)
					% self.layers = self.bestNet;
					self.bestNet = [];
				end
				self.trainCost = self.trainCost(1:self.epoch);
				self.xValCost = self.xValCost(1:self.epoch);
				
				if self.interactive
					clear global n
				end
				break;
			else
				if self.interactive
					global n
					n = self;
				end
			end
			self.epoch = self.epoch + 1;
		end
		self.trainTime = toc;

		% RETURN VALUES FROM THE GPU
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
			self.gpuDevice = [];
		end
	end
	
	function [self,out] = fProp(self, netInput, targets)
	%[net,out] = fProp(netInput, targets)
	%--------------------------------------------------------------------------
	%Forward propagation of input signals, <netInput>. Also updates state of
	%network cost, if provided with <targets>. Also returns network output
	%<out>, if requested.
	%--------------------------------------------------------------------------
		if notDefined('targets'), targets = []; end
		nObs = size(netInput, 1);
		self.layers{1}.act = netInput;

		for lL = 2:self.nLayers 

			% LAYER PRE-ACTIVATION
			preAct = bsxfun(@plus,self.layers{lL - 1}.act* ...
		                    self.layers{lL - 1}.W',self.layers{lL - 1}.b');

			% LAYER ACTIVATION
			self.layers{lL}.act = self.calcAct(preAct,self.layers{lL}.actFun);

			% DROPOUT
			if self.dropout > 0 & lL < self.nLayers
				if self.testingNow % REWEIGHT ACT. DURING TESTING IF USING DROPOUT
					self.layers{lL}.act = self.layers{lL}.act*(1-self.dropout);						
				else 
					self.layers{lL}.act = self.layers{lL}.act.*(rand(size(self.layers{lL}.act))>self.dropout);
				end
			end

			% MOVING AVERAGE FOR TARGET SPARSITY (SIGMOID ONLY)
			if strcmp('sigmoid',self.layers{lL}.actFun)
				if self.sparsity>0
					self.layers{lL}.meanAct = 0.9 * self.layers{lL}.meanAct + 0.1*mean(self.layers{lL}.act);
				end
			end
		end
		
		% COST FUNCTION & OUTPUT ERROR SIGNAL
		if ~isempty(targets)
			[self.J, self.netError] = self.cost(targets,self.costFun);
		end
		if nargout > 1
			out = self.layers{end}.act;
		end
	end

	function [J, dJ] = cost(self,targets,costFun)
	%[J, dJ] = cost(targets,costFun)
	%---------------------------------------------------------------------------
	%Calculate the output error <J> (and error gradients <dJ>) based on
	%available cost functions ('mse', 'expll', 'xent', 'mcxent', 'class',
	%'correlation'). Note that 'correlation'and 'class' do not provide
	%gradients.
	%--------------------------------------------------------------------------
		netOut = self.layers{end}.act;

		[nObs,nTargets] = size(netOut);
		switch costFun
			case 'mse' % MEAN SQUARED ERROR (LINEAR REGRESSION)
				delta = targets - netOut;
				J = 0.5*sum(sum(delta.^2))/nObs;
				dJ = -delta;

			case 'expll' % EXPONENTIAL LOG LIKELIHOOD (POISSON REGRESSION)
				J = sum(sum((netOut-targets.*log(netOut))))/nObs;
				dJ = 1-targets./netOut;

			case 'xent' % CROSS ENTROPY (BINARY CLASSIFICATION/LOGISTIC REG.)
				J = -sum(sum(targets.*log(netOut) + (1-targets).*log(1-netOut)))/nObs;
				dJ = (netOut - targets)./(netOut.*(1-netOut));

			case 'mcxent' % MULTI-CLASS CROSS ENTROPY (CLASSIFICATION)
				J =  -sum(sum(targets.*log(netOut)))/nObs;
				dJ = sum(class - targets);

			case {'class','classerr'}  % CLASSIFICATION ERROR (WINNER TAKE ALL)
				[~, c] = max(netOut,[],2);
				[~, t] = max(targets,[],2);
				J = sum((c ~= t))/nObs;
				dJ = 'no gradient';
				
			case {'cc','correlation'}; % CORRELATION COEFFICIENT
			
				J = corr2(netOut(:),targets(:));
				dJ = 'no gradient';
		end
	end

	function self = bProp(self)
	%net = bProp()
	%--------------------------------------------------------------------------
	%Perform gradient descent no the loss w.r.t. each of the model parameters
	%using the backpropagation algorithm. Returns updated network object, <net>
	%--------------------------------------------------------------------------
	
		sparsityError = 0;
		
		% DERIVATIVE OF OUTPUT ACTIVATION FUNCTION
		dAct = self.calcActDeriv(self.layers{end}.act,self.layers{end}.actFun);

  		% ERROR PARTIAL DERIVATIVES
		dE{self.nLayers} = self.netError.*dAct;

		% BACKPROPAGATE ERRORS
		for lL = (self.nLayers - 1):-1:2
	    
			if strcmp(self.layers{lL}.type,'sigmoid') & self.sparsity > 0 & self.epoch > 1
				KL = -self.sparsity./self.layers{lL}.meanAct + (1 - self.sparsity)./(1 - self.layers{lL}.meanAct);
				sparsityError = self.sparseGain*self.layers{lL}.lRate.*KL;
			end

			% LAYER ERROR CONTRIBUTION
			propError = dE{lL + 1}*self.layers{lL}.W;

			% DERIVATIVE OF ACTIVATION FUNCTION
			dAct = self.calcActDeriv(self.layers{lL}.act,self.layers{lL}.actFun);

			% CALCULATE LAYER ERROR SIGNAL (INCLUDE SPARSITY)
			dE{lL} = bsxfun(@plus,propError,sparsityError).*dAct;
	    end

		% CALCULATE dE/dW FOR EACH LAYER
		for lL = 1:(self.nLayers - 1)
			self.layers{lL}.dW = (dE{lL + 1}' * self.layers{lL}.act)/size(dE{lL + 1}, 1);
			self.layers{lL}.db = sum(dE{lL + 1})'/size(dE{lL + 1}, 1);
			if self.normGrad % CONSTRAIN GRADIENTS TO HAVE UNIT NORM
				normdW = norm([self.layers{lL}.dW(:);self.layers{lL}.db(:)]);
				self.layers{lL}.dW = self.layers{lL}.dW/normdW;
				self.layers{lL}.db = self.layers{lL}.db/normdW;
			end
		end
	end

	function self = updateParams(self)
	%net = updateParams()
	%--------------------------------------------------------------------------
	%Update network parameters based on states of netowrk gradient, perform
	%regularization such as weight decay and weight rescaling
	%--------------------------------------------------------------------------
	
		wPenalty = 0;
		for lL = 1:(self.nLayers - 1)
			if self.wPenalty > 0
				% L2-WEIGHT DECAY
				wPenalty = self.layers{lL}.W*self.wPenalty;
			elseif self.wPenalty < 0
				% (APPROXIMATE) L1-WEIGHT DECAY
				wPenalty = sign(self.layers{lL}.W)*self.wPenalty;
			end

			% UPDATE WEIGHT AND BIAS MOMENTA
			self.layers{lL}.pW = self.momentum*self.layers{lL}.pW + self.layers{lL}.lRate*(self.layers{lL}.dW + wPenalty);
			self.layers{lL}.pb = self.momentum*self.layers{lL}.pb + self.layers{lL}.lRate*self.layers{lL}.db;

			% UPDATE WEIGHTS
			self.layers{lL}.W = self.layers{lL}.W - self.layers{lL}.pW;
			self.layers{lL}.b = self.layers{lL}.b - self.layers{lL}.pb;
			
			% CONSTRAIN NORM OF INPUT WEIGHTS TO BE WITHIN A
			% BALL OF RADIUS net.maxNorm
			if self.maxNorm < Inf & lL == 1
				rescale = sqrt(sum([self.layers{lL}.W,self.layers{lL}.b].^2,2));
				mask = rescale > self.maxNorm;
				rescale(~mask) = 1;
				rescale(mask) = rescale(mask)/self.maxNorm;
				
			 	self.layers{lL}.W = bsxfun(@rdivide,self.layers{lL}.W,rescale);
			 	self.layers{lL}.b = self.layers{lL}.b./rescale;
			end
	    end
	end

	function out = calcAct(self,in,actFun)
	%out = calcAct(in,actFun)
	%--------------------------------------------------------------------------
	%Calculate the output activation <out> from an input <in> for activation
	%function <actFun>. Available activation functions include 'linear','exp',
	%'sigmoid', 'softmax', 'tanh', and 'softrect'.
	%--------------------------------------------------------------------------
		switch actFun
			case 'linear'
				out = self.stabilizeInput(in,1);

			case 'exp'
				in = self.stabilizeInput(in,1);
				out = exp(in);

			case 'sigmoid'
				in = self.stabilizeInput(in,1);
				out = 1./(1 + exp(-in));

			case 'softmax'
				in = self.stabilizeInput(in,1);
				maxIn = max(in, [], 2);
				tmp = exp(bsxfun(@minus,in,maxIn));
				out = bsxfun(@rdivide,tmp,sum(tmp,2));

			case 'tanh'
				in = self.stabilizeInput(in,1);
				out = tanh(in);
				
			case 'softrect'
				k = 8;
				in = self.stabilizeInput(in,k);
				out = 1/k.*log(1 + exp(k*in));
		end
	end

	function in = stabilizeInput(self,in,k);
	%in = stabilizeInput(in,k);
	%--------------------------------------------------------------------------
	%Utility function to ensure numerical stability. Clips values of <in>
	%such that exp(k*in) is within single numerical precision.
	%--------------------------------------------------------------------------
		cutoff = log(realmin('single'));
		in(in*k>-cutoff) = -cutoff/k;
		in(in*k<cutoff) = cutoff/k;
	end

	function dAct = calcActDeriv(self,in,actFun)
	%dAct = calcActDeriv(in,actFun)
	%--------------------------------------------------------------------------
	%Calculate the output activation derivatives <dAct> from an input <in> for
	%activation function <actFun>. Available activation functions derivatives
	% include 'linear','exp', sigmoid','tanh', and 'softrect'.
	%--------------------------------------------------------------------------
	
		switch actFun
			case 'linear'
				dAct = ones(size(in));

			case 'exp';
				in = self.stabilizeInput(in,1);
				dAct = in;

			case 'sigmoid'
				in = self.stabilizeInput(in,1);
				dAct = in.*(1-in);
				
			case 'tanh'
				in = self.stabilizeInput(in,1);
				dAct = 1 - in.^2;
				
			case 'softrect'
				k = 8;
				in = self.stabilizeInput(in,k);
				dAct = 1./(1 + exp(-k*in));
		end
	end

	function self = processBatches(self,batches);
	% self = processBatches(batches);
	%--------------------------------------------------------------------------
	%Ensure that user-supplied batches are consistent. <batches> should be
	%a cell array of indices into the input data. If provided batches include
	%delays (i.e. more than one index per input), we assume the first index
	%is the zero-delay case.
	%--------------------------------------------------------------------------

		% GET A TEST CASE
		test = batches{1};
		% ASSUME DELAYS ARE ALONG LONGEST DIM,
		% THEN ENSURE THEY ARE ALONG ROWS
		sz = size(test);
		if sz(1) > sz(2);
			for iB = 1:numel(batches);
				batches{iB} = batches{iB}';
			end
			sz = sz';
		end
		nDelays = sz(1);

		self.inputDelays = 0:sz(1)-1;

		% ASSIGN BATCHES TO OBJECT FIELDS
		for iB = 1:numel(batches)
			nObs = size(batches{iB},2);

			if self.nXVal < 1
				nVal = round(self.nXVal*nObs);
			else
				nVal = self.nXVal;
			end
			xValIdx = nObs-nVal+1:nObs;
			nObs = nObs - nVal;
			idx = 1:nObs;
			self.trainBatches{iB} = uint32(batches{iB}(:,1:nObs));
			self.xValBatches{iB} = uint32(batches{iB}(:,xValIdx));
		end
	end

	function dataOut = getBatchInput(self,data,batchIdx);
	%data = getBatchInput(data,batchIdx);
	%--------------------------------------------------------------------------
	%Retrieve input data for the current batch.
	%--------------------------------------------------------------------------

		[nObs,nDim] = size(data);
		[nDelay,nBatch] = size(batchIdx);
		dataOut = zeros(nBatch,nDelay*nDim);
		for iD = 1:nBatch
			idx = batchIdx(:,iD);
			zeroIdx = find(idx<=0);
			
			idx(zeroIdx) = 1; % TAKE CARE OF WRAP-AROUND
			tmp = data(idx,:)';
			tmp(:,zeroIdx) = 0;
			
			dataOut(iD,:) = tmp(:); % RASTER AND APPEND
		end
	end

	function targets = getBatchTargets(self,targets,batchIdx);
	%targets = getBatchTargets(targets,batchIdx);
	%--------------------------------------------------------------------------
	% Retrieve target variables for the current batch.
	%--------------------------------------------------------------------------

		batchIdx = batchIdx(1,:); % ASSUME ZEROTH-DELAY FOR TARGETS
		targets = targets(batchIdx,:);
	end

	function self = makeBatches(self,data);
	% net = makeBatches(data);
	%--------------------------------------------------------------------------
	% Create batches based on data. Observations are along the rows of data.
	% Note that no temporal delays are included. If modeling delays is desired,
	% provide the corresponding batches during training (see processBatches
	% method).
	%--------------------------------------------------------------------------
	
		nObs = size(data,1);

		if self.nXVal < 1
			nVal = round(self.nXVal*nObs);
		else
			nVal = self.nXVal;
		end

		xValIdx = nObs-nVal+1:nObs;
		nObs = nObs - nVal;
		nBatches = ceil(nObs/self.batchSize);
		idx = round(linspace(1,nObs+1,nBatches+1));

		for iB = 1:nBatches
			if iB == nBatches
				tmp = idx(iB):nObs;
				
			else
				tmp = idx(iB):idx(iB+1)-1;
			end
			
			batchIdx{iB} = tmp(randperm(size(tmp,2)));
		end
		
		self.trainBatches = batchIdx;

		nxBatches = ceil(nVal/self.batchSize);
		if ~isempty(xValIdx)
			xValIdx = round(linspace(xValIdx(1),xValIdx(end)+1,nxBatches));

			for iB = 1:nxBatches-1
				if iB == nxBatches
					xBatchIdx{iB} = xValIdx(iB):nVal;
				else
					xBatchIdx{iB} = xValIdx(iB):xValIdx(iB+1)-1;
				end
			end
			self.xValBatches = xBatchIdx;
		end
	end

	function self = crossValidate(self,data,targets)
	%crossValidate(data,targets)
	%--------------------------------------------------------------------------
	%Run cross-validation on current model parameters.
	%--------------------------------------------------------------------------
		xValCost = 0;
		cnt = 0;
		for iB = 1:numel(self.xValBatches)
			idx = self.xValBatches{iB};
			dataxVal = self.getBatchInput(data,idx);
			targetsxVal = self.getBatchTargets(targets,idx);
			[p,tmpCost] = self.test(dataxVal,targetsxVal,self.xValCostFun);
			if ~any(isnan(tmpCost))
				xValCost = xValCost + mean(tmpCost);
				cnt = cnt+1;
			end
		end
		
		% AVERAGE CROSSVALIDATION ERROR
		self.xValCost(self.epoch) = xValCost/cnt;
	end

	function [pred,errors] = test(self,data,targets,costFun)
	%[pred,errors] = test(data,targets,costFun)
	%--------------------------------------------------------------------------
	%Assess predictions/errors on test data
	%--------------------------------------------------------------------------
		if notDefined('costFun'),costFun=self.costFun; end
		self.testingNow = 1;
		self = self.fProp(data,targets);
		pred = self.layers{end}.act;
		errors = self.cost(targets,costFun);
		self.testingNow = 0;
	end
	

	function self = assessNet(self)
	%assessNet()
	%--------------------------------------------------------------------------
	%Utility function to assess the quality of current netork parameters and
	%store net, if necessary.
	%--------------------------------------------------------------------------
		if any(strcmp(self.xValCostFun ,{'cc','correlation'})),
			evalStr = 'self.xValCost(self.epoch) > self.bestxValCost';
		else
			evalStr = 'self.xValCost(self.epoch) < self.bestxValCost';
		end
		
		if self.epoch > 1
			if eval(evalStr)
				self.bestNet = self.layers;
				self.bestxValCost = self.xValCost(self.epoch);
				self.stopEarlyCnt = 0;
			else
				self.stopEarlyCnt = self.stopEarlyCnt + 1;
			end
		else
			self.bestNet = self.layers; % STORE FIRST NET BY DEFAULT
			if any(strcmp(self.xValCostFun,{'cc','correlation'}))
				self.bestxValCost = -self.bestxValCost;
			end
		end
	end
	
	function printProgress(self,type)
	%printProgress(type)
	%--------------------------------------------------------------------------
	%Verbose utility function. <type> is the type of message to print.
	%--------------------------------------------------------------------------
		switch type
			case 'epoch'
				fprintf('Epoch: %i/%i',self.epoch,self.nEpoch);
			case 'trainCost'
				fprintf('\t%s: %2.3f\n',self.costFun,self.trainCost(self.epoch));
			case 'time'
				fprintf('\tTime: %g\n', toc);
			case 'xValCost'
				if ~self.stopEarlyCnt
					fprintf('\tCrossValidation Error:  %g (best net) \n',self.xValCost(self.epoch));
				else
					fprintf('\tCrossValidation Error:  %g\n',self.xValCost(self.epoch));
				end
			case 'gradCheck'
				netGrad = self.auxVars.netGrad;
				numGrad = self.auxVars.numGrad;
				gradFailed = self.auxVars.gradFailed;
				switch gradFailed
					case 1, gradStr = '(Failed)';
					otherwise, gradStr = '(Passed)';
				end
				fprintf('\tNetwork = %2.6f  \t Numerical = %2.6f  %s\n' ,netGrad,numGrad,gradStr);
			case 'save'
				fprintf('\nSaving...\n\n');
		end
	end

	function arch = ensureArchitecture(self,arch)
	%arch = ensureArchitecture(arch)
	%--------------------------------------------------------------------------
	%Utility function to reprocess a supplied architecture, <arch>
	%--------------------------------------------------------------------------
	
		if ~isstruct(arch),arch.size = arch; end
		% CHECK SIZE
		if any(arch.size) < 1,error('check architecture'); end
		nLayers = numel(arch.size);
		% CHECK ACTIVATION FUNCTIONS
		if ~isfield(arch,'actFun');
			arch.actFun = ['input',repmat({'sigmoid'},1,nLayers-1)];
		elseif numel(arch.actFun) == 1;
			arch.actFun = ['input',repmat(arch.actFun,1,nLayers-1)];
		elseif numel(arch.actFun) == nLayers-1
			arch.actFun = ['input',arch.actFun];
		end
		% CHECK LEARNING RATES
		if ~isfield(arch,'lRate')
			arch.lRate = repmat(0.1,1,nLayers-1);
		elseif numel(arch.lRate) == 1
			arch.lRate = repmat(arch.lRate,1,nLayers-1);
		end
	end

	function visLearning(self)
	%visLearning()
	%--------------------------------------------------------------------------
	%Utility function to perform learning isualizations.
	%--------------------------------------------------------------------------
		if ~isempty(self.visFun)
			try
				self.visFun(self);
			catch
				if ~isfield(self.auxVars,'printVisWarning')
					fprintf('\nWARNING: visualization failed.')
					self.auxVars.printVisWarning = true;
				end
			end
		end
	end

	function save(self);
	%save();
	%--------------------------------------------------------------------------
	%Utility function to save current/best network.
	%--------------------------------------------------------------------------
		if self.verbose, self.printProgress('save'); end
		if ~isdir(self.saveDir)
			mkdir(self.saveDir);
		end
		fileName = fullfile(self.saveDir,sprintf('mlnn.mat'));
		net = self.bestNet;
		save(fileName,'net');
	end
	
end % END METHODS
end % END CLASSDEF

