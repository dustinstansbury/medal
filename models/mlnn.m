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

properties
	class = 'mlnn';
	arch					% NETWORK ARCHITECTURE (STRUCT)
	nLayers;				% # OF UNIT LAYERS
	layers ;				% LAYER STRUCTS

	netOutput = [];			% CURRENT OUTPUT OF THE NETWORK
	netError;				% CURRENT NETWORK OUTPUT ERROR
	
	costFun = 'mse';		% COST FUNCTION
	J = [];					% CURRENT COST FUNCTION VALUE
	
	nEpoch = 10;			% TOTAL NUMBER OF TRAINING EPOCHS
	epoch = 1;				% CURRENT EPOCH
	
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
	
	wPenalty= 0.001;		% WEIGHT DECAY TERM
	momentum = .9;			% MOMENTUM
	
	sparsity = 0;			% TARGET SPARSITY (SIGMOID ONLY)
	sparseGain = 1;			% GAIN ON LEARNING RATE FOR SPARSITY (SIGMOID ONLY)
	dropout = 0;			% PROPORTION OF HIDDEN UNIT DROPOUT (SIGMOID ONLY)
	denoise = 0;			% PROPORTION OF VISIBLE UNIT DROPOUT (SIGMOID ONLY)

	beginAnneal=Inf;		% # OF EPOCHS AFTER WHICH TO BEGIN SIM. ANNEALING
	beginWeightDecay=1;		% # OF EPOCHS AFTER WHICH TO BEGIN WEIGHT DECAY
	
	saveEvery = 1e100;		% SAVE PROGRESS EVERY # OF EPOCHS
	saveDir = './mlnnSave';	% WHERE TO SAVE
	displayEvery = 10;		% VISUALIZE EVERY # OF EPOCHS
	visFun;					% VISUALIZATION FUNCTION HANDLE
	trainTime = Inf;		% TRAINING DURATION
	verbose = 500;			% DISPLAY THIS # OF WEIGHT UPDATES
	auxVars = [];			% AUXILIARY VARIABLES (VISUALIZATION, ETC)
	useGPU = 0;				% USE THE GPU, IF AVAILABLE
	gpuDevice = [];			% STORAGE FOR GPU DEVICE
	checkGradients = 0;		% NUMERICAL GRADIENT CHECKING
end

methods
	function self = mlnn(arch)
	% CONSTRUCTOR FUNCTION
		self = self.init(arch);
	end

	function print(self)
	% PRINT PROPERTIES
		properties(self)
		methods(self)
	end

	function self = init(self,arch)
	% INTITIALIZE A NEURAL NETWORK GIVEN AN ARCHITECTURE
	% <arch> IS A STRUCT WITH FIELDS:
	%	.size   --  [#input, #hid1, ..., #hidN, #output]; SIZE [1 x nUnits]
	%	.actFun -- SIZE [1 x (nUnits - 1)]
	%	.lRate -- SIZE [1 x (nUnits -1)]
		 
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

	function self = train(self, data, targets)
	% TRAIN A NEURAL NET USING STOCHASTIC GRADIENT DESCENT
		% PRELIMS
		if isempty(self.trainBatches)
			self = self.makeBatches(data);
		end

		self.trainCost = zeros(self.nEpoch,1);
		self.xValCost = zeros(self.nEpoch,1);
		nBatches = numel(self.trainBatches);
		tic; cnt = 1;
		
		if self.checkGradients
			checkNNGradients(self,data,targets);
		end

		self.epoch = 1;
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
				netInput = data(batchIdx,:);
				netTargets = (targets(batchIdx,:));

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
			if ~mod(self.epoch,self.saveEvery), self.save; end
	        if self.verbose, self.printProgress('trainCost');end
	        
			% DONE?
			if (self.epoch >= self.nEpoch) || ...
				(self.stopEarlyCnt >= self.stopEarly),
				if ~isempty(self.bestNet)
					self.layers = self.bestNet;
					self.bestNet = [];
				end
				self.trainCost = self.trainCost(1:self.epoch);
				self.xValCost = self.xValCost(1:self.epoch);
				break;
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
	
		function self = fProp(self,netInput, targets)
		% FORWARD PROPAGATION OF INPUT SIGNALS
	
		nObs = size(netInput, 1);
		self.layers{1}.act = netInput;

		for lL = 2:self.nLayers
			% LAYER PRE-ACTIVATION
			preAct = bsxfun(@plus,self.layers{lL - 1}.act* ...
		                    self.layers{lL - 1}.W',self.layers{lL - 1}.b');

			% LAYER ACTIVATION
			self.layers{lL}.act = self.calcAct(preAct,self.layers{lL}.actFun);

			% SPECIAL FOR SIGMOID NETS/AUTOENCODERS
			if strcmp('sigmoid',self.layers{lL}.actFun)
				% DROPOUT
				if self.dropout > 0 && lL < self.nLayers
					if self.testingNow
						self.layers{lL}.act = self.layers{lL}.act.*(1 - self.dropout);
					else
						self.layers{lL}.act = self.layers{lL}.act.*(rand(size(self.layers{lL}.act))>self.dropout);
					end
				end
				% MOVING AVERAGE FOR TARGET SPARSITY
				if self.sparsity>0
					self.layers{lL}.meanAct = 0.9 * self.layers{lL}.meanAct + 0.1*mean(self.layers{lL}.act);
				end
			end
		end
		% COST FUNCTION & OUTPUT ERROR SIGNAL
		[self.J, self.netError] = self.cost(targets,self.costFun);
	end

	function [J, dJ] = cost(self,targets,costFun)
	% AVAILABLE COST FUNCTIONS (& GRADIENTS)
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

			case 'xent' % CROSS ENTROPY (BINARY CLASSIFICATION)
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
	% ERROR BACKPROPAGATION
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
		end
	end

	function self = updateParams(self)
	% UPDATE NETWORK PARAMETERS WITH CALCULATED GRADIENTS

		for lL = 1:(self.nLayers - 1)
			% L2-WEIGHT DECAY
			wPenalty = self.layers{lL}.W*self.wPenalty;

				% UPDATE WEIGHT AND BIAS MOMENTA
			self.layers{lL}.pW = self.momentum*self.layers{lL}.pW + self.layers{lL}.lRate*(self.layers{lL}.dW + wPenalty);

			self.layers{lL}.pb = self.momentum*self.layers{lL}.pb + self.layers{lL}.lRate*self.layers{lL}.db;

			% UPDATE WEIGHTS
			self.layers{lL}.W = self.layers{lL}.W - self.layers{lL}.pW;
			self.layers{lL}.b = self.layers{lL}.b - self.layers{lL}.pb;
	    end
	end

	function out = calcAct(self,in,actFun)
	% AVAILABLE ACTIVATION FUNCTIONS
	
		switch actFun
			case 'linear'
				out = in;

			case 'exp'
				out = exp(in);

			case 'sigmoid'
				% NUMERICAL STABILITY
				in(in>30) = 0;
				in(in<-30) = 0;
				out = 1./(1 + exp(-in));

			case 'softmax'
				% NUMERICAL STABILITY
				maxIn = max(in, [], 2);
				tmp = exp(bsxfun(@minus,in,maxIn));
				out = bsxfun(@rdivide,tmp,sum(tmp,2));

			case 'tanh'
				out = tanh(in);
				
			case 'softrect'
				in(in>30) = 0; % NUMERICAL STABILITY
				in(in<-30) = 0;
				out = log(1 + exp(in));
		end
	end

	function dAct = calcActDeriv(self,in,actFun)
	% ACTIVATION FUNCTION DERIVATIVES
	
		switch actFun
			case 'linear'
				dAct = ones(size(in));

			case 'exp';
				dAct = in;

			case 'sigmoid'
				in(in>30) = 0; % NUMERICAL STABILITY
				in(in<-30) = 0;
				dAct = in.*(1-in);
				
			case 'tanh'
				dAct = 1 - in.^2;
				
			case 'softrect'
				in(in>30) = 0; % NUMERICAL STABILITY
				in(in<-30) = 0;
				dAct = 1./(1 + exp(-in));
		end
	end

	function self = makeBatches(self,data);
	% CREATE MINIBATCHES
	% (ASSUME THAT data IS RANDOMIZED ACROSS SAMPLES (ROWS))
	
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
				batchIdx{iB} = idx(iB):nObs;
			else
				batchIdx{iB} = idx(iB):idx(iB+1)-1;
			end
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
	% CROSSVALIDATE
		self.testingNow = 1;
		xValCost = 0;
		for iB = 1:numel(self.xValBatches)
			idx = self.xValBatches{iB};
			[~,tmpCost] = self.test(data(idx,:),targets(idx,:),self.costFun);
			xValCost = xValCost + mean(tmpCost);
		end
		% AVERAGE CROSSVALIDATION ERROR
		self.xValCost(self.epoch) = xValCost/iB;
		self.testingNow = 0;
	end

	function [pred,errors] = test(self,data,targets,costFun)
	% ASSES PREDICTIONS/ERRORS ON TEST DATA
		if notDefined('costFun'),costFun=self.costFun; end
		
		self = self.fProp(data,targets);
		pred = self.layers{end}.act;
		errors = self.cost(targets,costFun);
	end
	
	function self = assessNet(self)
	% ASSESS THE CURRENT PARAMETERS AND STORE NET, IF NECESSARY
		if self.epoch > 1
			if self.xValCost(self.epoch) < self.bestxValCost
				self.bestNet = self.layers;
				self.bestxValCost = self.xValCost(self.epoch);
				self.stopEarlyCnt = 0;
			else
				self.stopEarlyCnt = self.stopEarlyCnt + 1;
			end
		else
			self.bestNet = self.layers; % STORE FIRST NET BY DEFAULT
		end
	end
	
	function printProgress(self,type)
	% VERBOSE
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
	% PREPROCESS A SUPPLIED ARCHITECTURE
	
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
	% VISUALIZATIONS
		if ~isempty(self.visFun)
			try
				self.visFun(self);
			catch
%  				plot(self.trainCost(1:self.epoch-1));
%  				title(sprintf('Cost:=%s',upper(self.costFun)));
			end
		end
	end

	function save(self);
	% SAVE CURRENT BEST NETWORK
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

