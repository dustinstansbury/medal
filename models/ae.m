classdef ae
% Autoencoder Model object
%-----------------------------------------------------------------------------
% Initialize and train an autoencoder using stochastic gradient descent
%
% Supports multiple activation functions including linear, sigmoid, tanh,
% and soft rectification (softplus).
%
% Also supports mean squared error (mse), binary, and multi-class cross-
% entropy (xent, mcxent) cost functions.
%
% Supports multiple regularization techniques including L2 weight decay,
% hidden unit dropout, and early stopping. Can be used to train denoising
% autoencoders with or without dropout (sigmoid hidden units only).
%-----------------------------------------------------------------------------
% DES
% stan_s_bury@berkeley.edu

properties
	class = 'ae';
	arch					% NETWORK ARCHITECTURE (STRUCT)
	nLayers;				% # OF HIDDEN UNIT LAYERS
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
	xValCostFun =[]; 		% METRIC FOR CROSS VALIDATION
	bestxValCost = Inf		% TRACK BEST NETWORK ERROR

	nXVal = 0;				% PORTION OF DATA HELD OUT FOR XVALIDATION
	stopEarly = 5;			% # OF EARLY STOP EPOCHS (XVAL ERROR INCREASES)
	stopEarlyCnt = 0;		% EARLY STOPPING CRITERION COUNTER
	testingNow 				% ARE WE TESTING THE NETWORK NOW?
	bestNet = [];			% STORAGE FOR BEST NETWORK

	wPenalty = 0.001;		% WEIGHT DECAY TERM
	momentum = 0;			% MOMENTUM
	maxNorm = Inf;			% NORM CONSTRAINT ON INPUT WEIGHTS
	normGrad = 0;			% % CONSTRAIN GRADIENTS TO HAVE UNIT NORM

	sparsity = 0;			% TARGET SPARSITY (SIGMOID ONLY)
	sparseGain = 1;			% GAIN ON LEARNING RATE FOR SPARSITY (SIGMOID ONLY)
	dropout = 0;			% PROPORTION OF HIDDEN UNIT DROPOUT (SIGMOID ONLY)
	denoise = 0;			% PROPORTION OF VISIBLE UNIT DROPOUT (SIGMOID ONLY)

	saveEvery = 1e100;		% SAVE PROGRESS EVERY # OF EPOCHS
	saveDir;
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
	function self = ae(arch)
	% a = ae(arch)
	%--------------------------------------------------------------------------
	%ae (autoencoder) constructor method. Initilizes a mlnn object, <net> given 
	%a user-defined architecture, <arch>.
	%--------------------------------------------------------------------------
		self = self.init	(arch);
	end

	function print(self)
	%print()	
	%--------------------------------------------------------------------------
	%Print properties and methods for mlnn object.
	%--------------------------------------------------------------------------
		properties(self);
		methods(self);
	end

	function self = init(self,arch)
	%a = init(arch)
	%--------------------------------------------------------------------------
	%Utility function to used intitialize an autoencoder given an architecture
	%<arch> is a struct with required fields:
	%	.size   -- [#input, #hid1, ..., #hidN]; SIZE [1 x nUnits]
	%	.actFun -- SIZE [1 x (nUnits - 1)]
	%	.lRate -- SIZE [1 x (nUnits -1)]
	%
	% <arch> can also contain an options fields <.opt>, which is a cell array
	% of property-value pairs.
	%
	% Returns a mlnn object, <a>.
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
	    for iL = 2:self.nLayers
		    self.layers{iL}.type = 'coding';

		    % LEARNING RATES
			self.layers{iL-1}.lRate = arch.lRate(iL-1);

			% WEIGHTS AND WEIGHT MOMENTA (A' LA BENGIO)
			range = sqrt(6/((arch.size(iL) + arch.size(iL - 1))));
			self.layers{iL - 1}.W = (rand(arch.size(iL),arch.size(iL-1))-.5)*2*range;
			self.layers{iL - 1}.pW = zeros(size(self.layers{iL-1}.W));

			% BIASES AND BIAS MOMENTA
			self.layers{iL - 1}.b = zeros(arch.size(iL), 1);
			self.layers{iL - 1}.pb = zeros(size(self.layers{iL - 1}.b));

			% ACTIVATION FUNCTION
			self.layers{iL}.actFun = arch.actFun{iL};

			% MEAN ACTIVATIONS (FOR SPARSITY)
			self.layers{iL}.meanAct = zeros(1, arch.size(iL));
		end
		self.layers{end}.type = 'recon';

		% DISTRIBUTE VALUES TO THE GPU
		if self.useGPU
			self = gpuDistribute(self);
		end
	end

	function self = train(self, data,batches)
	%a = train(data, targets, [batches])
	%--------------------------------------------------------------------------
	% Train an autoencoder using stochastic gradient descent. 
	% |data| = [#Obs x #Input]
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
			checkNNGradients(self,data,data);
		end
		
		wPenalty = self.wPenalty;
		% MAIN
	    while 1
			if self.verbose, self.printProgress('epoch'); end
			batchCost = zeros(nBatches,1);

			for iB = 1:nBatches
				% GET BATCH DATA
				batchIdx = self.trainBatches{iB};
				netInput = data(batchIdx,:);

				% ADD BINARY NOISE TO INPUT (DENOISING AE)
				if self.denoise > 0
				    netInput = netInput.*(rand(size(netInput))>self.denoise);
				end

				% BACKPROP MAIN
				self = self.fProp(netInput, netInput);
				self = self.bProp;
				self = self.updateParams;

				% ASSESS BATCH LOSS
				batchCost(iB) = self.J;
				cnt = cnt + 1;
			end

	        % MEAN LOSS OVER ALL TRAINING POINTS
			self.trainCost(self.epoch) = mean(batchCost);

	        % CROSS-VALIDATION
			if ~isempty(self.xValBatches)
				self = self.crossValidate(data);
				self = self.assessNet;
				if self.verbose, self.printProgress('xValCost');end
			end

			% EARLY STOPPING
			if (self.epoch >= self.nEpoch) || ...
				(self.stopEarlyCnt >= self.stopEarly),
				self.trainCost = self.trainCost(1:self.epoch);
				self.xValCost = self.xValCost(1:self.epoch);
				if self.interactive
					clear global a
				end
				break;
			else
				if self.interactive
					global a
					a = self;
				end
			end

			% SAVE BEST AE
			if ~mod(self.epoch,self.saveEvery) & ~isempty(self.saveDir);
				self.save; 
			end

			% DISPLAY
			if self.verbose
				self.printProgress('trainCost');
				if ~mod(cnt,self.verbose);
					self.visLearning;
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

	function self = fProp(self,netInput, targets)
	%[net,out] = fProp(netInput, targets)
	%--------------------------------------------------------------------------
	%Forward propagation of input signals, <netInput>. Also updates state of
	%network cost, if provided with <targets>. Also returns network output
	%<out>, if requested.
	%--------------------------------------------------------------------------

		nObs = size(netInput, 1);
		self.layers{1}.act = netInput;

		for iL = 2:self.nLayers
			% LAYER PRE-ACTIVATION
			preAct = bsxfun(@plus,self.layers{iL - 1}.act* ...
		                    self.layers{iL - 1}.W',self.layers{iL - 1}.b');

			% LAYER ACTIVATION
			self.layers{iL}.act = self.calcAct(preAct,self.layers{iL}.actFun);

			
			% DROPOUT
			if self.dropout > 0 && iL < self.nLayers
				if self.testingNow
					self.layers{iL}.act = self.layers{iL}.act.*(1 - self.dropout);
				else
					self.layers{iL}.act = self.layers{iL}.act.*(rand(size(self.layers{iL}.act))>self.dropout);
				end
			end

			% SPECIAL FOR SIGMOID NETS/AUTOENCODERS
			if strcmp('sigmoid',self.layers{iL}.actFun)
				% MOVING AVERAGE FOR TARGET SPARSITY
				if self.sparsity>0
					self.layers{iL}.meanAct = 0.9 * self.layers{iL}.meanAct + 0.1*mean(self.layers{iL}.act);
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
	%available cost functions ('mse', 'expll', 'xent', 'mcxent'). 
	%Note that 'correlation'and 'class' do not provide gradients.
	%--------------------------------------------------------------------------
	
		netOut = self.layers{end}.act;

		[nObs,nTargets] = size(netOut);
		switch costFun
			case 'mse' % REGRESSION
				delta = targets - netOut;
				J = 0.5*sum(sum(delta.^2))/nObs;
				dJ = -delta;

			case 'xent' % BINARY CLASSIFICATION
				J = -sum(sum(targets.*log(netOut) + (1-targets).*log(1-netOut)))/nObs;
				dJ = (netOut - targets)./(netOut.*(1-netOut));

			case 'mcxent' % MULTI-CLASS CLASSIFICATION (UNDER DEVO)
				class = softMax(netOut);
				J = -sum(sum(targets.*log(class)))/nObs;
				dJ = sum(labels - targets);
		end
	end

	function self = bProp(self)
	%a = bProp()
	%--------------------------------------------------------------------------
	%Perform gradient descent no the loss w.r.t. each of the model parameters
	%using the backpropagation algorithm. Returns updated network object, <a>
	%--------------------------------------------------------------------------

		sparsityError = 0;
		% DERIVATIVE OF OUTPUT FUNCTION
		dAct = self.calcActDeriv(self.layers{end}.act,self.layers{end}.actFun);

		% ERROR DERIVATIVE
		dE{self.nLayers} = self.netError.*dAct;

		% BACKPROPAGATE ERRORS
		for iL = (self.nLayers - 1):-1:2

			if self.sparsity > 0
				KL = -self.sparsity./self.layers{iL}.meanAct + (1 - self.sparsity)./(1 - self.layers{iL}.meanAct);
				sparsityError = self.sparseGain*self.layers{iL}.lRate.*KL;
			end
			% LAYER ERROR CONTRIBUTION
			propError = dE{iL + 1}*self.layers{iL}.W;

			% DERIVATIVE OF ACTIVATION FUNCTION
			dAct = self.calcActDeriv(self.layers{iL}.act,self.layers{iL}.actFun);

			% CALCULATE LAYER ERROR SIGNAL (INCLUDE SPARSITY)
			dE{iL} = bsxfun(@plus,propError,sparsityError).*dAct;
	    end

		% CALCULATE dE/dW FOR EACH LAYER
		for iL = 1:(self.nLayers - 1)
			self.layers{iL}.dW = (dE{iL + 1}' * self.layers{iL}.act)/size(dE{iL + 1}, 1);
			self.layers{iL}.db = sum(dE{iL + 1})'/size(dE{iL + 1}, 1);
			if self.normGrad % CONSTRAIN GRADIENTS TO HAVE UNIT NORM
				normdW = norm([self.layers{lL}.dW(:);self.layers{lL}.db(:)]);
				self.layers{lL}.dW = self.layers{lL}.dW/normdW;
				self.layers{lL}.db = self.layers{lL}.db/normdW;
			end
		end
	end

	function self = updateParams(self)
	%a = updateParams()
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

	function out = calcAct(self,preAct,actFun)
	% AVAILABLE ACTIVATION FUNCTIONS

		switch actFun
			case 'linear'
				out = preAct;
			case 'sigmoid'
				out = 1./(1 + exp(-preAct));
			case 'tanh'
				out = tanh(preAct);
			case 'softrect'
				out = log(1 + exp(preAct));
		end
	end

	function dAct = calcActDeriv(self,layerOut,actFun)
	%out = calcAct(in,actFun)
	%--------------------------------------------------------------------------
	%Calculate the output activation <out> from an input <in> for activation
	%function <actFun>. Available activation functions include 'linear','exp',
	%'sigmoid', 'softmax', 'tanh', and 'softrect'.
	%--------------------------------------------------------------------------

		switch actFun
			case 'linear'
				dAct = ones(size(layerOut));
			case 'sigmoid'
				dAct = layerOut.*(1-layerOut);
			case 'tanh'
				dAct = 1 - layerOut.^2;
			case 'softrect'
				dAct = 1./(1 + exp(-layerOut));
		end
	end

	function self = makeBatches(self,data);
	% a = makeBatches(data);
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

	function self = crossValidate(self,data)
	%crossValidate(data)
	%--------------------------------------------------------------------------
	%Run cross-validation on current model parameters.
	%--------------------------------------------------------------------------
		xValCost = 0;
		cnt = 0;
		for iB = 1:numel(self.xValBatches)
			idx = self.xValBatches{iB};
			dataxVal = self.getBatchInput(data,idx);
			[p,tmpCost] = self.test(dataxVal,dataxVal,self.xValCostFun);
			if ~any(isnan(tmpCost))
				xValCost = xValCost + mean(tmpCost);
				cnt = cnt+1;
			end
		end
		
		% AVERAGE CROSSVALIDATION ERROR
		self.xValCost(self.epoch) = xValCost/cnt;
	end

	function [pred,errors] = test(self,data,costFun)
	%[pred,errors] = test(data,costFun)
	%--------------------------------------------------------------------------
	%Assess predictions/errors on test data
	%--------------------------------------------------------------------------
		if notDefined('costFun'),costFun=self.costFun; end
		self.testingNow = 1;
		self = self.fProp(data,data);
		pred = self.layers{end}.act;
		errors = self.cost(data,costFun);
		self.testingNow = 0;
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

	function self = assessNet(self)
	%assessNet()
	%--------------------------------------------------------------------------
	%Utility function to assess the quality of current netork parameters and
	%store net, if necessary.
	%--------------------------------------------------------------------------
		if strcmp(self.xValCostFun ,'cc'),
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
					fprintf('\tCrossValidation Error:  %g (best) \n',self.xValCost(self.epoch));
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
		arch.size = [arch.size,arch.size(1)];
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
		try
			self.visFun(self);
		catch
			plot(self.trainCost(1:self.epoch-1));
			title(sprintf('Cost:=%s',upper(self.costFun)));
		end
		drawnow
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
		fileName = fullfile(self.saveDir,sprintf('ae.mat'));
		a = self.bestNet;
		save(fileName,'a');
	end

end % END METHODS
end % END CLASSDEF

