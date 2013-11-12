classdef mlcnn
% Convolutional Multilayer Neural Network
%------------------------------------------------------------------------------
% Initialize, train, and test a multilayer convolutional neural network.
%
% Supports multiple activation functions including linear, sigmoid, tanh, and
% soft rectification (softplus).
%
% Also supports mean squared error (mse), binary (xent), and multi-class
% (mcxent) cross-entropy cost functions.
%
% Supports early stopping regularization.
%------------------------------------------------------------------------------
% DES
% stan_s_bury@berkeley.edu

properties
	class = 'mlcnn';
	nLayers;				% # OF UNIT LAYERS
	layers ;				% LAYER STRUCTS

	netOut = [];			% CURRENT OUTPUT OF THE NETWORK
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
	bestNet = [];			% STORAGE FOR BEST NETWORK

	denoise = 0;			% PROPORTION OF VISIBLE UNIT DROPOUT (SIGMOID ONLY)
	dropout = 'not used';	% (POTENTIAL) PROP. OF HID. UNIT DROPOUT (SIGMOID ONLY)

	wPenalty = 0;			% WEIGHT DECAY TERM
	beginWeightDecay=1;
	momentum = 'not used';	% (POTENTIAL) MOMENTUM
	

	saveEvery = realmax;	% SAVE PROGRESS EVERY # OF EPOCHS
	saveDir;
	visFun;					% VISUALIZATION FUNCTION HANDLE
	trainTime = Inf;		% TRAINING DURATION
	verbose = 500;			% DISPLAY THIS # OF WEIGHT UPDATES
	auxVars = [];			% AUXILIARY VARIABLES (VISUALIZATION, ETC)
	useGPU = 0;				% USE THE GPU, IF AVAILABLE
	gpuDevice = [];			% STORAGE FOR GPU DEVICE
	checkGradients = 0;		% NUMERICAL GRADIENT CHECKING
end

methods
	function self = mlcnn(arch)
	% net = mlnn(arch)
	%--------------------------------------------------------------------------
	%mlnn constructor method. Initilizes a mlnn object, <net> given a user-
	%provided architecture, <arch>.
	%--------------------------------------------------------------------------
		self = self.init(arch);
	end

	function print(self)
		properties(self)
		methods(self)
	end

	function self = train(self, data, targets)
	%net = train(data, targets)
	%--------------------------------------------------------------------------
	% Train a convolutional neural net using stochastic gradient descent. 
	% INPUT:
	%     <data>:  - [#PixelsY x #PixelsX x #Channels x #Obs];
	%
	%  <targets>:  - [#Output x #Obs]
	%--------------------------------------------------------------------------

		% DISTRIBUTE VALUES TO THE GPU
		if self.useGPU
			self = gpuDistribute(self);
		end

		self = self.makeBatches(data);
		self.trainCost = zeros(self.nEpoch,1);
		self.xValCost = zeros(self.nEpoch,1);
		nBatches = numel(self.trainBatches);
		tic; cnt = 1;

		if self.checkGradients,	checkNNGradients(self,data,targets); end

		% MAIN
	    while 1
			if self.verbose, self.printProgress('epoch'); end
			batchCost = zeros(nBatches,1);
			wPenalty = self.wPenalty;
			
			if self.epoch >= self.beginWeightDecay
				self.wPenalty = wPenalty;
			else
				self.wPenalty = 0;
			end

			for iB = 1:nBatches
				% GET BATCH DATA
				batchIdx = self.trainBatches{iB};
				netInput = data(:,:,:,batchIdx);
				netTargets = (targets(:,batchIdx));

				% ADD NOISE TO INPUT (DENOISING CAE)
				if self.denoise > 0
				    netInput = netInput.*(rand(size(netInput))>self.denoise);
				end

				% BACKPROP MAIN
				self = self.fProp(netInput, netTargets);
				self = self.bProp;
				self = self.updateParams;

				% ASSESS BATCH COST
				batchCost(iB) = self.J;
				cnt = cnt + 1;
			end

	        % AVERAGE COST OVER ALL TRAINING POINTS
			self.trainCost(self.epoch) = mean(batchCost);

	        % CROSS-VALIDATION
			if ~isempty(self.xValBatches)
				self = self.crossValidate(data,targets);
				self = self.assessNet;
				if self.verbose, self.printProgress('xValCost');end
			end

			% EARLY STOPPING
			if (self.epoch >= self.nEpoch) || ...
				(self.stopEarlyCnt >= self.stopEarly),
				self.trainCost = self.trainCost(1:self.epoch);
				self.xValCost = self.xValCost(1:self.epoch);
				break;
			end

			% SAVE BEST NETWORK
			if ~mod(self.epoch,self.saveEvery) & ~isempty(self.saveDir)
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
		if notDefined('targets'), targets = []; end
		nObs = size(netInput, 1);		
		self.layers{1}.fm = netInput;

		for lL = 2:self.nLayers
			switch self.layers{lL}.type
			case 'conv'
			% LOOP OVER LAYER FEATURE MAPS
			for jM = 1:self.layers{lL}.nFM

				[nInY,nInX,nInFM,nObs] = size(self.layers{lL-1}.fm);
				
				% INITIALIZE LAYER MAP -- [nY,nX,nM,nObs]
				featMap = zeros([self.layers{lL}.fmSize,1,nObs]);

				% POOL OVER INPUT FEATURE MAPS,
				% CALC LAYER PRE-ACTIVATION
				for iM = 1:self.layers{lL-1}.nFM
					featMap = featMap + convn(self.layers{lL-1}.fm(:,:,iM,:), ...
					self.layers{lL}.filter(:,:,iM,jM),'valid');
				end
				
				if any(isnan(self.layers{lL}.b(jM))), keyboard, end

				% ADD LAYER BIAS
				featMap = featMap + self.layers{lL}.b(jM);

				% COMPLETE FEATURE MAP
				self.layers{lL}.fm(:,:,jM,:) = self.calcAct(featMap,self.layers{lL}.actFun);
				
			end

			case 'subsample'
				stride = self.layers{lL}.stride;
				% DOWNSAMPLE THE FEATURE MAPS FROM LAYER (l-1)
				for jM = 1:self.layers{lL-1}.nFM
					layerIn = self.layers{lL-1}.fm(:,:,jM,:);
					self.layers{lL}.fm(:,:,jM,:) = self.DOWN(layerIn,stride);
				end

			case 'output'
				% UNPACK OUTPUT FEATURES & CALCULATE NETWORK OUTPUT
				self = self.calcOutput;
			case 'rect'
			case 'lcn'
			case 'pool'
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

	function self = calcOutput(self);
	% net = calcOutput();
	%--------------------------------------------------------------------------
	% Calculate the network ouput given the current parameters
	%--------------------------------------------------------------------------

		[nY,nX,nM,nObs]= size(self.layers{end-1}.fm);
		
		% # OF ENTRIES IN EACH FEATURE MAP
		nMap = prod([nY,nX]);

		% INITIALIZE OUTPUT FEATURES
		self.layers{end}.features = zeros(nMap*nM,nObs);

		% UNPACK MAPS INTO A MATRIX FOR CALCULATING OUTPUT
		for jM = 1:self.layers{end-1}.nFM
			map = self.layers{end-1}.fm(:,:,jM,:);
			self.layers{end}.features((jM-1)*nMap+1:jM*nMap,:) = reshape(map,nMap,nObs);
		end
		
		% CALC NET OUTPUTS
		preAct = bsxfun(@plus,self.layers{end}.W* ...
						self.layers{end}.features, ...
		                self.layers{end}.b);

		self.netOut = self.calcAct(preAct,self.layers{end}.actFun);
		
		if any(isnan(self.netOut)), keyboard; end
	end

	function [J, dJ] = cost(self,targets,costFun)
	% [J, dJ] = cost(self,targets,costFun)
	%--------------------------------------------------------------------------
	% Calculate the cost function <J> and derivative thereof for a set of targets
	% and the current state of the network.
	%--------------------------------------------------------------------------

		netOut = self.netOut;
	
		[nTargets,nObs] = size(netOut);
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

		case {'class','classerr'}  % CLASSIFICATION ERROR (WINNER TAKE ALL)
		
			[~, class] = max(netOut,[],1);
			[~, t] = max(targets,[],1);
			J = sum((class ~= t))/nObs;
			dJ = 'no gradient';
		case {'correlation','cc'}
			J = corr2(netOut,targets);
			dJ = 'no gradient';
		end
		if any(isnan(self.J)), keyboard, end
	end

	function self = bProp(self)
	%net = bProp()
	%--------------------------------------------------------------------------
	%Perform gradient descent no the loss w.r.t. each of the model parameters
	%using the backpropagation algorithm. Returns updated network object, <net>
	%--------------------------------------------------------------------------
	
		% DERIVATIVE OF OUTPUT ACTIVATION FUNCTION
		dAct = self.calcActDeriv(self.netOut,self.layers{end}.actFun);

		% OUTPUT ERROR SIGNAL -- [#Out x #Obs]
		outES = self.netError.*dAct;

		% ERROR SIGNAL  -- [#Features x #Obs]
		es = self.layers{end}.W'*outES;

		% IN CASE LAST FEATURE LAYER IS CONV.
		if strcmp(self.layers{end-1}.type,'conv')
			dAct = self.calcActDeriv(self.layers{end}.features,self.layers{end-1}.actFun);
			es = es.*dAct;
		end

		% REPACK ERROR SIGNAL INTO 2-D FEATURE MAP REPRESENTATION
		[nY,nX,nM,nObs] = size(self.layers{end-1}.fm);
		self.layers{end-1}.es = zeros([nY,nX,nM,nObs]);

		nMap = prod([nY*nX]); % NUMBER OF ENTRIES PER 2D MAP

		for jM = 1:self.layers{end-1}.nFM
			self.layers{end-1}.es(:,:,jM,:) = ...
			reshape(es((jM-1)*nMap+1:jM*nMap,:),[nY,nX,1,nObs]);
		end

		% BACKPROPATE ERROR SIGNAL
		for lL = self.nLayers-2:-1:2
			switch self.layers{lL}.type
			case 'conv'
				stride = self.layers{lL + 1}.stride;
				mapSz = size(self.layers{lL}.fm);
				self.layers{lL}.es = zeros(mapSz);

				for jM = 1:self.layers{lL}.nFM
					switch self.layers{lL+1}.type
					case 'subsample'
						% UPSAMPLE ES FROM ABOVE SUBSAMPLE LAYER
						propES = self.UP(self.layers{lL+1}.es(:,:,jM,:), ...
						                 [stride(1), stride(2),1,1])/prod(stride);
					case 'rect'
					case 'lcn'
					end

                   % DERIVATIVE OF ACTIVATION FUNCTION
                   dAct = self.calcActDeriv(self.layers{lL}.fm(:,:,jM,:), ...
											self.layers{lL}.actFun);

                   % CALCULATE LAYER ERROR SIGNAL
					self.layers{lL}.es(:,:,jM,:) = propES.*dAct;
				end
			case 'rect'
			case 'lcn'
			case 'pool'

			case 'subsample'
				[nY,nX,nM,nObs] = size(self.layers{lL}.fm);
				self.layers{lL}.es = zeros([nY,nX,nM,nObs]);
				for jM = 1:self.layers{lL}.nFM
					% FORM FEATURE MAP ERROR SIGNAL
					propES = zeros(nY,nX,1,nObs);
					for kM = 1:self.layers{lL+1}.nFM
						rotFilt = self.ROT(self.layers{lL+1}.filter(:,:,jM,kM));
						es = self.layers{lL+1}.es(:,:,kM,:);
						propES = propES + convn(es,rotFilt,'full');
					end
					self.layers{lL}.es(:,:,jM,:) = propES;
					if any(isnan(self.layers{lL}.es(:))), keyboard; end
					if any(isinf(self.layers{lL}.es(:))), keyboard; end
				end
			end
		end
		
		% CALCULATE THE GRADIENTS
		for lL = 2:self.nLayers-1
			[nX,nY,nM,nObs] = size(self.layers{lL}.fm);
			switch self.layers{lL}.type
			case 'conv'
				for jM = 1:self.layers{lL}.nFM
					es = self.layers{lL}.es(:,:,jM,:);
					for iM = 1:self.layers{lL-1}.nFM
						input = self.FLIPDIMS(self.layers{lL-1}.fm(:,:,iM,:));
						dEdFilter = convn(input,es,'valid')/nObs;
						self.layers{lL}.dFilter(:,:,iM,jM) = dEdFilter;
						if isnan(any(dEdFilter)), keyboard; end
					end
					self.layers{lL}.db(jM) = sum(es(:))/nObs;
				end
			case 'rect'
			case 'lcn'
			case 'pool'
			end
		end
		
		% GRADIENTS FOR OUTPUT LAYER WEIGHTS AND BIASES
		self.layers{end}.dW = outES*self.layers{end}.features'/nObs;
		self.layers{end}.db = mean(outES,2);
		
	end

	function self = updateParams(self)
	%net = updateParams()
	%--------------------------------------------------------------------------
	%Update network parameters based on states of netowrk gradient, perform
	%regularization such as weight decay and weight rescaling
	%--------------------------------------------------------------------------
		wPenalty = 0;
		for lL = 2:self.nLayers-1
			switch self.layers{lL}.type
			% CURRENTLY, ONLY UPDATE FILTERS AND FM BIASES
			% PERHAPS, IN THE FUTURE, WE'LL BE FANCY, AND DO FANCY UPDATES
			case {'conv','output'}
				lRate = self.layers{lL}.lRate;
				for jM = 1:self.layers{lL}.nFM
					% UPDATE FEATURE BIASES
					self.layers{lL}.b(jM) = self.layers{lL}.b(jM) - ...
					                    lRate*self.layers{lL}.db(jM);
					                    
					% UPDATE FILTERS
					for iM = 1:self.layers{lL-1}.nFM
						if self.wPenalty > 0 % L2 REGULARIZATION
							wPenalty = self.layers{lL}.filter(:,:,iM,jM)*self.wPenalty;
						elseif self.wPenalty < 0 % L1-REGULARIZATION (SUB GRADIENTS)
							wPenalty = sign(self.layers{lL}.filter(:,:,iM,jM))*abs(self.wPenalty);
						end
						self.layers{lL}.filter(:,:,iM,jM) = ...
						self.layers{lL}.filter(:,:,iM,jM) - ...
						lRate*(self.layers{lL}.dFilter(:,:,iM,jM)+wPenalty);
					end
				end
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

	function out = DOWN(self,data,stride)
	%out = DOWN(data,stride)
	%--------------------------------------------------------------------------
	% Downsample 1st 2 dimensions of a tensor
	%--------------------------------------------------------------------------
		tmp = ones(stride(1),stride(2));
		tmp = tmp/prod(stride(:));
		out = convn(data,tmp,'valid');
		out = out(1:stride(1):end,1:stride(2):end,:,:,:);
	end
	
	function out = UP(self,data,scale);
	%out = UP(data,stride)
	%--------------------------------------------------------------------------
	% Upsample 1st 2 dimensions of a tensor
	%--------------------------------------------------------------------------
	% UPSAMPLE DIMESIONS OF A TENSOR
		dataSz = size(data);
		idx = cell(numel(dataSz),1);
		for iD = 1:numel(dataSz)
			tmp = zeros(dataSz(iD)*scale(iD),1);
			tmp(1:scale(iD):dataSz(iD)*scale(iD)) = 1;
			idx{iD} = cumsum(tmp);
		end
		out = data(idx{:});
	end

	function out = ROT(self,out)
	%out = ROT(self,in)
	%--------------------------------------------------------------------------
	% Rotate the 1st two dimensions of a tensor by one-half rotation
	%--------------------------------------------------------------------------
		out = out(end:-1:1,end:-1:1,:,:);
	end

	function out = FLIPDIMS(self,out)
	%out = FLIPDIMS(in)
	%--------------------------------------------------------------------------
	% Flip all dimensions of a tensor <in>
	%--------------------------------------------------------------------------
		for iD = 1:numel(size(out))
			out = flipdim(out,iD);
		end
	end

	function self = makeBatches(self,data);
	% net = makeBatches(data);
	%--------------------------------------------------------------------------
	% Create batches based on data. Observations are along the rows of data.
	% (ASSUME THAT data IS RANDOMIZED ACROSS SAMPLES (ROWS))
	%--------------------------------------------------------------------------

		nObs = size(data,4);

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
	%net = crossValidate(data,targets)
	%--------------------------------------------------------------------------
	%Run cross-validation on current model parameters.
	%--------------------------------------------------------------------------
	
		xValCost = 0;
		for iB = 1:numel(self.xValBatches)
			idx = self.xValBatches{iB};
			tmpCost = self.test(data(:,:,:,idx),targets(:,idx),self.costFun);
			xValCost = xValCost + mean(tmpCost);
		end
		% AVERAGE data-VALIDATION ERROR
		self.xValCost(self.epoch) = xValCost/iB;
	end

	% ASSES PREDICTIONS/ERRORS ON TEST DATA
	function [cost,pred] = test(self,data,targets,costFun)
		if notDefined('costFun') costFun=self.costFun; end

		for lL = 1:self.nLayers
			try
				self.layers{lL}.fm = [];
			catch
			end
		end
		self = self.fProp(data,targets);
		pred = self.netOut;
		cost = self.cost(targets,costFun);
	end

	function self = assessNet(self)
	%assessNet()
	%--------------------------------------------------------------------------
	%Utility function to assess the quality of current netork parameters and
	%store net, if necessary.
	%--------------------------------------------------------------------------
	
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

	function self = init(self,arch)
	%net = init(arch)
	%--------------------------------------------------------------------------
	%Utility function to used intitialize a neural network given an architecture
	%<arch> is a cell array of structs, one for each layer in the network. The
	%fields of each structure will depend on the type of layer. Supported lay-
	%ers include 'input','conv','subsample','output'.
	%
	%Note, the first and last layers should be of 'type' 'input' and 'output'
	%respectively.
	%
	% Returns a mlcnn object, <net>.
	%--------------------------------------------------------------------------
	% INTITIALIZE A NEURAL NETWORK GIVEN AN ARCHITECTURE
	% <arch> IS A CELLARRAY OF STRUCTS WITH LAYER-SPECIFIC PARAMETERS

		arch = self.ensureArchitecture(arch);
		self.nLayers = numel(arch);

		% INITIALIZE LAYERS FROM ARCHITECTURE
	    for lL = 1:numel(arch)
		    self.layers{lL}.type = arch{lL}.type;
		    switch arch{lL}.type
		    case 'input'
			    self.layers{lL}.dataSize = arch{lL}.dataSize;
			    self.layers{lL}.fmSize = arch{lL}.dataSize(1:2);
			    self.layers{lL}.nFM = arch{lL}.dataSize(3);

			case 'conv'
				if strcmp(arch{lL-1}.type,'input')
					nInY = arch{lL-1}.dataSize(1);
					nInX = arch{lL-1}.dataSize(2);
					nInFM = arch{lL-1}.dataSize(3);
				else
					nInX = self.layers{lL-1}.fmSize(2);
					nInY = self.layers{lL-1}.fmSize(1);
					nInFM = self.layers{lL-1}.nFM;
					
				end
				% INTERMEDIATE VARIABLES
				nFM = arch{lL}.nFM;
				filtSize = arch{lL}.filterSize;
				fmSize = [nInY,nInX] - filtSize + 1;
				fanIn = nInFM*prod(filtSize);
				fanOut = nFM*prod(filtSize);
				range = 2*sqrt(6/((fanIn + fanOut)));

				% INITIALIZE LAYER PARAMETERS
				self.layers{lL}.actFun = arch{lL}.actFun;
				self.layers{lL}.lRate = arch{lL}.lRate;
				self.layers{lL}.filterSize = filtSize;
				self.layers{lL}.fmSize = fmSize;
				self.layers{lL}.nFM = nFM;
				self.layers{lL}.fm = [];
				self.layers{lL}.filter = range*(rand([filtSize,nInFM,nFM])-.5);
				self.layers{lL}.dFilter = zeros(size(self.layers{lL}.filter));
				self.layers{lL}.b = zeros(nFM,1);
				self.layers{lL}.db = zeros(nFM,1);


			case 'subsample'
				% INTERMEDIATE VARIABLES
				stride = arch{lL}.stride;
				fmSize = floor(self.layers{lL-1}.fmSize./stride);
				nFM = self.layers{lL-1}.nFM;

				% INITIALIZE LAYER PARAMETERS
				self.layers{lL}.stride = stride;
				self.layers{lL}.fmSize = fmSize;
				self.layers{lL}.nFM = nFM;
				self.layers{lL}.fm = [];
				self.layers{lL}.b = zeros(nFM,1);
				self.layers{lL}.db = zeros(nFM,1);

			case 'output'

				nOut = arch{lL}.nOut;
				nFMIn = self.layers{lL-1}.nFM;
				nOutFeats = prod([self.layers{lL-1}.fmSize,nFMIn]);
				range = 2*sqrt(6/(nOutFeats + nOut));

				% ADJUST NETWORK OUTPUT LAYER PARAMETERS
				self.layers{lL}.nOut = nOut;
				self.layers{lL}.actFun = arch{lL}.actFun;
				self.layers{lL}.lRate = arch{lL}.lRate;
				self.layers{lL}.W = range*(rand(nOut,nOutFeats)-.5);
				self.layers{lL}.dW = zeros(size(self.layers{lL}.W));
				self.layers{lL}.b = zeros(nOut,1);
				self.layers{lL}.db = zeros(nOut,1);
		    end
		end
	end


	function arch = ensureArchitecture(self,arch)
	%arch = ensureArchitecture(arch)
	%--------------------------------------------------------------------------
	%Utility function to reprocess a supplied architecture, <arch>
	%--------------------------------------------------------------------------
	
		if ~iscell(arch), error('<arch> needs to be a cell array of layer params');end
		if ~strcmp(arch{1}.type,'input'), error('define an input layer'); end
		if ~strcmp(arch{end}.type,'output'), error('define an output layer'); end

		% ENSURE LAYER-SPECIFIC PARAMS
		for lL = 1:numel(arch)
			lParams = fields(arch{lL});
			switch arch{lL}.type
			case 'input'
				if ~any(strcmp(lParams,'dataSize')), error('must provide data size');end

			case 'conv'
				if ~any(strcmp(lParams,'filterSize')) || isempty(arch{lL}.filterSize)
					arch{lL}.filterSize = [5 5];
				elseif numel(arch{lL}.filterSize) == 1;
					arch{lL}.filterSize = repmat(arch{lL}.filterSize,[1,2]);
				end
				if ~any(strcmp(lParams,'nFM'));
					arch{lL}.nFM = 10;
				end
				if ~any(strcmp(lParams,'lRate'));
					arch{lL}.lRate = .5;
				end
				if ~any(strcmp(lParams,'actFun'));
					arch{lL}.actFun = 'sigmoid';
				end

			case 'subsample'
				if ~any(strcmp(lParams,'stride')) || isempty(arch{lL}.stride);
					arch{lL}.stride = [2 2];
				elseif numel(arch{lL}.stride) == 1;
					arch{lL}.stride = repmat(arch{lL}.stride,[1,2]);
				end
			case 'output'
				if ~any(strcmp(lParams,'nOut'))
					error('must provide number of outputs');
				end
				if ~any(strcmp(lParams,'actFun'));
					arch{lL}.actFun = 'sigmoid';
				end
				if ~any(strcmp(lParams,'lRate'));
					arch{lL}.lRate = .5;
				end
			case 'rect'
			case 'lcn'
			case 'pool'
			end
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


	function visLearning(self)
	%visLearning()
	%--------------------------------------------------------------------------
	%Utility function to perform learning isualizations.
	%--------------------------------------------------------------------------
		try
			self.visFun(self);
		catch
			if ~isfield(self.auxVars,'printVisWarning')
				fprintf('\nWARNING: visualization failed.')
				self.auxVars.printVisWarning = true;
			end
		end
	end

	function save(self);
	%save();
	%--------------------------------------------------------------------------
	%Utility function to save current network.
	%--------------------------------------------------------------------------
		if self.verbose, self.printProgress('save'); end
		if ~isdir(self.saveDir)
			mkdir(self.saveDir);
		end
		fileName = fullfile(self.saveDir,sprintf('mlcnn.mat'));
		net = self.bestNet;
		save(fileName,'net');
	end
end % END METHODS
end % END CLASSDEF
