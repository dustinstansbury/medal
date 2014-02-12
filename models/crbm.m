classdef crbm
% Convolutional Restricted Boltzmann Machine Object
%-------------------------------------------------------------------------------
% Initialize and train a convolutional Restricted Bolzmann Machine model.
%
% Supports binary and Gaussian inputs.
%
% Current inmplementation is based on the implementation described in Honglak
% Lee's 2010 Dissertation. This implementation only supports a single input map
% per datapoint, and treats each datapoint as a minibatch.
%
% Supports L2 weight decay and hidden unit target sparsity, as described in
% Lee (2010)
%-------------------------------------------------------------------------------
% DES
% stan_s_bury@berkeley.edu

properties
	class = 'crbm'			% MODEL CLASS
	inputType = 'binary'	% DEFAULT INPUT TYPE
	visSize					% SIZE OF INPUT FEATURE MAPS
	filterSize = [7 7];		% (DEFAULT) SIZE OF FILTERS TO LEARN
	stride = [2 2];			% (DEFAULT) STRIDE LENGTH
	nFM = 12;				% (DEFAULT) # OF FEATURE MAPS IN HIDDEN LAYER

	% MODEL PARAMETERS
	W						% CONNECTION WEIGHTS (FILTERS)
	b						% HIDDEN UNIT BIASES
	c						% VISIBLE UNIT BIASES
	% PARAMETER GRADIENTS
	dW					
	db
	dc
	% UNIT STATES/STATISTICS
	eVis					% VISIBLE UNIT EXPECTATION
	eHid					% HIDDEN UNIT EXPECTATION
	eHid0					% INITIAL HIDDEN UNIT EXPECTATION (FOR GRADENTS)
	hidI					% BOTTOM-UP SIGNAL FROM VISIBLES TO HIDDENS
	visI					% TOP-DOWN SIGNAL FROM HIDDEN FEATURE MAPS TO VISIBLES
	ePool					% EXPECTATION OF POOLING LAYER
	% LEARNING PARAMETERS
	nEpoch = 10;			% # OF TIMES WE SEE THE DAT
	lRate = 0.1;			% LEARNING RATE
	nGibbs = 1;				% # OF GIBBS SAMPLES (CONTRASTIVE DIVERGENCE)
	% DISPLAYING
	verbose = 1;			% DISPLAY STDOUT
	displayEvery = 100;		% DISPLAY AFETER THIS # OF WEIGHT UPDATES
	visFun = [];			% VISUALIZATION FUNCTION HANDLE
	% REGULARIZATION
	sparsity = 0.02;		% TARGET HIDDEN UNIT SPARSITY
	sparseGain = 1;			% GAIN ON THE LEARNING RATE FOR SPARSITY CONSTRAINTS
	momentum = 0.9;			% (DEFAULT) GRADIENT MOMENTUM FOR WEIGHT UPDATES
	wPenalty = .05;			% L2 WEIGHT PENALTY
	beginAnneal = Inf;		% BEGIN SIMULUATED ANNEALING AFTER THIS # OF EPOCHS
	beginWeightDecay = 1;	% BEGIN WEIGHT DECAY AFTER THIS # OF EPOCHS
	% GENERAL VARIABLES
	log = struct();			% FOR KEEPING TRACK OF ERRORS
	saveEvery = 1e10;		% SAVE EVERY # OF EPOCHS
	saveFold;
	trainTime;				% TIME TO TRAIN
	auxVars;				% AUXILARY VARIABLES (FOR VISUALIZATION, ETC)
	useGPU = 0;				% USE GPU IF AVAILABLE
	gpuDevice;				% THE GPU DEVICE MODEL IS USING
end % END PROPERTIES

methods

	function self = crbm(arch);
	% net = mlnn(arch)
	%--------------------------------------------------------------------------
	%crbm constructor method. Initilizes a mlnn object, <net> given a user-
	%provided architecture, <arch>.
	%--------------------------------------------------------------------------
			self = self.init(arch);
	end

	function [] = print(self)
		%print()
	%--------------------------------------------------------------------------
	%Print properties and methods for crbm object.
	%--------------------------------------------------------------------------
		properties(self);
		methods(self);
	end

	function self = train(self,data)
	%c = train(data, targets, [batches])
	%--------------------------------------------------------------------------
	% Train a crbm using stochastic gradient descent. 
	%--------------------------------------------------------------------------
		if self.useGPU
			self = gpuDistribute(self);
			data = gpuArray(data);
		end
		tic
		[nY,nX,nBatch] = size(data);
		wPenalty = self.wPenalty;
		dCount = 1;
		iE = 1;
		while 1
			sumErr = 0;
			if self.verbose
				self.printProgress('epoch',[iE,self.nEpoch])
			end
			
			% BEGIN SIMULATED ANNEALING?
			if self.beginAnneal
				self.lRate = max(self.lRate/max(1,iE/self.beginAnneal),1e-10);
			end

			% BEGIN WEIGHT DECAY?
			if iE < self.beginWeightDecay
				self.wPenalty = 0;
			else
				self.wPenalty = wPenalty;
			end

			% LOOP OVER INPUTS (WE CONSIDER EACH IMAGE A "MINIBATCH")
			for jV = 1:nBatch
				batchX = data(:,:,jV);
				
				[self,dW,db,dc] = self.calcGradients(batchX);
				self = self.applyGradients(dW,db,dc);

				batchErr = self.batchErr(batchX);
				sumErr = sumErr + batchErr;

				% INDUCE HIDDEN UNIT SPARSITY (Eq 5.5; Lee, 2010)
				if self.sparsity
					dcSparse = self.lRate*self.sparseGain*(squeeze(self.sparsity -mean(mean(self.eHid0))));
					self.c = self.c + dcSparse;
				end

				self.log.err(iE) = single(sumErr);
				self.log.lRate(iE) = self.lRate;

				% VERBOSE/VISUALIZATIONS
				if self.verbose & ~mod(dCount,self.displayEvery);
					self.auxVars.batchErr = batchErr;
					self.auxVars.jV = jV;
					self.auxVars.nBatch = nBatch;
					self.printProgress('batchErr')
					if ~isempty(self.visFun)
						self.auxVars.error = self.log.err(1:iE);
						self.auxVars.lRate = self.log.lRate(1:iE);
						self.auxVars.batchX = batchX;
						try self.auxVars.dcSparse = dcSparse; catch, end
						self.visLearning();
					end
				end
				dCount = dCount+1;
			end

			% EPOCH ERROR
			if self.verbose 
				self.auxVars.iE = iE;
				self.auxVars.sumErr = sumErr;
				self.printProgress('epochErr');
			end

			if iE > 1
				if iE == self.saveEvery & ~isempty(self.saveFold)
					self.save(iE)
				end
			end
			
			if iE >= self.nEpoch
				break
			end
			iE = iE + 1;
		end
		self.trainTime = toc;
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
			self.gpuDevice = [];
		end
		fprintf('\n');
	end

	function [self,dW,db,dc] = calcGradients(self,data)
	% [c,dW,db,dc] = calcGradients(data)
	%--------------------------------------------------------------------------
	% Draw MCMC samples from the current model and calculate parameter
	% gradients. 
	%--------------------------------------------------------------------------
		self = self.runGibbs(data);
		for iK = 1:self.nFM
			dW(:,:,iK)  = (conv2(data,self.ff(self.eHid0(:,:,iK)),'valid') - ...
			conv2(self.eVis,self.ff(self.eHid(:,:,iK)),'valid'));
		end
		db = sum(sum(data - self.eVis));
		dc = squeeze(sum(sum(self.eHid0 - self.eHid)));
	end

	function self = runGibbs(self,data)
	% self = runGibbs(X,targets)
	%--------------------------------------------------------------------------
	% Draw MCMC samples from the current model via Gibbs sampling.
	%--------------------------------------------------------------------------
	% INPUT:
	%       <X>:  - minibatch data.
	%
	% OUTPUT:
	%    <self>:  - RBM object with updated states
	%--------------------------------------------------------------------------
		% INITIAL PASS
		[self,self.eHid0] = self.hidGivVis(data);
		for iC = 1:self.nGibbs
			% RECONSTRUCT VISIBLES
			self = self.visGivHid(self.drawBernoulli(self.eHid));
			% FINISH CD[n]
			self = self.hidGivVis(self.eVis);
		end
	end

	function self = applyGradients(self,dW,db,dc)
	%c = applyGradients(dW,db,dc)
	%--------------------------------------------------------------------------
	% Update all parameters given current gradients.
	%--------------------------------------------------------------------------
		[self.W,self.dW] = self.updateParams(self.W,dW,self.dW,self.wPenalty);
		[self.b,self.db] = self.updateParams(self.b,db,self.db,0);
		[self.c,self.dc] = self.updateParams(self.c,dc,self.dc,0);
	end

	function [params,grad] = updateParams(self,params,grad,gradPrev,wPenalty)

		grad = self.momentum*gradPrev + (1-self.momentum)*grad;
		params = params + self.lRate*(grad - wPenalty*params);
	end

	function [self,eHid] = hidGivVis(self,vis)
	% [self,eHid] = hidGivVis(X,targets,[sampleHid])
	%--------------------------------------------------------------------------
	% Update hidden unit expectations conditioned on the current states of the 
	% visible units.
	%--------------------------------------------------------------------------
	% INPUT:
	%         <X>:  - batch data.
	%   <targets>:  - possible target variables.
	% <sampleHid>:  - flag indicating to sample the states of the hidden units.
	%
	% OUTPUT:
	%      <self>:  - RBM object with updated hidden unit probabilities/states.
	%--------------------------------------------------------------------------
	
		for iK = 1:self.nFM
			self.hidI(:,:,iK) = conv2(vis,self.ff(self.W(:,:,iK)),'valid')+self.c(iK);
		end
		eHid = exp(self.hidI)./(1 + self.pool(exp(self.hidI))); % (Eq 5.3; Lee, 2010)
		self.eHid = eHid;
	end

	function self = visGivHid(self,hid)
	% self = hidGivVis(hid)
	%--------------------------------------------------------------------------
	% Update hidden unit expectations conditioned on the current states of the 
	% visible units.
	%--------------------------------------------------------------------------
	% INPUT:
	%         <X>:  - batch data.
	%   <targets>:  - possible target variables.
	% <sampleHid>:  - flag indicating to sample the states of the hidden units.
	%
	% OUTPUT:
	%      <self>:  - RBM object with updated hidden unit probabilities/states.
	%--------------------------------------------------------------------------

		% CALCULATE VISIBLE EXPECTATIONS GIVEN EACH HIDDEN FEATURE MAP
		for iK = 1:self.nFM
			self.visI(:,:,iK) = conv2(hid(:,:,iK),self.W(:,:,iK),'full');
		end
		I = sum(self.visI,3) + self.b;
		if strcmp(self.inputType,'binary'); % USING MEAN FIELD
			self.eVis = self.sigmoid(I);
		else
			self.eVis = I;
		end
	end

	function [self,ePool] = poolGivVis(self,vis)
	%[c,ePool] = poolGivVis(vis)
	%--------------------------------------------------------------------------
	% Calculate pooling layer expectations given visibles (for sampling & dbns)
	%--------------------------------------------------------------------------	
		I = zeros(size(self.eHid));
		for iK = 1:self.nFM
			I(:,:,iK) = conv2(vis,self.ff(self.W(:,:,iK)),'valid') + self.c(iK);
		end
		ePool = 1 - (1./(1 + self.pool(exp(I)))); % (Eq 5.4; Lee, 2010)
		self.ePool = ePool;
	end

	function blocks = pool(self,I); % [[hiddenSize]/stride data nFeatures]
	%blocks = pool(I)
	%--------------------------------------------------------------------------
	% Pool hidden activations I by summing over blocks b_\alpha
	%--------------------------------------------------------------------------
		nCols = size(I,1);
		nRows = size(I,2);
		yStride = self.stride(1);
		xStride = self.stride(2);
		blocks = zeros(size(I));
		% DEFINE BLOCKS B_\alpha
		for iR = 1:ceil(nRows/yStride)
			rows = (iR-1)*yStride+1:iR*yStride;
			for jC = 1:ceil(nCols/xStride)
				cols = (jC-1)*xStride+1:jC*xStride;
				% TAKE SUMS
				blockVal = squeeze(sum(sum(I(rows,cols,:))));
				blocks(rows,cols,:) = repmat(permute(blockVal, ...
				 [2,3,1]),numel(rows),numel(cols));
			end
		end
	end

	function arch = ensureArchitecture(self,arch)
	%arch = ensureArchitecture(arch)
	%--------------------------------------------------------------------------
	%Utility function to reprocess a supplied architecture, <arch>
	%--------------------------------------------------------------------------
	
		if ~isstruct(arch), error('<arch> needs to be a struct');end
		if ~isfield(arch,'dataSize'), error('must provide the size of the input');end
		if ~isfield(arch,'inputType');
			arch.inputType = 'binary';
		end

	end
	
	function self = init(self,arch)
	%net = init(arch)
	%--------------------------------------------------------------------------
	%Utility function to used intitialize a convolutional rbm given an archi-
	%tecture.
	%<arch> is a struct with required fields:
	%	.dataSize  --  [#XPixels x #YPixels] size of input data
	%	.inputType -- the class of inputs ('binary' or 'gaussian')
	%
	% <arch> can also contain an options fields <.opt>, which is a cell array
	% of property-value pairs.
	%
	% Returns a mlnn object, <net>.
	%--------------------------------------------------------------------------

		arch = self.ensureArchitecture(arch);

		if isfield(arch,'opts')
			opts = arch.opts;
			fn = fieldnames(self);
			for iA = 1:2:numel(opts)
				if ~isstr(opts{iA})
					error('<opts> must be a cell array string-value pairs.')
				elseif sum(strcmp(fn,opts{iA}))
					self.(opts{iA})=opts{iA+1};
				end
			end
		end

		self.visSize(1) = arch.dataSize(1);
		self.visSize(2) = arch.dataSize(2);
		self.inputType = arch.inputType;
		
		if isfield(arch,'nFM'), self.nFM = arch.nFM; end

		if isfield(arch,'filterSize');
			if numel(arch.filterSize) == 1;
				self.filterSize = ones(1,2)*self.filterSize;
			else
				self.filterSize = arch.filterSize;
			end
		end
		
		weightSize = [self.filterSize(1),self.filterSize(2),self.nFM];
		
		fmSize = self.visSize - self.filterSize + 1;

		self.hidI = zeros(fmSize(1),fmSize(2),self.nFM);
		self.visI = zeros(self.visSize(1),self.visSize(2),self.nFM);

		fanIn = prod(self.visSize);
		fanOut = fanIn*self.nFM;
		range = sqrt(6/(fanIn+fanOut));
		self.W = 2*range*(rand(weightSize)-.5);
		self.dW = zeros(size(self.W));

		self.b = 0; % SINGLE SHARED BIAS FOR ALL VISIBLE UNITS
		self.db = self.b;

		self.c = zeros(self.nFM,1);
		self.dc = zeros(size(self.c));

		self.log.err = zeros(1,self.nEpoch);
		self.log.lRate = self.log.err;

	end

	function [] = save(self,epoch)
	%save(epoch)
	%--------------------------------------------------------------------------
	% Save network at a given epoch
	%--------------------------------------------------------------------------
		r = self;
		if ~exist(r.saveFold,'dir')
			mkdir(r.saveFold);
		end
		save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',epoch)),'r'); clear r;
	end

	function p = sigmoid(self,data)
		% p = sigmoid(X)
	%--------------------------------------------------------------------------
	% Sigmoid activation function
	%--------------------------------------------------------------------------
		p = arrayfun(@(data)(1./(1 + exp(-data))),data);
	end

	function p = drawBernoulli(self,p);
		% p = drawNormal(mu);
	%--------------------------------------------------------------------------
	% Draw samples from a multivariate normal  with mean <mu> and identity
	% covariance.
	%--------------------------------------------------------------------------
		p = double(rand(size(p)) < p);
	end

	function p = drawMultinomial(self,p); % (UNUSED)
	% p = drawNormal(mu);
	%--------------------------------------------------------------------------
	% Draw samples from a multinomial distribution probabilities <p>.
	%--------------------------------------------------------------------------
	
		p = mnrnd(1,p,1);
	end

	function p = drawNormal(self,mu,sigma2); % (UNUSED)
	% p = drawNormal(mu);
	%--------------------------------------------------------------------------
	% Draw samples from a multivariate normal with mean <mu> and identity
	% covariance of dimension <sigma2>.
	%--------------------------------------------------------------------------
		S = eye(numel(sigma2));
		S(find(S)) = sigma2;
		p = mvnrnd(mu,S);
	end

	function out = ff(self,in)
	%out = ff(in)
	%--------------------------------------------------------------------------
	% Flip 1st 2 dimensions of a tensor
	%--------------------------------------------------------------------------
		out = in(end:-1:1,end:-1:1,:);
	end

	function out = tXY(self,in);
	%out = tXY(in);
	%--------------------------------------------------------------------------
	% Transpose 1st 2 dimensions of tensor
	%--------------------------------------------------------------------------
		out = permute(in,[2,1,3]);
	end

	function err = batchErr(self,data);
	%--------------------------------------------------------------------------
	% Calculate squared error on batch data
	%--------------------------------------------------------------------------
		err = sum((data(:)-self.eVis(:)).^2);
	end

	function [] = printProgress(self,type,aux)
	% printProgress(type,aux)
	%--------------------------------------------------------------------------
	% Utility function to display a particular <type> of message. <aux> are 
	% optional auxiliary variables for printing a message.
	%--------------------------------------------------------------------------
	
		switch type
		case 'epoch'
			fprintf('\nEpoch %d/%d',aux(1),aux(2));
		case 'batchErr'
			batchErr = self.auxVars.batchErr;
			jV = self.auxVars.jV;
			nBatch = self.auxVars.nBatch;
			fprintf('\nBatch %d/%d --> mse = %1.2f',jV,nBatch,batchErr);
		case 'epochErr'
			iE = self.auxVars.iE;
			sumErr = self.auxVars.sumErr;
			if iE > 1
				if self.log.err(iE) > self.log.err(iE-1) & iE > 1
					indStr = '(UP)    ';
				else
					indStr = '(DOWN)  ';
				end
			else
				indStr = '';
			end
			fprintf('\n\nEpoch %d/%d --> Total mse: %f %s\n', ...
			iE,self.nEpoch,sumErr,indStr);
		end
	end

	% VISUALIZATION
	function visLearning(self,iE,jV);
		self.visFun(self);
	end

end % END METHODS
end% END CLASSDEF
