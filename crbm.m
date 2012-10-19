classdef crbm
% A convolutional Restricted Boltzmann Machine Object
properties
	class = 'crbm'
	type = 'BB'
	X
	nObs
	visSize
	W
	dW
	b
	db
	c
	dc
	eVis
	eHid
	eHid0
	hidI
	visI
	ePool
	visFun = [];
	nEpoch = 10;
	nFeats = 12;
	featSize = [7 7];
	poolSize = [2 2];
	eta = 0.1;
	nGibbs = 1;
	displayEvery = 100;
	checkConverged = 0;
	anneal = 0;
	sparsity = 0.03;
	momentum = 0.9;
	wPenalty = .01;
	saveFold = './convRBMSave';
	chkConverge = 0;
	log = struct();
	verbose = 1;
	visLearn = 0;
	varyEta = 0;
	saveEvery = 500;
	dataPoint = 0;
	dcSparse=0;
	useGPU = 1;
	gpuDevice;
end % END PROPERTIES

methods
	
	function self = crbm(args,data);
	% CONSTRUCTOR
		if notDefined('data'),data = [];end
		if ~nargin,
			self = self.default;
		elseif strcmp(args,'empty')
			% PASS AN EMPTY cRBM
		else
			self = self.init(args,data);
		end
	end

	function [] = print(self)
		properties(self);
		methods(self);
	end

	function self = train(self)
		eta0 = self.eta;
		wPenalty0 = self.wPenalty;
		dCount = 1;
		
		for iE = 1:self.nEpoch
			sumErr = 0;
			% (LINEAR) SIMULATED ANNEALING
			if self.anneal
				self.wPenalty = self.wPenalty/max(1,iE/self.anneal);
			end

			% LOOP OVER INPUTS (WE CONSIDER EACH IMAGE A "MINIBATCH")
			for jV = 1:self.nObs
				self.dataPoint = jV;
				X = self.X(:,:,jV);
				[self,dW,db,dc] = self.calcGradients(X);
				
				self = self.applyGradients(dW,db,dc);

				sumErr = self.accumErr(X,sumErr);
				if ~isempty(self.visFun) && ~mod(dCount,self.displayEvery);
					self.visLearning();
				end

				if self.verbose && ~mod(dCount,self.displayEvery);
					self.printProgress(sumErr,iE,jV);
				end
				
				% INDUCE HIDDEN UNIT SPARSITY (Eq 5.5; Lee, 2010)
				if self.sparsity
					self.dcSparse = self.eta*squeeze(self.sparsity - mean(mean(self.eHid0)));
					self.c = self.c + self.dcSparse;
				end

				dCount = dCount+1;
			end
			self.log.err(iE) = single(sumErr);
			self.log.eta(iE) = self.eta;

			% (LINEARLY) DECREASE LEARNING RATE
			if self.varyEta
				self.eta = eta0 - 0.9*(iE/self.nEpoch)*eta0;
			end

			if iE > 1
				if iE == self.saveEvery
					self.save(iE)
				end
				
				if iE > 10 && self.chkConverge
					if self.converged(self.e(iE-10:iE))
						self.e(iE+1:end) = [];
						self.a(iE+1:end) = [];
						break
					end
				end
			end
		end
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
			self.gpuDevice = [];
		end
		fprintf('\n');
	end

	function [self,dW,db,dc] = calcGradients(self,X)
		self = self.runGibbs(X);
		for iK = 1:self.nFeats
			dW(:,:,iK)  = (conv2(X,self.ff(self.eHid0(:,:,iK)),'valid') - ...
						   conv2(self.eVis,self.ff(self.eHid(:,:,iK)),'valid'));
		end
		db = sum(sum(X - self.eVis));
		dc = squeeze(sum(sum(self.eHid0 - self.eHid)));
	end

	function self = runGibbs(self,X)
		
		% INITIAL PASS
		[self,self.eHid0] = self.hidGivVis(X);
		for iC = 1:self.nGibbs
			% RECONSTRUCT VISIBLES
			self = self.visGivHid(self.drawBernoulli(self.eHid));
			% FINISH CD[n]
			self = self.hidGivVis(self.eVis);
		end
	end
	
	function self = applyGradients(self,dW,db,dc)
		[self.W,self.dW] = self.updateParams(self.W,dW,self.dW,self.wPenalty);
		[self.b,self.db] = self.updateParams(self.b,db,self.db,0);
		[self.c,self.dc] = self.updateParams(self.c,dc,self.dc,0);
	end

	function [params,grad] = updateParams(self,params,grad,gradPrev,wPenalty)
		
		grad = self.momentum*gradPrev + (1-self.momentum)*grad;
		params = params + self.eta*(grad - wPenalty*params);
	end

	function [self,eHid] = hidGivVis(self,vis)
	% CALCULATE HIDDEN EXPECTATIONS GIVEN VISIBLE UNITS
		for iK = 1:self.nFeats
			self.hidI(:,:,iK) = exp(conv2(vis,self.ff(self.W(:,:,iK)),'valid')+self.c(iK));
		end
		eHid = self.hidI./(1 + self.pool(self.hidI)); % (Eq 5.3; Lee, 2010)
		self.eHid = eHid;
	end

	function self = visGivHid(self,hid)  
	% CALCULATE VISIBLE EXPECTATIONS GIVEN EACH HIDDEN FEATURE MAP
		for iK = 1:self.nFeats
			self.visI(:,:,iK) = conv2(hid(:,:,iK),self.W(:,:,iK),'full');
		end
		I = sum(self.visI,3) + self.b;
		if strcmp(self.type,'BB')
			self.eVis = self.sigmoid(I);
		else
			self.eVis = I;
		end
		return
	end
	
	function self = poolGivVis(self,vis) % (FOR SAMPLING & DBNs -- NOT CURRENTLY USED)
	% CALCULATE POOLING LAYER EXPECTATIONS GIVEN VISIBLES
		I = zeros(self.hidSize(1),self.hidSize(2),self.nFeats);
		for iK = 1:self.nFeats
			I(:,:,iK) = exp(convn(vis,self.ff(W(:,:,iK)),'valid')+self.c(iK));
		end
		self.ePool = 1-(1./(1+self.pool(exp(I)))); % (Eq 5.4; Lee, 2010)
	end

	function blocks = pool(self,I); % [[hiddenSize]/poolSize x nFeatures]
	% POOL HIDDEN ACTIVATIONS INTO BLOCKS
		hCols = size(I,1);
		hRows = size(I,2);
		pRows = self.poolSize(1);
		pCols = self.poolSize(2);
		blocks = zeros(size(I));
		% DEFINE BLOCKS B_alpha
		for iR = 1:ceil(hRows/pRows)
			rows = (iR-1)*pRows+1:iR*pRows;
			for jC = 1:ceil(hCols/pCols)
				cols = (jC-1)*pCols+1:jC*pCols;
				% MAIN POOLING
				blockVal = squeeze(sum(sum(I(rows,cols,:))));
				blocks(rows,cols,:) = repmat(permute(blockVal, ...
				 [2,3,1]),numel(rows),numel(cols));
			end
		end
	end

	function self = init(self,args,data)
	
		% PARSE ARGUMENTS/OPTIONS
		if ~iscell(args) || mod(numel(args),2)
			error('<args> must be a cell array of string-value pairs.')
		end
		fn = fieldnames(self);
		for iA = 1:2:numel(args)
			if ~isstr(args{iA})
				error('<args> must be a cell array string-value pairs.')
			elseif sum(strcmp(fn,args{iA}))
				self.(args{iA})=args{iA+1};
			end
		end

		% INITIALIZE MODEL PARAMS FROM DATA
		if notDefined('data')
			defaultDataFile = 'defaultData.mat';
			load(defaultDataFile);
			fprintf('\nNote: using default dataset in: \n--> %s\n',defaultDataFile);
		end
		
		self.X = data;
		[self.visSize(1),self.visSize(2),self.nObs]=size(data);

		if numel(self.featSize) == 1;
			self.featSize = ones(1,2)*self.featSize;
		end

		weightSize = [self.featSize(1),self.featSize(2),self.nFeats];
		
		convSize = self.visSize - self.featSize + 1;
		
		self.hidI = zeros(convSize(1),convSize(2),self.nFeats);
		self.visI = zeros(self.visSize(1),self.visSize(2),self.nFeats);


		self.W = randn(weightSize)/sqrt(prod(self.visSize)/self.nFeats);
		self.dW = zeros(size(self.W));
		
		self.b = 0; % SINGLE SHARED BIAS FOR ALL VISIBLE UNITS
		self.db = self.b;
		
		self.c = zeros(self.nFeats,1)*2;
		self.dc = zeros(size(self.c));
		
		self.log.err = zeros(1,self.nEpoch);
		self.log.eta = self.log.err;
		if self.useGPU
			self = gpuDistribute(self);
		end
		
	end

	function [] = save(self,epoch)
	% SAVE NETWORK AT A GIVEN EPOCH
		r = self;
		if ~exist(r.saveFold,'dir')
			mkdir(r.saveFold);
		end
		save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',epoch)),'r'); clear r;
	end

	function c = default(self)
	% EDIT defaultCRBM.m FOR DEFAULT ARGUMENTS
		args = defaultCRBM();
		c = self.init(args);
		fprintf('\nNote: using default arguments (see defaultRBM.m)\n\n');
	end

	function p = sigmoid(self,X)
		p = arrayfun(@(x)(1./(1 + exp(-x))),X);
	end

	function p = drawBernoulli(self,p);
		p = double(rand(size(p)) < p);
	end

	function p = drawMultinomial(self,p); % (UNUSED)
		p = mnrnd(1,p,1);
	end

	function p = drawNormal(self,mu,sigma2); % (UNUSED)
		S = eye(numel(sigma2));
		S(find(S)) = sigma2;
		p = mvnrnd(mu,S);
	end

	function out = ff(self,in)
	% FLIP FIRST 2 DIMENSIONS OF A 3 TENSOR 
		out = in(end:-1:1,end:-1:1,:);
	end

	function out = tXY(self,in);
	% TRANSPOSE 1ST TWO DIMENSIONS OF 3 TENSOR
		out = permute(in,[2,1,3]);
	end

	function err = accumErr(self,X,err0);
	% ACCUMULATE ERROR
		err = sum(sum((X-self.eVis).^2));
		err = err + err0;
	end

	function success = converged(self,errors);
	% CHECK CONVERGENCE (BASED ON AVERAGE LOCAL GRADIENT)
		errors = errors(end:-1:1);
		success = abs(mean(diff(errors)./errors(1:end-1))) <= 1e-4;
		if success
			fprintf('\nCONVERGED.\n');
		end
	end
	
	function [] = printProgress(self,sumErr,iE,jB)
	% PRINT LEARNING PROGRESS
		if iE > 1
			if self.log.err(iE) > self.log.err(iE-1) & iE > 1
				indStr = '(UP)    ';
			else
				indStr = '(DOWN)  ';
			end
		else
			indStr = '';
		end
		fprintf('Epoch %d/%d Sample %d/%d--> Recon. error: %f %s\r', ...
		iE,self.nEpoch,self.dataPoint,self.nObs,sumErr,indStr);
	end

	% VISUALIZATION
	function visLearning(self,iE,jV);
		if isempty(self.visFun)
			switch self.type
				case 'BB'
					visCBBLearning(self,self.X(:,:,self.dataPoint));
				case 'GB'
				visCBBLearning(self,self.X(:,:,self.dataPoint));
			end
		else
			self.visFun(self,self.X(:,:,self.dataPoint));
		end
	end

end % END METHODS
end% END CLASSDEF
