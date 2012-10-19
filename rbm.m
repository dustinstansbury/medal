classdef rbm
%-------------------------------------------------------------------
% Restrictied Boltzmann Machine model class
%------------------------------------------------------------------
% DES

properties
	class = 'rbm'; 		% GENERAL CLASS OF MODEL
	type = 'BB';		% TYPE OF RBM ('BB','GB','GG')
	X = [];				% TRAINING DATA
	nObs = [];			% # OF TRAINING OBSERVATIONS
	nVis = [];			% # OF VISIBLE UNITS (DIMENSIONS)
	nHid = 100;			% # OF HIDDEN UNITS
	W = [];				% CONNECTION WEIGHTS
	dW = []				% LEANING INCREMENT FOR CONN. WEIGHTS
	b = [];				% VISIBLE UNIT BIASES
	db = [];			% LEARNING INCREMENT FOR VIS. BIAS
	c = [];				% HIDDEN UNIT BIASES
	dc = [];			% LEARNING INCREMENT FOR HID. BIAS
	log = [];			% ERROR AND LEARNING RATE LOGS
	aVis = [];			% VISIBLE LAYER ACTIVATIONS
	pVis = [];			% VISIBLE LAYER PROBS
	aHid = [];			% HIDDEN LAYER ACTIVATION
	pHid = [];			% HIDDEN LAYER PROBS
	pHid0 = [];			% INITIAL HIDDEN LAYER PROBS
	aHid0 = [];			% INITIAL HIDDEN LAYER ACTIVATION
	eHid = [];			% EXPECTATION OF HIDDEN STATE (POST TRAINING)
	eta = 0.1;			% LEARNING RATE
	etaFinal = 1e-8;	% FINAL LEARNING RATE
	batchIdx = [];		% BATCH INDICES INTO TRAINING DATA
	z = [];				% LOG VARIANCES (GAUSS UNITS)
	dz = [];			% LEARNING INCREMENT FOR LOG VARIANCES
	sigma2 = [];		% VARIANCE OF (G-) UNIT
	sampleVis = 0;		% SAMPLE THE VISIBLE UNITS
	sampleHid = 1;		% SAMPLE HIDDEN UNITS 
	momentum = 0.5;		% MOMENTUM TERM FOR WEIGHT ESTIMATION
	nEpoch = 1000;		% # OF FULL PASSES THROUGH TRIANING DATA
	wDecay = 0.0002;	% WEIGHT DECAY
	wPenalty = 0;		% CURRENT WEIGHT PENALTY
	sparsity = 0;		% SPARSENESS FACTOR
	batchSz = 100;		% # OF TRAINING POINTS PER BATCH
	nGibbs = 1;			% CONTRASTIVE DIVERGENCE (1)
	anneal = 0;			% # OF EPOCHS TO START SIMULATED ANNEALING
	varyEta = .1;		% VARY LEARNING RATE
	verbose = 1;		% DISPLAY PROGRESS
	saveEvery = 0;		% # OF EPOCHS TO SAVE INTERMEDIATE MODELS
	displayEvery = 50;	% DIPLAY EVERY # UPDATES
	visFun = [];		% USER-DEFINED FUNCTION ('@myFun')
	auxVars = []; 		% AUXILLARY VARIABLES, JUST IN CASE
	centerData = 0;		% SUBTRACT MEAN FROM INPUTS
	scaleData = 0;		% SCALE DATA BY STDEV
	chkConverge = 0;	% FLAG FOR CHECKING CONVERBENCE
	useGPU = 1; 		% USE CUDA, IF AVAILABLE
	gpuDevice = [];		% GPU DEVICE STRUCTURE
	saveFold='./rbmSave';% # DEFAULT SAVE FOLclass14DER
		
end % END PROPERTIES

methods
	% CONSTRUCTOR
	function self = rbm(args, data)
		if notDefined('data')
			data = [];
		end
		if ~nargin
			self = self.defaultRBM;
		elseif strcmp(args,'empty');
		% PASS AN EMPTY RBM 
		else
			self = self.init(args,data);
		end
	end

	function [] = print(self)
		properties(self)
		methods(self)
	end
	
	%---------------------------------------------------------------------
	% MAIN
	function self = train(self)
		eta0 = self.eta;
		etaFinal = self.etaFinal;
		dCount = 1;
		for iE = 1:self.nEpoch
			sumErr = 0;

			% (LINEAR) SIMULATED ANNEALING
			if self.anneal
				self.wPenalty = self.wPenalty/max(1,iE/self.anneal);j
			end

			% LOOP OVER BATCHES
			for jB = 1:numel(self.batchIdx)
				X = self.X(self.batchIdx{jB},:);
				self = self.runGibbs(X);
				self = self.updateParams(X);
				sumErr = self.accumErr(X,sumErr);
				
				if ~isempty(self.visFun) & ~mod(dCount,self.displayEvery)&iE>1;
					self.visLearning(iE-1,jB);
				end
				dCount = dCount+1;
			end

			% SPARSITY
			if self.sparsity
				dcSparse = -self.eta*(mean(self.pHid)-self.sparsity);
				self.c = self.c + dcSparse;
			end
			self.log.err(iE) = sumErr;
			self.log.eta(iE) = self.eta;
			
			% DECREASE LEARNING RATE
		   if self.varyEta
%  				self.eta = eta0*exp(-self.varyEta*iE);
				self.eta = eta0 - 1.0*(iE/self.nEpoch)*eta0;
			end

			
			if self.verbose
				self.printProgress(sumErr,iE,jB);
			end
			
			if iE > 1
				if iE == self.saveEvery
					r = self;
					if ~exist(r.saveFold,'dir')
						mkdir(r.saveFold);
					end
					save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',iE)),'r'); clear r;
				end
				if iE > 10 && self.chkConverge
					if self.converged(self.e(iE-10:iE))
						self.log.err(iE+1:end) = [];
						self.log.eta(iE+1:end) = [];
						break
					end
				end
			end

		end
		
		% HIDDEN UNIT EXPECTATIONS 
		switch self.type(2)
		case 'B'
			self.eHid = self.sigmoid(bsxfun(@plus,self.X*self.W,self.c));
		case 'G'
			self.eHid = bsxfun(@plus,self.X*self.W,self.c);
		end
		
		% PULL DATA FROM GPU, IF NEEDED
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
			self.gpuDevice = [];
		end
		fprintf('\n');
	end

	%---------------------------------------------------------------------
	% MAIN GIBBS SAMPLER
	function self = runGibbs(self,X,nGibbs)
		if notDefined('nGibbs'), nGibbs = self.nGibbs;end;
		nObs = size(X,1);
		for iC = 1:nGibbs
			% CD[1]
			% GO UP
			if iC == 1
				[self.pHid,self.aHid] = self.hidGivVis(X,1);
				self.pHid0 = self.pHid;
				self.aHid0 = self.aHid;
			else
				[self.pHid, self.aHid] = self.hidGivVis(X,0);
			end
			% GO DOWN
			[self.pVis,self.aVis] = self.visGivHid(self.aHid,self.sampleVis);
			X = self.aVis;
			% FINISH 
			if iC == self.nGibbs
				[self.pHid,self.aHid] = self.hidGivVis([],0);
			end
		end
	end

	%---------------------------------------------------------------------
	% p(H|V), SAMPLE H, IF NEEDED
	function [pHid,aHid] = hidGivVis(self,X,sampleHid)
		if notDefined('X'), X = self.aVis;end
		switch self.type
			case 'BB'
				pHid = self.sigmoid(bsxfun(@plus,X*self.W,self.c));
				if sampleHid
					aHid = single(pHid>rand(size(X,1),self.nHid));
				else
					aHid = pHid;
				end
			case 'GB'
				pHid = self.sigmoid(bsxfun(@plus,X*self.W,self.c));
				if sampleHid
					aHid = single(pHid>rand(size(X,1),self.nHid));
				else
					aHid = pHid;
				end
		end
	end
	
	%-------------------------------------------------------------------
	% p(V|H)
	function [pVis,aVis] = visGivHid(self,aHid,sampleVis)
		nObs = size(aHid,1);
		switch self.type
			case {'BB'}
				pVis = self.sigmoid(bsxfun(@plus,aHid*self.W',self.b));
				if sampleVis
					aVis = pVis>rand(nObs,self.nVis);
				else
					aVis = pVis;
				end
			case 'GB'
				mu = bsxfun(@plus,aHid*self.W',self.b);
				pVis = self.drawNormal(mu);
				
				if sampleVis
					aVis = pVis;
				else
					aVis = mu;
				end
			case 'GG'
				error('GG model not implemented yet')
		end
	end
	%-------------------------------------------------------------------
	% LEARNING RULES
	function self = updateParams(self,X);
		nObs = size(X,1);
		switch self.type
		% BERNOULLI-BERNOULLI UNITS
		case 'BB'
			dW=(X'*self.pHid0 - self.aVis'*self.pHid)/nObs;
			self.dW=self.momentum*self.dW + (1-self.momentum)*dW;
			self.W = self.W + self.eta*self.dW - self.wPenalty*self.W; 

			db = mean(X) - mean(self.pVis);
			self.db = self.momentum*self.db + self.eta*db;
			self.b = self.b + self.db; 

			dc = mean(self.pHid0) - mean(self.pHid);
			self.dc = self.momentum*self.dc + self.eta*dc;
			self.c = self.c + self.dc; 

		% GAUSSIAN-BERNOULLI UNITS 
		case 'GB'
			% CONNECTION WEIGHTS
			dW=bsxfun(@rdivide,(X'*self.pHid0 - self.aVis'*self.pHid),nObs');
			self.dW=self.momentum*self.dW + self.eta*dW*(1-self.momentum) - self.wPenalty*self.W;
			self.W = self.W + self.dW;

			% VISIBLE BIASES
			db = mean(X) - mean(self.aVis);
			self.db = self.momentum*self.db + self.eta*db;
			self.b = self.b + self.db;

			% HIDDEN BIASES
			dc = mean(self.pHid0) - mean(self.pHid); 
			self.dc = self.momentum*self.dc + self.eta*dc;
			self.c = self.c + self.dc;


		end
	end

	% ACCUMULATE ERROR
	function err = accumErr(self,X,err0);
		err = sum(sum((X-self.aVis).^2));
		err = err + err0;
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
		
		self.log.err = zeros(1,self.nEpoch);
		self.log.eta = self.log.err;
		[self.nObs,self.nVis]=size(data);
		
		switch self.type
		case {'GB'}
			self.W = (2/(self.nHid+self.nVis))*rand(self.nVis,self.nHid) - ...
			1/(self.nVis + self.nHid);
		case 'BB'
			self.W = 1/sqrt(self.nVis +  self.nHid)* ...
			2*(rand(self.nVis,self.nHid)-.5);
		end
		self.dW = zeros(size(self.W));
		self.b = zeros(1,self.nVis);
		self.db = zeros(size(self.b));
		self.c = zeros(1,self.nHid);
		self.dc = zeros(size(self.c));
		
		% INIT. LOG VARIANCES (GAUSSIAN UNITS)
		if strcmp(self.type(1),'G') % IF VISIBLE
			self.sigma2 = diag(ones(1,self.nVis));
		end
		if self.centerData
			data = bsxfun(@minus,data,mean(data,1));
		end
		if self.scaleData
			data = bsxfun(@rdivide,data,std(data,1));
		end
		self.X = data;
		self.batchIdx = self.createBatches;

		if self.useGPU
			self = gpuDistribute(self);
		end	
	end

	function batchIdx = createBatches(self)
		nBatches = ceil(self.nObs/self.batchSz);
		tmp = repmat(1:nBatches, 1, self.batchSz);
		tmp = tmp(1:self.nObs);
		randIdx=randperm(self.nObs);
		tmp = tmp(randIdx);
		for iB=1:nBatches
		    batchIdx{iB} = find(tmp==iB);
		end
	end

	function rbm = defaultRBM(self)
		% EDIT THIS FILE FOR DEFAULT ARGUMENTS
		args = defaultRBM();
		rbm = self.init(args);
		fprintf('\nNote: using default arguments (see defaultRBM.m)\n\n');
	end
	
	function p  = sigmoid(self,X)
		p = arrayfun(@(x)(1./(1 + exp(-x))),X);
	end

	function p = drawNormal(self,mu);
		p = mvnrnd(mu,self.sigma2);
	end

	% VISUALIZATION
	function visLearning(self,iE,jB);
		if isempty(self.visFun)
			switch self.type
				case 'BB'
					visBBLearning(self,iE,jB);
				case 'GB'
					visGBLearning(self,iE,jB);
			end
		else
			self.visFun(self,iE,jB);
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

	% DRAW SAMPLE FROM MODEL (RUN GIBBS)
	function samps = sample(self,data,nSamples,nIters)
		self.sampleVis = 0;
		self.sampleHid = 0;
		if notDefined('nSamples');nSamples = 1;end
		if notDefined('nIters'),nIters = 10;end
		[nObs,nVis] = size(data);
%  		samps = repmat(zeros(size(data)),1,nSamples);
		samps = zeros(nObs,nVis,nSamples);

		if self.useGPU
			samps = gpuArray(single(samps));
			self = gpuDistribute(self);
		end
		
%  			samps = reshape(samps,nObs,nVis,nSamples);
%  		end

		for iS = 1:nSamples
			vis = data;
			for iI = 1:nIters
				switch self.type
				case 'BB'
					hid = self.sigmoid(bsxfun(@plus,binornd(1,vis)* ...
					self.W,self.c));
					vis = self.sigmoid(bsxfun(@plus,binornd(1,hid)* ...
					self.W',self.b));
				case 'GB'
					hid = self.sigmoid(bsxfun(@plus,binornd(1,vis)* ...
					self.W,self.c));
					vis = bsxfun(@plus,hid*self.W',self.b);
				end
			end
			samps(:,:,iS)=vis;
		end
		if self.useGPU
			samps = gather(samps);
		end
	end

	function [F] = freeEnergy(self,X)
		nSamps = size(X,1);
		upBound = 50; loBound = -50; % FOR POSSIBLE OVERFLOW
		switch self.type(1)
			case 'B'
				H = ones(nSamps,1)*self.c + X*self.W;
				H = max(min(H,upBound),loBound);
				% SAMPLE ENERGIES
				sampE = -X*self.b' - sum(-log(1./exp(H)),2);
				F = mean(sampE);
			case 'G'
				H = ones(nSamps,1)*self.c + X*self.W;
		             
				H = max(min(H,upBound),loBound);
				% SAMPLE ENERGIES
				sampE = bsxfun(@minus,X,self.b);
				sampE = sum(sampE.^2,2)/2;
				sampE = sampE - sum(-log(1./(1+exp(H))),2);
				F = mean(sampE);
		end
	end

	function success = converged(self,errors);
		success = (abs(mean(gradient(smooth(errors)))/max(self.e)) <= 0.0001);
		if success
			fprintf('\nCONVERGED.\n');
		end
	end

	% VISUALIZE CONNECTION WEIGHTS
	function im = vis(self,visData,lims);
		if notDefined('visData')
			visData = self.W;
		end
		if notDefined('lims')
			lims = [];
		end
		switch self.type
			case 'BB'
				im = visWeights(visData,1,lims,1);
			case 'GB'
				im = visWeights(visData,0,lims,1);
		end
	end
end % END METHODS
end % END CLASSDEF