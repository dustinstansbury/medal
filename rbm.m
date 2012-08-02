classdef rbm
%-------------------------------------------------------------------
% Restrictied Boltzmann Machine model class
%------------------------------------------------------------------
% DES

properties

	type = 'BB';		% TYPE OF RBM ('BB','GB','GG')
	X = [];				% TRAINING DATA
	sigmaX = []; 		% STDEV OF TRAINING DATA
	nObs = [];			% # OF TRAINING OBSERVATIONS
	nVis = [];			% # OF VISIBLE UNITS (DIMENSIONS)
	nHid = 100;			% # OF HIDDEN UNITS
	W = [];				% CONNECTION WEIGHTS
	dW = []				% LEANING INCREMENT FOR CONN. WEIGHTS
	b = [];				% VISIBLE UNIT BIASES
	db = [];			% LEARNING INCREMENT FOR VIS. BIAS
	c = [];				% HIDDEN UNIT BIASES
	dc = [];			% LEARNING INCREMENT FOR HID. BIAS
	e = [];				% RECONSTRUCTION ERROR OVER EPOCHS
	a = [];				% WEIGHT DECAY RATE OVER EPOCHS
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
	batchX = [];		% CURRENT BATCH DATA
	z = [];				% LOG VARIANCES (GAUSS UNITS)
	dz = [];			% LEARNING INCREMENT FOR LOG VARIANCES
	sigma2 = [];		% CURRENT ESTIMATE OF VARIANCE OF (G-) UNIT
	dSigma2 = [];		% LEARNING INCREMENT FOR VARIANCE
	learnSigma2 = 0;	% FLAG FOR LEARNING GAUSSIAN VARIANCES
	sampleVis = 0;		% SAMPLE THE VISIBLE UNITS
	sampleHid = 1;		% SAMPLE HIDDEN UNITS 
	momentum = 0.5;		% MOMENTUM TERM FOR WEIGHT ESTIMATION
	nEpoch = 1000;		% # OF FULL PASSES THROUGH TRIANING DATA
	wDecay = 0.0002;	% WEIGHT DECAY
	wPenalty = 0;		% CURRENT WEIGHT PENALTY
	sparse = 0;			% SPARSENESS FACTOR
	batchSz = 100;		% # OF TRAINING POINTS PER BATCH
	nGibbs = 1;			% CONTRASTIVE DIVERGENCE (1)
	anneal = 0;			% FLAG FOR SIMULATED ANNEALING
	varyEta = .1;		% VARY LEARNING RATE
	verbose = 1;		% DISPLAY PROGRESS
	saveEvery = 0;		% # OF EPOCHS TO SAVE INTERMEDIATE MODELS
	displayEvery = 50;	% DIPLAY EVERY # UPDATES
	visFun = [];		% USER-DEFINED FUNCTION ('@myFun')
	centerData = 0;		% SUBTRACT MEAN FROMclass14 INPUTS
	scaleData = 0;		% SCALE DATA BY STDEV
	chkConverge = 0;	% FLAG FOR CHECKING CONVERBENCE
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
			self.wPenalty = wPenalty0 - 0.9*(iE/self.nEpoch)*wPenalty0;
		end

			% LOOP OVER BATCHES
			for jB = 1:numel(self.batchIdx)
				X = self.X(self.batchIdx{jB},:);
				self = self.runGibbs(X);
				self = self.updateParams(X);
				sumErr = self.accumErr(X,sumErr);
				
				if self.verbose & ~mod(dCount,self.displayEvery)&iE>1;
					self.visLearning(iE-1,jB);
				end
				dCount = dCount+1;
			end

			% SPARSITY
			if self.sparse
				dcSparse = -self.eta*(mean(self.pHid)-self.sparse);
				self.c = self.c + dcSparse;
			end
			self.e(iE) = single(sumErr);
			self.a(iE) = self.eta;
			
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
						self.e(iE+1:end) = [];
						self.a(iE+1:end) = [];
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
			[self.pVis,self.aVis] = self.visGivHid();
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
		if notDefined('sampleHid'), sampleHid = self.sampleHid;end
		switch self.type
			case 'BB'
				pHid = self.sigmoid(bsxfun(@plus,X*self.W,self.c));
				if sampleHid
					aHid = pHid>rand(size(X,1),self.nHid);
				else
					aHid = pHid;
				end
			case 'GB'
				scaledX = bsxfun(@rdivide,X,self.sigma2);
				pHid = self.sigmoid(bsxfun(@plus,scaledX*self.W,self.c));
				if sampleHid
					aHid = pHid>rand(size(X,1),self.nHid);
				else
					aHid = pHid;
				end
			case 'BG'
				mu = bsxfun(@plus,X*self.W,self.c);
				pHid = self.drawNormal(mu,self.sigma2);
				if sampleHid
					aHid = pHid;
				else
					aHid = mu;
				end
			case 'GG'
				error('GG model not implemented yet')
		end
	end
	
	%-------------------------------------------------------------------
	% p(V|H)
	function [pVis,aVis] = visGivHid(self,aHid,sampleVis)
		if notDefined('aHid'),aHid = self.aHid;end
		if notDefined('sampleVis'),sampleVis = self.sampleVis;end
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
				pVis = self.drawNormal(mu,self.sigma2);
				
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
			sigma2 = repmat(self.sigma2,nObs,1);
			% CONNECTION WEIGHTS
			dW=bsxfun(@rdivide,(X'*self.pHid0 - self.aVis'*self.pHid),(nObs*self.sigma2)'); 
			self.dW=self.momentum*self.dW + self.eta*dW*(1-self.momentum) - self.wPenalty*self.W;
			self.W = self.W + self.dW;

			% VISIBLE BIASES
			db = mean(X./sigma2) - mean(self.aVis./sigma2); 
			self.db = self.momentum*self.db + self.eta*db;
			self.b = self.b + self.db;

			% HIDDEN BIASES
			dc = mean(self.pHid0) - mean(self.pHid); 
			self.dc = self.momentum*self.dc + self.eta*dc;
			self.c = self.c + self.dc;

			if self.learnSigma2
				% HERE WE LEARN LOG VARIANCES TO ENFORCE
				% POSITIVITY CONSTRAINT ON VARIANCES
				tmp = bsxfun(@minus,X,self.b).^2;
				EzDat = bsxfun(@rdivide,mean(tmp-(X.*(self.pHid0*self.W'))),self.sigma2);

				tmp = bsxfun(@minus,self.aVis,self.b).^2;
				EzMod = bsxfun(@rdivide,mean(tmp-(self.aVis.*(self.pHid*self.W'))),self.sigma2);
				% DERIVATIVE OF LOG-VARIANCE
				dz = bsxfun(@times,(EzDat-EzMod),exp(-self.z)); 
				self.dz = self.momentum*self.dz + self.eta*dz -self.wPenalty*self.z;
				self.z = self.z + self.dz;
				self.z(isnan(self.z)) = 1e-6;
				self.z(find(self.z>=0)) = 1e-6;
				% UPDATE THE VARIANCES
				self.sigma2 = exp(self.z) + 1e-8;
			end
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
		
		self.e = zeros(1,self.nEpoch);
		self.a = self.e;
		[self.nObs,self.nVis]=size(data);
		
		switch self.type
		case {'GB','BG'}
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
			self.sigma2 = ones(1,self.nVis);
			self.z = log(self.sigma2);
			self.dz = zeros(size(self.z));
		elseif strcmp(self.type(2),'G'); % IF HIDDEN
			self.sigma2 = ones(1,self.nHid);
			self.z = log(self.sigma2);
			self.dz = zeros(size(self.z));
		end
		if self.centerData
			data = bsxfun(@minus,data,mean(data,1));
		end
		if self.scaleData
			data = bsxfun(@rdivide,data,std(data,1));
		end
		self.X = data;
		self.batchIdx = self.createBatches;
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
		p = 1./(1 + exp(-X));
	end

	function p = drawNormal(self,mu,sigma2);
		S = eye(numel(sigma2));
		S(find(S)) = sigma2;
		p = mvnrnd(mu,S);
	end

	% VISUALIZATION
	function visLearning(self,iE,jB);
		if isempty(self.visFun)
			switch self.type
				case 'BB'
					visBBLearning(self,iE,jB);
				case 'GB'
					visGBLearning(self,iE,jB);
				case 'BG'
					visBGLearning(self,iE,jB);
			end
		else
			self.visFun(self,iE,jB);
		end
	end

	% VERBOSE
	function [] = printProgress(self,sumErr,iE,jB)	
		if iE > 1
			if self.e(iE) > self.e(iE-1) & iE > 1
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
		samps = repmat(zeros(size(data)),1,nSamples);
		
%  		if size(samps,1) ==1
			samps = reshape(samps,nObs,nVis,nSamples);
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
				case 'BG'
					hid = self.drawNormal(bsxfun(@plus,hid*self.W,self.c),self.sigma2);
					vis = self.sigmoid(bsxfun(@plus,binornd(1,hid)* ...
					self.W',self.b));
				end
			end
			samps(:,:,iS)=vis;
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
				H = ones(nSamps,1)*self.c + ...
		             bsxfun(@rdivide,X,self.sigma2)*self.W;
		             
				H = max(min(H,upBound),loBound);
				% SAMPLE ENERGIES
				sampE = bsxfun(@minus,X,self.b);
				sampE = sum(bsxfun(@rdivide,sampE.^2,self.sigma2),2)/2;
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
			case 'BG'
				im = visWeights(visData,1,lims,1);
		end
	end
end % END METHODS
end % END CLASSDEF