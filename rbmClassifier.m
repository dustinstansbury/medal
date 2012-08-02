classdef rbmClassifier
%-------------------------------------------------------------------
% Restrictied Boltzmann Machine Classifier model class
%------------------------------------------------------------------
% DES

properties

	type = 'BB';		% TYPE OF RBM ('BB','GB','GG')
	X = [];				% TRAINING DATA
	Y = [];				% TRAINING CLASSES
	labels = [];		% UNIQUE LABELS
	targs = [];			% TRAINING CLASSES AS BINARY VECTORS
	nObs = [];			% # OF TRAINING OBSERVATIONS
	nVis = [];			% # OF VISIBLE UNITS (DIMENSIONS)
	nHid = [];			% # OF HIDDEN UNITS
	nClasses = [];		% # OF POSSIBLE CLASSES
	W = [];				% CONNECTION WEIGHTS
	dW = []				% LEANING INCREMENT FOR CONN. WEIGHTS
	classW = [];		% CLASSIFICATION WEIGHTS
	dClassW = []		% GRADIENT FOR CLASS. WEIGHTS
	b = [];				% VISIBLE UNIT BIASES
	db = [];			% GRADIENT FOR VIS. BIAS
	c = [];				% HIDDEN UNIT BIASES
	dc = [];			% GRADIENT FOR HID. BIAS
	d = [];				% CLASSIFIER BIAS
	dd = [];			% GRADIENT FOR CLASS. BIAS
	e = [];				% RECONSTRUCTION ERROR
	a = [];				% LEARNING RATE PER EPOCH
	aVis = [];			% VISIBLE LAYER ACTIVATIONS
	pVis = [];			% VISIBLE LAYER PROBS
	aHid = [];			% HIDDEN LAYER ACTIVATION
	pHid = [];			% HIDDEN LAYER PROBS
	pHid0 = [];			% INITIAL HIDDEN LAYER PROBS
	pClass = [];		% CLASS INDICATOR PROBS
	aClass = [];		% CLASS INDICATOR ACTIVATIONS
	batchIdx = [];		% BATCH INDICES INTO TRAINING DATA
	sigma2 = [];		% VARIANCE (GAUSSIAN UNITS)
	dSigma2 = [];		% LEARNING INCRENMENT FOR VARIANCE
	z = [];				% LOG VARINACES (GAUSSIAN UNITS)
	dz = [];			% GRADIENT OF LOG VARIANCES 
	learnSigma2=[];		% FLAG FOR LEARNING GUASSIAN VARIANCE
	sampleVis = 1;		% SAMPLE VISIBLE UNITS (GAUSSIAN)
	eta = 0.1;			% (INITIAL) LEARNING RATE (ALL PARAMS)
	etaFinal=1e-6;		% FINAL LEARNING RATE
	momentum = 0.5;		% MOMENTUM TERM FOR WEIGHT ESTIMATION
	nEpoch = 100;		% # OF FULL PASSES THROUGH TRIANING DATA
	wDecay = 0.0002;	% WEIGHT DECAY
	wPenalty = 0.0002	% CURRENT WEIGHT PENALTY
	sparse = 0;			% SPARSENESS TERM (HIDDEN UNITS)
	batchSz = 100;		% # OF TRAINING POINTS PER BATCH
	nGibbs = 1;			% CONTRASTIVE DIVERGENCE (1)
	anneal = 1;			% FLAG FOR SIMULATED ANNEALING
	varyEta = 1;		% FLAG TO DECREASE LEARNING RATE
	verbose = 1;		% FLAG FOR DISPLAYING PROGRESS
	saveEvery = 500; 	% SAVE MODEL EVERY # OF EPOCHS
	displayEvery=17;	% # OF UPDATES TO DISPLAY
	visFun = [];	% SUB IN CUSTOM DISPLAY FUNCTION
	chkConverge = 1;	% FLAG FOR CHECKING CONVERGENCE
	saveFold='./models';% PLACE TO SAVE MODELS

end % END PROPERTIES

methods

	% CONSTRUCTOR
	function self = rbmClassifier(args, data,labels)
		if notDefined('data')
			data = [];
		end
		if ~nargin
			self = self.defaultRBMClassifier;
		else
			self = self.init(args,data,labels);
		end
	end

	function [] = print(self)
		properties(self)
		methods(self)
	end

	% MAIN
	function self = train(self)
		eta0 = self.eta;
		etaFinal = self.etaFinal;
		dCount = 1;
		for iE = 1:self.nEpoch
			sumErr = 0;
			% WEIGHT DECAY
			if self.anneal
				self.wPenalty = self.wDecay - 0.9*iE/self.nEpoch*self.wDecay;
			end

			% LOOP OVER BATCHES
			for jB = 1:numel(self.batchIdx)
				X = self.X(self.batchIdx{jB},:);
				targs = self.targs(self.batchIdx{jB},:);
				self = self.runGibbs(X,targs);
				self = self.updateParams(X,targs);
				sumErr = self.accumErr(X,sumErr);
				
				if self.verbose & ~mod(dCount,self.displayEvery) & iE > 1;
					self.visLearning(iE-1,jB);
				end
				dCount = dCount+1;
			end
			self.e(iE) = single(sumErr);
			self.a(iE) = self.eta;

			if self.verbose && iE > 1
				self.printProgress(sumErr,iE,jB);
			end
			if self.varyEta
				self.eta = eta0 + etaFinal - eta0*iE/self.nEpoch;
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
	end

	function self = runGibbs(self,X,targs)
		nObs = size(X,1);
		% CD[1]
		% UP...
		[self.pHid,self.aHid] = self.VtoH(X,targs);
		self.pHid0 = self.pHid;
		% DOWN...
		[self.pVis,self.aVis,self.pClass,self.aClass] = self.HtoV();
		% BACK UP.
		[self.pHid,self.aHid] = self.VtoH(self.aVis,self.aClass);

		% CD [2...nGibbs]
		for iC = 1:self.nGibbs-1
			[self.pVis,self.aVis,self.pClass,self.aClass] = self.HtoV();
			[self.pHid,self.aHid] = self.VtoH(self.aVis,self.aClass);
		end
	end

	% CONDITIONAL p(h |v, W), AND HIDDEN ACTIVATIONS
	function [pHid,aHid] = VtoH(self,X,targs)
		nObs = size(X,1);
		if notDefined('targs');
			targs = ones(size(X,1),size(self.classW,1));
		end
		switch self.type
		case 'BB'
			pHid = self.sigmoid(bsxfun(@plus,X*self.W+targs*self.classW ,self.c));
			aHid = pHid>rand(nObs,self.nHid);
		case 'GB'
			scaledX = bsxfun(@rdivide,X,self.sigma2);
			pHid = self.sigmoid(bsxfun(@plus,scaledX*self.W+targs*self.classW,self.c));
			aHid = pHid>rand(size(X,1),self.nHid);
		end
	end

	% CONDITIONAL p(v|h, W), AND VISIBLE/CATEGORY ACTIVATIONS
	function [pVis,aVis,pClass,aClass] = HtoV(self,aHid)
		if notDefined('aHid')
			aHid = self.aHid;
		end
		if notDefined('sampleVis')
			sampleVis = self.sampleVis;
		end
		nObs = size(aHid,1);
		switch self.type
		case 'BB'
			pVis = self.sigmoid(bsxfun(@plus,aHid*self.W',self.b));
			aVis = pVis>rand(nObs,self.nVis);
			pClass = self.softMax(bsxfun(@plus,self.aHid*self.classW',self.d));
			aClass = self.sampleClasses(pClass);
		case 'GB'
			mu = bsxfun(@plus,aHid*self.W',self.b);
			pVis = self.drawNormal(mu,self.sigma2);
			if sampleVis
				aVis = pVis;
			else
				aVis = mu;
			end
			pClass = self.softMax(bsxfun(@plus,self.aHid*self.classW',self.d));
			aClass = self.sampleClasses(pClass);
		end
	end

	% LEARNING RULES
	function self = updateParams(self,X,targs);
		nObs = size(targs,1);
		switch self.type
		case 'BB'
			% CONNECTION WEIGHTS
			dW=(X'*self.pHid0 - self.aVis'*self.pHid)/nObs;
			self.dW=self.momentum*self.dW + self.eta*(dW - self.wPenalty*self.W);
			self.W = self.W + self.dW;

			% VISIBLE BIASES
			db = (sum(X) - sum(self.aVis))/nObs;
			self.db = self.momentum*self.db + self.eta*db;
			self.b = self.b + self.db;

			% HIDDEN BIASES
			dc = (sum(self.pHid0) - sum(self.pHid))/nObs;
			self.dc = self.momentum*self.dc + self.eta*dc;
			self.c = self.c + self.dc;

			% CLASSIFIER WEIGHTS
			dClassW=(targs'*self.pHid0 - self.aClass'*self.pHid)/nObs;
			self.dClassW=self.momentum*self.dClassW+self.eta*(dClassW-self.wPenalty*self.classW);
			self.classW = self.classW + self.dClassW;

			% CLASSIFIER BIASES
			dd = (sum(targs) - sum(self.aClass))/nObs;
			self.dd = self.momentum*self.dd + self.eta*dd;
			self.d = self.d + self.dd;
		case 'GB'
			sigma2 = repmat(self.sigma2,nObs,1);
			% CONNECTION WEIGHTS
			dW=bsxfun(@rdivide,(X'*self.pHid0 - self.aVis'*self.pHid),(nObs*self.sigma2)');% EQ 5.11
			self.dW=self.momentum*self.dW + self.eta*dW - self.wPenalty*self.W;
			self.W = self.W + self.dW;

			% VISIBLE BIASES
			db = mean(X./sigma2) - mean(self.aVis./sigma2); % EQ 5.12
			self.db = self.momentum*self.db + self.eta*db;
			self.b = self.b + self.db;

			% HIDDEN BIASES
			dc = mean(self.pHid0) - mean(self.pHid); % EQ 5.13
			self.dc = self.momentum*self.dc + self.eta*dc;
			self.c = self.c + self.dc;

			% CLASSIFIER WEIGHTS
			dClassW=(targs'*self.pHid0 - self.aClass'*self.pHid)/nObs;
			self.dClassW=self.momentum*self.dClassW+self.eta*(dClassW-self.wPenalty*self.classW);
			self.classW = self.classW + self.dClassW;

			% CLASSIFIER BIASES
			dd = (sum(targs) - sum(self.aClass))/nObs;
			self.dd = self.momentum*self.dd + self.eta*dd;
			self.d = self.d + self.dd;
			if self.learnSigma2
				% LOG VARIANCES
				tmp = bsxfun(@minus,X,self.b).^2;
				EzDat = bsxfun(@rdivide,mean(tmp-(X.*(self.pHid0*self.W'))),self.sigma2);

				tmp = bsxfun(@minus,self.aVis,self.b).^2;
				EzMod = bsxfun(@rdivide,mean(tmp-(self.aVis.*(self.pHid*self.W'))),self.sigma2);

				dz = bsxfun(@times,(EzDat-EzMod),exp(-self.z)); % EQ 5.14(c)
				self.dz = self.momentum*self.dz + self.eta*dz -self.wPenalty*self.z;
				self.z = self.z + self.dz;
				self.z(isnan(self.z)) = 1e-6;
				self.z(find(self.z>=0)) = 1e-6;
				self.sigma2 = exp(self.z) + 1e-8;
			end
		end
	end

	function [pred,predError,misClass] = predict(self,testData,testLabels)
		nClass = size(self.classW,1);
		nObs = size(testData,1);
		freeEnergy = zeros(nObs,nClass);
		for iC = 1:nClass
			X = zeros(size(freeEnergy));
			X(:,iC) = 1;
			freeEnergy(:,iC) = repmat(self.d(iC),nObs,1).*X(:,iC) + ...
			                       sum(log(exp(testData*self.W + ...
			                       bsxfun(@plus,X*self.classW,self.c))+1),2);
		end
		[foo,predIdx]=max(freeEnergy,[],2);
		pred = zeros(size(predIdx));
		
		for iP = 1:numel(pred)
			pred(iP) = self.labels(predIdx(iP));
		end
		
		if ~notDefined('testLabels')
			misClass = pred~=testLabels;
			predError = sum(misClass/numel(pred));
			misClass = find(misClass);
		else
			misClass = 'unknown';
			predError = misClass;
		end
	end

	function err = accumErr(self,X,err0);
		err = sum(sum((X-self.pVis).^2));
		err = err + err0;
	end

	function self = init(self,args,data,labels)
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

		% INITIALIZE MODEL PARAMS FROM DATA AND LABELS
		if notDefined('data')
			defaultDataFile = 'defaultData.mat';
			load(defaultDataFile,'data');
			if self.verbose
				fprintf('\nNote: using default dataset in: \n-->  || %s\n',defaultDataFile);
			end
		end
		if notDefined('labels')
			defaultDataFile = 'defaultData.mat';
			load(defaultDataFile,'labels');
			if self.verbose
				fprintf('\nNote: using default labels in: \n--> %s\n',defaultDataFile);
			end
		end
		self.e = zeros(1,self.nEpoch);
		[self.nObs,self.nVis]=size(data);
		self.nClasses=max(labels);
		self.labels = unique(labels);
		switch self.type
		case {'GB','BG'}
			self.W = (2/(self.nHid+self.nVis))*rand(self.nVis,self.nHid) - ...
			1/(self.nVis + self.nHid);
		case 'BB'
			self.W = 0.1*rand(self.nVis,self.nHid);
		end
		self.dW = zeros(size(self.W));
		self.classW = 0.1*randn(self.nClasses,self.nHid);		
		self.dClassW = zeros(size(self.classW));
		self.b = zeros(1,self.nVis);
		self.db = self.b;
		self.c = zeros(1,self.nHid);
		self.dc = self.c;
		self.d = zeros(1,self.nClasses);
		self.dd = self.d;
		if strcmp(self.type,'GB')
			self.sigma2 = ones(1,self.nVis);
			self.z = log(self.sigma2);
			self.dz = zeros(size(self.z));
		end
		self.X = data;
		self.Y = labels;
		self.batchIdx = self.createBatches;
		self.targs = self.createTargets(labels);
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

	% 1 OF K REPRESENTATION FOR CLASS LABELS
	function targs = createTargets(self,labels)
		classes = unique(labels);
		targs = zeros(numel(labels),max(classes));
		for iL = 1:numel(classes)
			targs(labels==classes(iL),iL)=1;
		end
	end

	function rbm = defaultRBMClassifier(self)
		% EDIT THIS m-FILE FOR DEFAULT ARGUMENTS
		args = defaultRBMClassifier();
		rbm = self.init(args);
		fprintf('\nNote: using default RBM (see defaultRBMClass.m)\n\n');
	end

	function y = sigmoid(self,X)
		y = 1./(1 + exp(-X));
	end

	function c = softMax(self,X)
		c = bsxfun(@rdivide,exp(X),sum(exp(X),2));
	end

	function classes = sampleClasses(self,pClass)
		[nObs,nClass]=size(pClass);
		classes = zeros(size(pClass));
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

	function samps = sample(self,data,nSamps,nIters)
		if notDefined('nIters')
			nIters = 1;
		end
		if notDefined('nSamps')
			nSamps = 1;
		end
		for iS = 1:nSamps
			for iI = 1:nIters
				aHid = self.VtoH(data);
				data = self.HtoV(aHid);
			end
			samps(:,:,iS) = data;
		end
	end

	function p = drawNormal(self,mu,sigma2);
		S = eye(numel(sigma2));
		S(find(S)) = sigma2;
		p = mvnrnd(mu,S);
	end
	
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

	function [] = printProgress(self,sumErr,iE,jB)
		if iE > 1
			if self.e(iE) > self.e(iE-1) & iE > 1
				indStr = '(UP)  ';
			else
				indStr = '(DOWN)';
			end
		else
			indStr = '';
		end
		fprintf('Epoch %d/%d --> Recon. error: %f %s\r', ...
		iE,self.nEpoch,sumErr,indStr);
	end

	function data= recon(self,data,nIters)
		if notDefined('nIters')
			nIters = 1;
		end
		for iI = 1:nIters
			aHid = self.VtoH(data);
						data = self
						subplot(3,3,7);
						semilogy(RBM.a(1:iE));
						title('Learning Rate');
		end
	end

	function success = converged(self,errors);
		success = (abs(mean(gradient(smooth(errors)))/max(self.e)) <= 0.0001);
		if success
			fprintf('\nCONVERGED.\n');
		end
	end

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
			im = visWeights(visData',0,lims,1);
		end
	end

end % END METHODS

end % END CLASSDEF