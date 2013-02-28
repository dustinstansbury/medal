classdef mcrbm
% Mean-Covariance Restricted Boltzmann Machine Model object:
%----------------------------------------------------------------------------
% Initialize and train a mean-covariance RBM as desribed in:
%
% Ranzato et al (2010). "Modeling Pixel Means and Covariances Using Factorized
% Third-Order Boltzmann Machines".
%
% This function is based on the pycuda implementation available online at
% ###
%----------------------------------------------------------------------------
% DES
% stan_s_bury@berkeley.edu
properties
	class = 'mcrbm'

	% DATA PARAMS	
	X 					% TRAINING DATA
	batchIdx			% MINBATCH INDICES
	
	%% MODEL PARAMS
	modelMeans=1;		% IS THIS JUST A 3-WAY FACTORED RBM (modelMeans=0)?
	nVis				% # VISIBLES
	nHidMean			% # OF MEAN HIDDENS
	nHidCov				% # OF COVARIANCE HIDDENS
	nFactors			% # OF FACTORS
	

	% CONNECTION WEIGHTS
	C					% VISIBLE TO FACTOR WEIGHTS - [nVis x nFactors]
	P					% FACTOR TO HIDDEN WEIGHTS - [nFactors x nHidCov]
	W 					% MEAN UNIT WEIGHTS - [nvis x nHidMean]

	% BIASES
	bC					% COVARIANCE BIAS - [nHidCov x 1]
	bM					% MEANS BIAS - [nHidMean x 1]
	bV					% VISIBLE BIAS - [nVis x 1]
	
	% GRADIENTS
	dC					% COVARIANCE GRADIENT
	dP					% FACTORS GRADIENT
	dW					% MEANS GRADIENT
	dbC					% COV. BIAS GRADIENT
	dbV					% VISIBLE BIAS GRADIENT
	dbM					% MEAN BIAS GRADIENT

	%% LEARNING PARAMS
	wPenalty = 0.001;	% WEIGHT DECAY
	lRate0 = 0.01;		% BASE LEARNING RATE (LR)
	lRateC				% COVARIANCE LR
	lRateP				% FACTORS LR
	lRateW				% MEANS LR
	lRateb				% GENERAL BIAS LR
	lRatebM				% MEAN BIAS LR

	%% HMC PARAMS
	nLeapFrog = 20;		% # OF LEAP FROG STEPS
	hmcTargetRej = 0.1;% TARGET REJECTION RATE 
	hmcStep = 0.01;		% HMC STEP SIZE

	%% GENERAL PARAMS
	nEpoch = 20;		% # OF TIMES SEE TRAINING DATA
	batchSz = 128;		% SIZE OF MINIBATCHES
	topoMask			% MASK FOR TOPOLOGY
	
	beginAnneal = 1;	% BEGIN ANNEALING AFTER # EPOCH
	beginWeightDecay = 8; % BEGIN WD AFTER # EPOCH
	beginPUpdates = 10;% BEGUP UPDATING FACTORS AFTER # EPOCH
	pcd = 1;			% USE PERSISTENT CONTRASTIVE DIVERGENCE

	verbose = 1;
	displayEvery = 10;
	trainTime;
	visFun;
	saveEvery = 500;
	saveFold='./mcRBMSave'
	auxVars = struct();
	useGPU = 1;
	gpuDevice;			
	log					% LOGGING FIELD
end % END PROPERTIES

methods

	function self = mcrbm(arch)
	% self = mcrbm(arch)
		self = self.init(arch);
	end
	function print(self)
	% PRINT ATTRIBUTES
		properties(self)
		methods(self)
	end

	function self = train(self,data)
	%% MAIN TRAINING

		% INITIAL LEARNING RATES
		lRateC0 = 2*self.lRate0;
		lRateb0 = .01*self.lRate0;
		lRateW0 = .2*self.lRate0;
		lRatebM0 = .02*self.lRate0;
		self.lRateP = .02*self.lRate0;
		
		wPenalty = self.wPenalty;
		
		% INITIALIZE SOME TEMPORARY VARIABLES
		t.data = zeros(self.nVis,self.batchSz);
		t.normData = t.data;		
		t.filtOut= zeros(self.nFactors,self.batchSz);
		t.filtOutSq = t.filtOut;
		t.negData = randn(size(t.data));	% NEG DATA
		t.negData0 = t.data;		     	% INITIAL NEG. DATA
		t.maskData = t.data;
		t.maskNegData = t.data;
		t.F0 = zeros(1,self.batchSz);   	% INITIAL FREE ENERGY
		t.F = t.F0;					     	% FREE ENERGY
		t.U = t.F;						 	% POTENTIAL ENERGY 
		t.dF = t.data;					 	% ENERGY GRADIENT
		t.dNorm = t.data;
		t.thresh = zeros(1,self.batchSz);
		t.hmcAverageRej = 0;
		t.detect = t.thresh;
		t.meanFiltOut = zeros(self.nHidMean,self.batchSz);
		t.p = randn(self.nVis,self.batchSz);
		t.vectLen = t.thresh;
		t.vectLenSq = t.thresh;
		t.normFactor = t.thresh;
		t.normC = 1;

		if self.useGPU
			% ENSURE GPU HAS ENOUGH MEMORY TO TRAIN
			try
				self = gpuDistribute(self);
				t = gpuDistribute(t);
				data = gpuArray(data);
			catch
			end
		end

		self.batchIdx = self.createBatches(data);
		tic
		t.clock = 0;
		dCount = 1;
		iE = 1;
		while 1
			sumErr = 0;
			% SIMULATED ANNEALING
			if self.beginAnneal
				self.lRateC = max(lRateC0/max(1,iE/self.beginAnneal),.001);
				self.lRateb = lRateb0/max(1,iE/self.beginAnneal);
				self.lRateW = lRateW0/max(1,iE/self.beginAnneal);
				self.lRatebM = lRatebM0/max(1,iE/self.beginAnneal);
			end
			
			if iE <= self.beginWeightDecay,
				self.wPenalty = 0;
			else
				self.wPenalty = wPenalty;
			end
			
			% LOOP OVER BATCHES
			for jB = 1:numel(self.batchIdx)
				t.data = data(:,self.batchIdx{jB});
				
				[self,t] = self.paramGradientsUp(t);   % (1)

				% PERSISTENT CONTRASTIVE DIVERGENCE?
				if self.pcd
					dataField = 'negData';
				else
					dataField = 'data';
				end
				
				[self,t] = self.sampleHMC(t,dataField);  % (2)
				[self,t] = self.paramGradientsDown(t);   % (3)
				[self,t] = self.updateFactors(t,iE);     % (4) Eq. 10
				
				sumErr = self.accumErr(t,sumErr);
				
				if  ~mod(dCount,self.displayEvery);
					try
%  						self.auxVars.error = self.log.err(1:iE-1);
						self.auxVars.t = t;
						self.visFun(self)
					catch
					end
				end
				
				dCount = dCount+1;
			end
			
			% NORMALIZE P AND UPDATE MEAN
			% PARAMETERS
			self = self.normalizeP();

			if self.modelMeans
				self = self.updateMeans(); % (4) Eq.
			end

			% LOG ERRORS
			self.log.err(iE) = single(gather(sumErr));
%  			self.log.lRate(iE) = self.lRate;

			if self.verbose,
				self.printProgress(sumErr,iE,t);
				t.clock = toc;
			end

			% SAVE CURRENT NETWORK
			if ~mod(iE,self.saveEvery)
				r = self;
				if ~exist(r.saveFold,'dir'),mkdir(r.saveFold);end
				save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',iE)),'r'); clear r;
			end
			if iE >= self.nEpoch, break; end
			iE = iE + 1;
		end
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
		end
		fprintf('\n');
		self.trainTime = toc;
	end

	function [self,t] = paramGradientsUp(self,t)
	% CALCUALATE INITIAL PART OF PARAMETER GRADIENTS 

		t.normData = self.normalizeData(t.data);

		% COVARIANCE GRADIENTS
		t.filtOut = t.normData'*self.C;  % [#Samps x #Factors]
		t.filtOutSq = t.filtOut.^2;      % [#Samps x #Factors]
		
		% p(h | v), Eq (2)
		pHid = self.sigmoid(bsxfun(@plus,(-.5*t.filtOutSq*self.P),self.bC')); % [#Samps x #Cov]
		
		self.dC = t.normData*((pHid*self.P').*t.filtOut); % [#Vis x #Fators]

		% FACTORS GRADIENTS
		self.dP = (t.filtOutSq'*pHid); 	% [#Fact x #Cov]

		if self.modelMeans
			% MEANS GRADIENTS
			t.meanFiltOut = -self.sigmoid(bsxfun(@plus,self.W'*t.data,self.bM)); %[#Mean x #Samps]
			self.dW = t.data*t.meanFiltOut'; %[nVis x nMean]
		end
		
		% BIAS GRADIENTS
		self.dbC = -sum(pHid)';         % COVARIANCE BIAS  [#Cov x 1]
		self.dbV = -sum(t.data,2);      % VISIBLE BIAS	   [nVis x 1]
		self.dbM = sum(t.meanFiltOut,2);% MEAN BIAS  [nMean x 1]
	end
	

	function [self,t] = paramGradientsDown(self,t)
	% FINALIZE PARAMETER GRADIENTS AFTER SAMPLES
		
		t.normData = self.normalizeData(t.negData);

		% COVARIANCE GRADIENTS
		t.filtOut = t.normData'*self.C;  % [#Samps x #Factors]
		t.filtOutSq = t.filtOut.^2;      % [#Samps x #Factors]
		
		% p(h | v), Eq (2)
		pHid = self.sigmoid(bsxfun(@plus,(-.5*t.filtOutSq*self.P),self.bC')); % [#Samps x #Cov]
		
		self.dC = self.dC - t.normData*((pHid*self.P').*t.filtOut); % [#Vis x #Fators]

		% FACTORS GRADIENTS
		self.dP = .5*(self.dP - (t.filtOutSq'*pHid)); % P

		if self.modelMeans
			% MEANS GRADIENTS
			t.meanFiltOut = self.sigmoid(bsxfun(@plus,self.W'*t.negData,self.bM));

			self.dW = self.dW + t.negData*t.meanFiltOut';
		end
		% BIAS GRADIENTS
		self.dbC = self.dbC + sum(pHid)'; % COVARIANCE BIAS
		self.dbV = self.dbV + sum(t.negData, 2); % VISIBLE BIAS
		self.dbM = self.dbM + sum(t.meanFiltOut, 2);% MEAN BIAS
		
	end

	function t = energyGradient(self,t,dataField)
	% DERIVATIVE OF FREE ENERGY AT DATAPOINT
		
		% NORMALIZE DATA
		[t.normData,t.vectLenSq] = self.normalizeData(t.(dataField));

		t.filtOut = self.C'*t.normData; 	% [nFactors x batchSz]
		t.filtOutSq = t.filtOut.*t.filtOut;% [nFactors x batchSz]
		
		t.dNorm = self.C*(self.P*self.sigmoid(bsxfun(@plus,-.5*self.P'*t.filtOutSq,self.bC)).*t.filtOut);

		% BACKPROP ENERGY DERIVATIVE THROUGH NORMALIZATION - [nVis x batchSz]
		t.dF = bsxfun(@times,t.(dataField),-sum(t.dNorm.*t.(dataField))/self.nVis);
		t.dF = t.dF + bsxfun(@times,t.dNorm,t.vectLenSq);
		
		% ADD SMALL AMOUNT (0.5) HERE SO THAT GRADIENT DOESN'T BLOW UP
		tmp2 = bsxfun(@rdivide, t.dF, sqrt(t.vectLenSq).*t.vectLenSq+.5);
		
		% ADD QUADRATIC TERM
		t.dF = t.dF + t.(dataField);
		
		% ADD VISIBILE BIAS CONTRIBUTION
		t.dF = bsxfun(@plus,t.dF,-self.bV);

		if self.modelMeans
			% ADD MEAN CONTRIBUTION
			t.meanFiltOut = self.sigmoid(bsxfun(@plus,self.W'* ...
			                    t.(dataField),self.bM));
	        t.dF = t.dF - self.W*t.meanFiltOut;
        end
	end

	function [self,t] = updateFactors(self,t,iE)
	% L1-REGULARIZE AND UPDATE FACTOR PARAMETERS
		
		%  COVARIANCE HIDDENS, C
		self.dC = self.dC + sign(self.C)*self.wPenalty;
		self.C = self.C - self.dC*self.lRateC/self.batchSz;

		% NORMALIZE C
		[self,t] = self.normalizeC(t);
		
		% COVARIANCE AND VISIBLE BIASES
		self.bC = self.bC - self.dbC*self.lRateb/self.batchSz;
		self.bV = self.bV - self.dbV*self.lRateb/self.batchSz;

		% P 
		if iE >= self.beginPUpdates
			if self.beginAnneal
				self.lRateP = max(1e-10,self.lRateP/max(1,(self.beginPUpdates-iE)/self.beginAnneal));
			end
			self.dP = self.dP + sign(self.P)*self.wPenalty;
			self.P = self.P - self.dP*self.lRateP/self.batchSz;
		end
		
		% (ANTI) RECTIFY P
		self.P(self.P < 0) = 0;
		
		% ENFORCE TOPOGRAPHY  
		if ~isempty(self.topoMask)
			self.P = self.P.*self.topoMask;
		end
	end

	function self = updateMeans(self)
	% L1-REGULARIZE MEAN WEIGHTS/BIASES AND UPDATE
		self.dW = self.dW + sign(self.W)*self.wPenalty;
		self.W = self.W - self.dW*self.lRateW/self.batchSz;
		self.bM = self.bM - self.dbM*self.lRatebM/self.batchSz;
	end

	function [self,t] = sampleHMC(self,t,dataField)
	% HYBRID MONTE-CARLO SAMPLER ON FREE ENERGY
	
		% SAMPLE MOMENTA
		t.p = randn(size(t.p));

		% ENERGY AT INITIAL/POSITIVE DATA
		t = self.freeEnergy(t,dataField,'F0');
		
		% CALC INITIAL GRADIENT
		t = self.energyGradient(t,dataField);

		% BEGIN LEAPFROG ALGORITHM
		%----------------------------------------------
		% FIRST HALF STEP
		t.p = t.p - .5*t.dF*self.hmcStep;
		t.negData = t.negData + t.p*self.hmcStep;
		
		% FULL STEPS
		for iS = 1:(self.nLeapFrog-1)
			% GRADIENT AT CURRENT RECONSTRUCTION
			t = self.energyGradient(t,'negData');


			% UPDATE MOMENTUM
			t.p = t.p - t.dF*self.hmcStep;
			% UPDATE SAMPLES/POSITION
			t.negData = t.negData + t.p*self.hmcStep;
			
		end

		% LAST HALF STEP
		t = self.energyGradient(t,'negData');
		t.p = t.p - .5*t.dF*self.hmcStep;

		% WE SHOULD NEGATE MOMENTUM HERE FOR 
		% SYMMETRIC PROPOSAL (NEAL, 1996), BUT
		% SINCE USING CD[1], IT DOESN'T MATTER
		% ALSO END UP SQUARING p LATER, SO MEH
		% t.p = -t.p;
		
		%----------------------------------------------
		% END LEAPFROG ALGORITHM

		% NEGATIVE ENERGY
		t = self.freeEnergy(t,'negData','F');
		
		% EVALUATE ACCEPT/REJECTION CRITERION
		t.thresh = exp(t.F0 - t.F);

		t.detect = 1*(t.thresh >= rand(size(t.thresh)));
		t.nReject = sum(t.detect);

		% UPDATE REJECTION RATE
		t.rejectRate = t.nReject/self.batchSz;
		t.hmcAverageRej = 0.9*t.hmcAverageRej + 0.1*t.rejectRate;

		% UPDATE NEGATIVE DATA ACCORDING TO ACCEPT/REJECT
		t.maskData = bsxfun(@times,t.data,t.detect);
		t.maskNegData = bsxfun(@times,t.negData,t.detect);
		
		t.negData = t.negData - t.maskNegData;
		
		t.negData = t.negData + t.maskData;
		
		% UPDATE STEPSIZE
		if t.hmcAverageRej < self.hmcTargetRej
			self.hmcStep = min(0.25,self.hmcStep*1.01);
		else
			self.hmcStep = max(0.001,self.hmcStep*0.99);
		end
	end

	function t = freeEnergy(self,t,dataField,energyField)
	  	% F = - sum log(1+exp(- .5 P (C data/norm(data))^2 + bias_cov)) +...
	  	%    - sum log(1+exp(w_mean data + bias_mean)) + ...
	  	%     - bias_vis data + 0.5 data^2
	  	
	  	% NORMALIZE DATA
		[t.normData,t.vectLenSq] = self.normalizeData(t.(dataField));

		% REGULARIZATION TERM CONTRIBUTION TO ENERGY
		t.U = .5*t.vectLenSq;

		%% COMPOSE HAMILTONIAN - POTENTIAL ENERGY
		% COVARIANCE CONTRIBUTION
		t.filtOut = self.C'*t.normData;
		t.filtOutSq = t.filtOut;
		t.U = t.U-sum(log(1+exp(bsxfun(@plus,-.5*self.P'*t.filtOutSq,self.bC))));

		if self.modelMeans
			% MEAN CONTRIBUTION
			t.U = t.U -sum(log(1+exp(bsxfun(@plus,self.W'*t.data,self.bM))));
		end
		
		% VISIBLE BIAS 
		t.U = t.U + sum(-bsxfun(@times,t.data,self.bV));

		%% ADD KINETIC ENERGY TERM
		t.(energyField) = t.U + .5*sum(t.p.*t.p);
		
	end

	function arch = ensureArchitecture(self,arch);
	% PARSE SUPPLIED ARCHITECTURE
		% <arch> is either a [1 x 4] vector that is shorthand for:
		%       [#Vis #hidMean #hidCov #Factors], in which case we use
	    %       default parameters, or a struct, with the fields
	    %       .nVis, .nHidMean, .nHidCov, .nFactors, and, optionally, a
	    %       field .opts, which is a cell array of field-value global model
	    %       options
	    
		if ~isstruct(arch) % PARSE SHORTHAND INITIALIZATION
			archTmp = arch;
			arch.nVis = archTmp(1);
			arch.nHidMean = archTmp(2);
			arch.nHidCov = archTmp(3);
			arch.nFactors = archTmp(4);
		end
		if ~isfield(arch,'nVis'),
			error('must provide # of inputs in architecture');
		end
		if ~isfield(arch,'nHidMean'),
			error('must provide # of mean hidden units in architecture');
		end
		if ~isfield(arch,'nHidCov'),
			error('must provide # of covariance hidden units in architecture');
		end
		if ~isfield(arch,'nFactors'),
			error('must provide # of factors in architecture');
		end

	end
	
	function self = init(self,arch) %%
	% INITIALIZE mcRBM
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
			arch = rmfield(arch,'opts'); clear opts
		end

		self.nVis = arch.nVis;
		self.nHidMean = arch.nHidMean;
		self.nHidCov = arch.nHidCov;
		self.nFactors = arch.nFactors;
		
		self.log.err = zeros(1,self.nEpoch);
		self.log.lRate = self.log.err;
		if isempty(self.nHidMean) || (self.nHidMean == 0)
			self.modelMeans = false;
		end
		
		% CONNECTION WEIGHTS
		self.C = 0.02*randn(self.nVis,self.nFactors);
		
		% TODO: ADD TOPOGRAPHY MASKING FOR P/P
		if ~isempty(self.topoMask);
			if exist(self.topoMask,'file')
				% INITIALIZE POOLING MATRIX AND MASK
				load(self.topoMask,'PO','mask');
				self.topoMask = mask; clear mask;
				self.P = P0; clear P0;
			end
		else			
			self.P = eye(self.nFactors,self.nHidCov);
			self.topoMask = [];
		end

		if self.modelMeans
			self.W = 0.05*randn(self.nVis,self.nHidMean);
			self.dW = zeros(self.nVis,self.nHidMean);
		end

		% BIASES		
		self.bC = 2*ones(self.nHidCov,1);
		self.bV = zeros(self.nVis,1);
		self.bM = -2*ones(self.nHidMean,1);
		
		% GRADIENTS
		self.dC = zeros(self.nVis,self.nFactors);
		self.dP = zeros(self.nFactors, self.nHidCov);
		self.dbC = zeros(self.nHidCov,1);
		self.dbV = zeros(self.nVis,1);
		self.dbM = zeros(self.nHidMean,1);
	end

	function save(self,epoch) %%
			if self.useGPU
				r = gpuGather(self);
			else
				r = self;
			end
			
			if ~exist(r.saveFold,'dir')
				mkdir(r.saveFold);
			end
			save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',epoch)),'r'); clear r;
	end


	% ACCUMULATE ERROR
	function err = accumErr(self,t,err0);
		err = nansum(sum((t.data-t.negData).^2));
		err = err + err0;
	end

	function printProgress(self,sumErr,iE,t)
		fprintf('\n---- Epoch %d / %d ----\n',iE,self.nEpoch)
		fprintf('\n|C|   = %3.2e',norm(self.C)) ;
		fprintf('\n|dC|  = %3.2e',norm(self.dC)*(self.lRateC/self.batchSz)) ;

		if self.modelMeans
			fprintf('\n|W|   = %3.2e',norm(self.W)) ;
			fprintf('\n|dW|  = %3.2e',norm(self.dW)*(self.lRateW/self.batchSz));
		end
		fprintf('\n|P|   = %3.2e',norm(self.P)) ;
		fprintf('\n|dP|  = %3.2e',norm(self.dP)*(self.lRateP/self.batchSz)) ;
		
		fprintf('\n|bC|  = %3.2e',norm(self.bC)) ;
		fprintf('\n|dbC| = %3.2e',norm(self.dbC)*(self.lRateb/self.batchSz));
		
		fprintf('\n|bM|  = %3.2e',norm(self.bM)) ;
		fprintf('\n|dbM| = %3.2e',norm(self.dbM)*(self.lRatebM/self.batchSz));
		
		fprintf('\n|bV|  = %3.2e',norm(self.bV)) ;
		fprintf('\n|dbV| = %3.2e',norm(self.dbV)*(self.lRateb/self.batchSz));
		
		fprintf('\n\nHMC step = %3.2e',self.hmcStep);
		fprintf('\nHMC rej rate = %3.2e (target = %3.2e)\n',t.hmcAverageRej,self.hmcTargetRej);
		
		fprintf('\nEpoch duration = %3.2f s\n',toc - t.clock);
	end

	function batchIdx = createBatches(self,X)
	% CREATE MINIBATCHES
		[nVis,nObs] = size(X);
		
		nBatches = ceil(nObs/self.batchSz);
		tmp = repmat(1:nBatches, 1, self.batchSz);
		tmp = tmp(1:nObs);
		randIdx = randperm(nObs);
		tmp = tmp(randIdx);
		for iB=1:nBatches
		    batchIdx{iB} = find(tmp==iB);
		end
	end

	function [data,vectLenSq] = normalizeData(self,data)
	% L2-NORMALIZE DATA
		vectLenSq = dot(data,data);
		vectLen = sqrt(vectLenSq/self.nVis + .5);
		normData = bsxfun(@rdivide,data,vectLen);
	end

	function [self,t] = normalizeC(self,t)
	% NORMALIZE COLUMNS OF C BY SMOOTHED
	% (RUNNING AVERAGE OF) L2-NORM
	
		vectNorm = sqrt(dot(self.C,self.C));
		t.normC = 0.95*t.normC+(.05/self.nFactors)*sum(vectNorm);
		self.C = bsxfun(@rdivide,self.C,vectNorm)*t.normC;
	end

	function [self] = normalizeP(self)
	% SET L1-NORM OF EACH COLUMN OF P EQUAL TO ONE
		self.P = bsxfun(@rdivide,self.P,sum(self.P));
	end

	function p = sigmoid(self,X)
		p = 1./(1+exp(-X));
	end

end % END METHODS
end % END CLASSDEF