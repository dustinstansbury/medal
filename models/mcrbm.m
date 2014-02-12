classdef mcrbm
% Mean-Covariance Restricted Boltzmann Machine Model object:
%----------------------------------------------------------------------------
% Initialize and train a mean-covariance RBM as desribed in:
%
% Ranzato et al (2010). "Modeling Pixel Means and Covariances Using Factorized
% Third-Order Boltzmann Machines".
%
% This function is based on the pycuda implementation available online at
% http://www.cs.toronto.edu/~ranzato/publications/mcRBM/code/mcRBM_04May2010.zip
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
	bW					% MEANS BIAS - [nHidMean x 1]
	bV					% VISIBLE BIAS - [nVis x 1]
	
	% GRADIENTS
	dC					% COVARIANCE GRADIENT
	dP					% FACTORS GRADIENT
	dW					% MEANS GRADIENT
	dbC					% COV. BIAS GRADIENT
	dbV					% VISIBLE BIAS GRADIENT
	dbW					% MEAN BIAS GRADIENT

	%% LEARNING PARAMS
	wPenalty = 0.001;	% WEIGHT DECAY
	lRate0 = 0.01;		% BASE LEARNING RATE (LR)
	lRateC				% COVARIANCE LR
	lRateP				% FACTORS LR
	lRateW				% MEANS LR
	lRateb				% GENERAL BIAS LR
	lRatebW				% MEAN BIAS LR

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

	epoch = 1;
	verbose = 1;
	displayEvery = 10;
	trainTime;
	visFun;
	saveEvery = 500;
	saveFold;
	auxVars = struct();
	useGPU = 1;
	gpuDevice;			
	log					% LOGGING FIELD
end % END PROPERTIES

methods

	function self = mcrbm(arch)
	% m = mcrbm(arch)
	%--------------------------------------------------------------------------
	% mcRBM constructor
	%--------------------------------------------------------------------------
	% INPUT:
	%  <arch>:  - a set of arguments defining the RBM architecture.
	%
	% OUTPUT:
	%     <m>:  - an mcRBM model object.
	%--------------------------------------------------------------------------
		self = self.init(arch);
	end
	function print(self)
	%print()
	%--------------------------------------------------------------------------
	% Print mcrbm attributes
	%--------------------------------------------------------------------------
		properties(self)
		methods(self)
	end
	
	function self = train(self,data)
	% m = train(data)
	%--------------------------------------------------------------------------
	% Train an mcRBM using Contrastive Divergence
	%--------------------------------------------------------------------------
	% INPUT:
	%  <data>:  - to-be modeled data |X| = [#Obs x #Vis]
	% OUTPUT:
	%     <m>:  - trained RBM object.
	%--------------------------------------------------------------------------

		% INITIAL LEARNING RATES
		lRateC0 = 2*self.lRate0;
		lRateP0 = .02*self.lRate0;
		lRateb0 = .02*self.lRate0;
		lRateW0 = .2*self.lRate0;
		lRatebW0 = .1*self.lRate0;
		
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

		self.lRateC = lRateC0;
		self.lRateP = lRateP0;
		self.lRateb = lRateb0;
		self.lRateW = lRateW0;
		self.lRatebW =lRatebW0;

		if self.useGPU
			% ENSURE GPU HAS ENOUGH MEMORY TO TRAIN
			try
				self = gpuDistribute(self);
				t = gpuDistribute(t);
				data = gpuArray(data);
			catch
			end
		end
		
		
		nObs = size(data,2);
		self.batchIdx = self.createBatches(data);
		tic
		t.clock = 0;
		dCount = 1;

		while 1
			sumErr = 0;
			% SIMULATED ANNEALING
			if self.epoch >= self.beginAnneal

				self.lRateC = max(lRateC0*((self.epoch-self.beginAnneal+1)^(-.25)),1e-6);
				self.lRateP = max(lRateP0*((self.epoch-self.beginAnneal+1)^(-.25)),1e-6);
				self.lRateb = max(lRateb0*((self.epoch-self.beginAnneal+1)^(-.25)),1e-6);
				self.lRateW = max(lRateW0*((self.epoch-self.beginAnneal+1)^(-.25)),1e-6);
				self.lRatebW = max(lRatebW0*((self.epoch-self.beginAnneal+1)^(-.25)),1e-6);
			end
			
			if self.epoch >= self.beginWeightDecay,
				self.wPenalty = wPenalty;
			else
				self.wPenalty = 0;
			end
			
			% LOOP OVER BATCHES
			for jB = 1:numel(self.batchIdx)
				t.data = data(:,self.batchIdx{jB});

				% POSITIVE GRADIENTS
				[self,t] = self.paramGradientsUp(t);   % (1)

				% PERSISTENT CONTRASTIVE DIVERGENCE?
				if self.pcd, dataField = 'negData';
				else, dataField = 'data';
				end

				% APPROXIMATE SAMPLES FROM THE MODEL
				[self,t] = self.sampleHMC(t,dataField);  % (2)
				
				% NEGATIVE GRADIENTS
				[self,t] = self.paramGradientsDown(t);   % (3)

				% UPDATE C AND P (REGULARIZE)
				[self,t] = self.updateFactors(t);     % (4) Eq. 10
				
				% BATCH ERROR
				sumErr = self.accumErr(t,sumErr);

				% DISPLAY
				if  ~mod(dCount,self.displayEvery);
					try
						self.auxVars.error = self.log.err(1:iE-1);
						self.auxVars.batchX = t.data;
						t.iE = iE;
						self.auxVars.t = t;
						self.visFun(self)
					end
				end
				dCount = dCount+1;
			end
			
			% NORMALIZE P
			self = self.normalizeP();

			% UPDATE MEAN PARAMETERS
			if self.modelMeans
				self = self.updateMeans(size(t.data,2)); % (4) Eq.
			end

			% LOG TOTAL ERROR
			self.log.err(self.epoch) = single(gather(sumErr))/nObs;

			if self.verbose,
				self.printProgress(sumErr,t);
				t.clock = toc;
			end

			% SAVE CURRENT PROGRESS?
			if ~mod(self.epoch,self.saveEvery)&~isempty(self.saveFold)
				r = self;
				if ~exist(r.saveFold,'dir'),mkdir(r.saveFold);end
				save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',self.epoch)),'r'); clear r;
			end
			if self.epoch >= self.nEpoch, break; end
			self.epoch = self.epoch + 1;
		end
		
		% CLEAN UP
		if self.useGPU
			self = gpuGather(self);
			reset(self.gpuDevice);
		end
		fprintf('\n');
		self.trainTime = toc;
	end

	function [self,t] = paramGradientsUp(self,t)
	%[m,t] = paramGradientsUp(t)
	%--------------------------------------------------------------------------
	% Calcualate positive phase portion of likelihood gradients. t is a struct of 
	% temporary storage variables.
	%--------------------------------------------------------------------------
		% NORMALIZE INPUT
		t.normData = self.normalizeData(t.data);

		t.filtOut = self.C'*t.normData;  % [#Factors x #Samps]
		t.filtOutSq = t.filtOut.^2;      % [#Factors x #Samps]
		
		% p(h | v), Eq (2)
		pHid = self.sigmoid(bsxfun(@plus,(-.5*self.P'*t.filtOutSq),self.bC)); %[#Cov x #Samps]
		
		% FACTORS GRADIENTS
		self.dP = (t.filtOutSq*pHid'); % [#Fact x #Cov]

		% COVARIANCE GRADIENTS
		self.dC = t.normData*((self.P*pHid).*t.filtOut)'; % [#Vis x #Factors]

		% BIAS GRADIENTS
		self.dbC = -sum(pHid,2);       %  [#Cov x 1]
		self.dbV = -sum(t.data,2);     %  [#Vis x 1]

		if self.modelMeans
			% MEANS GRADIENTS
			t.meanFiltOut = self.sigmoid(bsxfun(@plus,self.W'*t.data,self.bW)); %[#Mean x #Samps]
			self.dW = -t.data*t.meanFiltOut'; %[nVis x nMean]
			self.dbW = -sum(t.meanFiltOut,2);% MEAN BIAS  [nMean x 1]
		end
	end

	function [self,t] = paramGradientsDown(self,t)
	%[m,t] = paramGradientsUp(t)
	%--------------------------------------------------------------------------
	% Calcualate negative phase of of likelihood gradients. t is a struct of 
	% temporary storage variables.
	%--------------------------------------------------------------------------		
	% FINALIZE PARAMETER GRADIENTS AFTER SAMPLES

		% NORMALIZE NEGATIVE DATA
		t.normData = self.normalizeData(t.negData);

		% FILTER OUTPUTS
		t.filtOut = self.C'*t.normData;  % [#Factors x #Samps]
		t.filtOutSq = t.filtOut.^2;      % [#Factors x #Samps]
		
		% p(h^(c) | v), Eq (2)
		pHid = self.sigmoid(bsxfun(@plus,(-.5*self.P'*t.filtOutSq),self.bC)); %[#Cov

		% FACTORS GRADIENTS
		self.dP = 0.5*(self.dP - (t.filtOutSq*pHid')); % [#Fact x #Cov]

		% COVARIANCE GRADIENTS
		self.dC = self.dC - t.normData*((self.P*pHid).*t.filtOut)'; % [#Vis x #Factors]
		

		% BIAS GRADIENTS
		self.dbC = self.dbC + sum(pHid,2); % COVARIANCE BIAS
		self.dbV = self.dbV + sum(t.negData,2); % VISIBLE BIAS

		if self.modelMeans
			% MEANS GRADIENTS
			% Eq p(h^(m) | v) (5)
			t.meanFiltOut = self.sigmoid(bsxfun(@plus,self.W'*t.negData,self.bW));
			self.dW = self.dW + t.negData*t.meanFiltOut';
			self.dbW = self.dbW + sum(t.meanFiltOut, 2);% MEAN BIAS
		end
	end

	function t = energyGradient(self,t,dataField)
	%t = energyGradient(t,dataField)
	%--------------------------------------------------------------------------		
	% Calculate derivative of mcRBM free energy  at a datapoint t.(dataField).
	% <t> is a struct of tmpororary storage variables. <dataField> is a pointer
	% to the current data used.
	%--------------------------------------------------------------------------		
		% NORMALIZE DATA
		[t.normData,t.vectLenSq] = self.normalizeData(t.(dataField));
		t.vectLenSq = t.vectLenSq/self.nVis + .5;

		% LINEAR AND SQUARED FILTER OUTPUTS
		t.filtOut = self.C'*t.normData; 	% [nFactors x batchSz]
		t.filtOutSq = t.filtOut.*t.filtOut;% [nFactors x batchSz]

		% NORMALIZED GRADIENT
		t.dNorm = self.C*((self.P*self.sigmoid(bsxfun(@plus,-.5*self.P'*t.filtOutSq,self.bC))).*t.filtOut);

		% BACKPROP ENERGY DERIVATIVE THROUGH NORMALIZATION - [nVis x batchSz]
		t.dF = bsxfun(@times,t.(dataField),-sum(t.dNorm.*t.(dataField))/self.nVis);
		t.dF = t.dF +  bsxfun(@times,t.dNorm,t.vectLenSq);
		t.dF = bsxfun(@rdivide,t.dF,sqrt(t.vectLenSq).*t.vectLenSq);

		% ADD QUADRATIC TERM
		t.dF = t.dF + t.(dataField);

		% ADD VISIBILE BIAS CONTRIBUTION
		t.dF = bsxfun(@minus,t.dF,self.bV);

		if self.modelMeans
			% ADD MEAN CONTRIBUTION
			t.meanFiltOut = self.sigmoid(bsxfun(@plus,self.W'*t.(dataField),self.bW));
			t.dF = t.dF - self.W*t.meanFiltOut;
        end
	end

	function t = freeEnergy(self,t,dataField,energyField)
		%t = freeEnergy(t,dataField,energyField)
		%--------------------------------------------------------------------------		
		% Calculate the free energy for the mcRBM at data indicated by t.<dataFields>
		% and store it in t.<entergyField>
		%--------------------------------------------------------------------------		

	  	% NORMALIZE DATA
		[t.normData,t.vectLenSq] = self.normalizeData(t.(dataField));

		%% COMPOSE HAMILTONIAN 
		
		% REGULARIZATION TERM CONTRIBUTION
		t.U = .5*t.vectLenSq;

		% POTENTIAL ENERGIES
		% COVARIANCE CONTRIBUTION
		t.filtOut = self.C'*t.normData;
		t.filtOutSq = t.filtOut.^2;

		% Eqn (8), 1ST TERM
		t.U = t.U - sum(log(1+exp(bsxfun(@plus,-.5*self.P'*t.filtOutSq,self.bC))));

		if self.modelMeans
			% MEAN CONTRIBUTION
			% Eqn (8) 2ND TERM
			t.U = t.U - sum(log(1+exp(bsxfun(@plus,self.W'*t.data,self.bW))));
		end

		% VISIBLE BIAS
		t.U = t.U - sum(bsxfun(@times,t.data,self.bV));

		%% ADD KINETIC ENERGY TERM
		t.(energyField) = t.U + .5*sum(t.p.*t.p);
	end


	function [self,t] = sampleHMC(self,t,dataField)
	%[m,t] = sampleHMC(t,dataField)
	%--------------------------------------------------------------------------		
	% Draw a sample from the mcRBM using hybrid monte-carlo sampler on free 
	% energy.
	%--------------------------------------------------------------------------		
		% SAMPLE MOMENTA
		t.p = randn(size(t.p));

		% INITIAL ENERGY (F0) AT POSITIVE DATA
		t = self.freeEnergy(t,dataField,'F0');
		
		% CALC INITIAL GRADIENT
		t = self.energyGradient(t,dataField);

		% BEGIN LEAPFROG ALGORITHM
		%----------------------------------------------
		% FIRST HALF STEP
		t.p = t.p - .5*t.dF.*self.hmcStep;
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
		% SYMMETRIC PROPOSAL, BUT SINCE USING CD[1],
		% IT DOESN'T MATTER; ALSO END UP SQUARING p
		% LATER, SO MEH
		% t.p = -t.p;
		%----------------------------------------------
		% END LEAPFROG ALGORITHM

			% FINAL ENERGY (F) AT NEGATIVE DATA 
		t = self.freeEnergy(t,'negData','F');

		% EVALUATE ACCEPT/REJECTION CRITERION
		t.thresh = exp(t.F0 - t.F);

		% DETECT REJECTED SAMPLES
		t.detect = 1*(t.thresh < rand(size(t.thresh)));
		t.nReject = sum(t.detect,2);

		% UPDATE REJECTION RATE
		% NOTE: HERE WE ASSUME ALL BATCHES ARE SAME SIZE; PERHAPS FIX
		t.rejectRate = t.nReject/size(t.data,2);  
		t.hmcAverageRej = 0.9*t.hmcAverageRej + 0.1*t.rejectRate;

		% UPDATE NEGATIVE DATA ACCORDING TO ACCEPT/REJECT
		t.maskData = bsxfun(@times,t.data,t.detect);
		t.maskNegData = bsxfun(@times,t.negData,t.detect);

		t.negData = t.negData - t.maskNegData + t.maskData;

		% UPDATE STEPSIZE
		if t.hmcAverageRej < self.hmcTargetRej
			self.hmcStep = min(0.25,self.hmcStep*1.01);
		else
			self.hmcStep = max(1e-3,self.hmcStep*0.99);
		end
	end
	
	function [self,t] = updateFactors(self,t)
	%[m,t] = updateFactors(t)	
	%--------------------------------------------------------------------------		
	% L1-regularize and update factor parameters, C and P
	%--------------------------------------------------------------------------		
		%  COVARIANCE HIDDENS, C
		self.dC = self.dC + sign(self.C)*self.wPenalty;
		self.C = self.C - self.dC*self.lRateC/size(t.data,2);

		% NORMALIZE C
		[self,t] = self.normalizeC(t);

		% COVARIANCE AND VISIBLE BIASES
		self.bC = self.bC - self.dbC*self.lRateb/size(t.data,2);
		self.bV = self.bV - self.dbV*self.lRateb/size(t.data,2);

		% P
		if self.epoch >= self.beginPUpdates
			self.dP = self.dP + sign(self.P)*self.wPenalty;
			self.P = self.P - self.dP*self.lRateP/size(t.data,2);
		end

		% (ANTI) RECTIFY P
		self.P(self.P < 0) = 0;

		% ENFORCE TOPOGRAPHY
		if ~isempty(self.topoMask)
			self.P = self.P.*self.topoMask;
		end
	end

	function self = updateMeans(self,nVis)
	%--------------------------------------------------------------------------		
	% L1-regularize mean weights/biases and update.
	%--------------------------------------------------------------------------		
		self.dW = self.dW + sign(self.W)*self.wPenalty;
		self.W = self.W - self.dW*self.lRateW/nVis;
		self.bW = self.bW - self.dbW*self.lRatebW/nVis;
	end
	
	function arch = ensureArchitecture(self,arch);
	% arch = ensureArchitecture(self,arch)
	%--------------------------------------------------------------------------
	% Preprocess the provided architecture structure <arch>.
	%--------------------------------------------------------------------------
	% PARSE SUPPLIED ARCHITECTURE
	% <arch> is either a [1 x 4] vector that is shorthand for:
	%       [#Vis #hidMean #hidCov #Factors], in which case we use
    %       default parameters, or a struct, with the fields
    %       .nVis, .nHidMean, .nHidCov, .nFactors, and, optionally, a
    %       field .opts, which is a cell array of field-value global model
    %       options
	%--------------------------------------------------------------------------
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
	% m = init(arch)
	%--------------------------------------------------------------------------
	% Initialize an mcrbm object based on provided architecture, <arch>.
	%--------------------------------------------------------------------------
	% INPUT:
	%    <arch>:  - a structure of architecture options. Possible fields are:
	%               .nVis     --> #Visible units
	%               .nHidMean --> # of mean hidden units
	%               .nHidCov  --> # of covariance hidden units
	%               .nPool    --> # of pooling units
	%               .opts     --> additional cell array of options, defined
	%                             in argument-value pairs.
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
		
		if ~isempty(self.topoMask);
				self.P = self.topoMask;
				self.topoMask = (self.topoMask > 1e-6);
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
		self.bW = -2*ones(self.nHidMean,1);
		
		% GRADIENTS
		self.dC = zeros(self.nVis,self.nFactors);
		self.dP = zeros(self.nFactors, self.nHidCov);
		self.dbC = zeros(self.nHidCov,1);
		self.dbV = zeros(self.nVis,1);
		self.dbW = zeros(self.nHidMean,1);
	end

	function save(self) %%
	 % save() 
	 %--------------------------------------------------------------------------
	 % Save current network
	 %--------------------------------------------------------------------------
			if self.useGPU
				r = gpuGather(self);
			else
				r = self;
			end
			if ~exist(r.saveFold,'dir')
				mkdir(r.saveFold);
			end
			save(fullfile(r.saveFold,sprintf('Epoch_%d.mat',self.epoch)),'r'); clear r;
	end


	% ACCUMULATE ERROR
	function err = accumErr(self,t,err0);
		err = nansum(sum((t.data-t.negData).^2));
		err = err + err0;
	end

	function printProgress(self,sumErr,t)
		iE = self.epoch;
		fprintf('\n---- Epoch %d / %d ----\n',iE,self.nEpoch)
		fprintf('\n|C|   = %3.2e',norm(self.C)) ;
		fprintf('\n|dC|  = %3.2e',norm(self.dC)*(self.lRateC/size(t.data,2))) ;

		if self.modelMeans
			fprintf('\n|W|   = %3.2e',norm(self.W)) ;
			fprintf('\n|dW|  = %3.2e',norm(self.dW)*(self.lRateW/size(t.data,2)));
		end
		fprintf('\n|P|   = %3.2e',norm(self.P)) ;
		fprintf('\n|dP|  = %3.2e',norm(self.dP)*(self.lRateP/size(t.data,2))) ;
		
		fprintf('\n|bC|  = %3.2e',norm(self.bC)) ;
		fprintf('\n|dbC| = %3.2e',norm(self.dbC)*(self.lRateb/size(t.data,2)));
		
		fprintf('\n|bW|  = %3.2e',norm(self.bW)) ;
		fprintf('\n|dbW| = %3.2e',norm(self.dbW)*(self.lRatebW/size(t.data,2)));
		
		fprintf('\n|bV|  = %3.2e',norm(self.bV)) ;
		fprintf('\n|dbV| = %3.2e',norm(self.dbV)*(self.lRateb/size(t.data,2)));
		
		fprintf('\n\nHMC step = %3.2e',self.hmcStep);
		fprintf('\nHMC rej rate = %3.2e (target = %3.2e)\n',t.hmcAverageRej,self.hmcTargetRej);
		
		fprintf('\nEpoch duration = %3.2f s\n',toc - t.clock);
	end

	function batchIdx = createBatches(self,X)
	% batchIdx = createBatches(X)
	%--------------------------------------------------------------------------
	% Create minibatches
	%--------------------------------------------------------------------------
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
	% [data,vectLenSq] = normalizeData(self,data)
	%--------------------------------------------------------------------------
	% L2-normalize <data>.
	%--------------------------------------------------------------------------
		vectLenSq = dot(data,data);
		% HERE WE SCALE THE VECTOR LENGTH BY # OF VISIBLES (LIKE STD),
		% AND ADD SMALL OFFSET (0.5) TO AVOID DIVISION BY ZERO
		data = bsxfun(@rdivide,data,sqrt(vectLenSq/self.nVis + .5));
	end

	function [self,t] = normalizeC(self,t)
	%[m,t] = normalizeC(t)
	%--------------------------------------------------------------------------
	% Normalize columns of C by smoothed  calculation (exponential running `
	% average of) of its L2-norm.
	%--------------------------------------------------------------------------
	%  		vectNorm = sqrt(dot(self.C,self.C));
		vectNorm = sqrt(sum(self.C.*self.C));
		t.normC = 0.95*t.normC+(.05/self.nFactors)*sum(vectNorm);
		self.C = bsxfun(@rdivide,self.C,vectNorm)*t.normC;
	end

	function [self] = normalizeP(self)
	%[m] = normalizeP(m)
	%--------------------------------------------------------------------------
	% Rescale P such that L1-norm of each column of p equal to one
	%--------------------------------------------------------------------------
		self.P = bsxfun(@rdivide,self.P,sum(self.P,1));
	end

	function p = sigmoid(self,X)
	%p = sigmoid(X)
	%--------------------------------------------------------------------------
	% Sigmoid activation function
	%--------------------------------------------------------------------------
		p = 1./(1+exp(-X));
	end

end % END METHODS
end % END CLASSDEF