classdef dae
%-----------------------------------------------------------
% Deep Autoencoder model class
%-----------------------------------------------------------
% DES

properties
	layers = [];		% FIT MODELS FOR EACH LAYER
	X = [];				% INPUT DATA
	mapX = []			% INPUT DATA MAPPED TO SUBSPACE 
	nLayers = [];		% # OF COMPRISING RBMS
	nFineTune = 20;		% MAXIMUM NUMBER OF BACKPROP ITERATIONS
	batchSz = 100;		% BATCH SIZE (FOR BACKPROP)
	verbose = 1;		% DISPLAY OUTPUT
	bpCost = 'xent'		% COST FUNCTION FOR BACKPROP [CROSS-ENTROPY]
	nSearch = 3;		% NUMBER OF LINESEARCHES FOR DURING BACKPROP
	rho = 0;			% SPARSENESS CONSTRAINT ON HIDDEN ACTIVATIONS
	rhoHat = [];		% AVERAGE ACTIVATION OF OUTPUT UNITS (FOR SPARSITY)
end

methods
	% CONSTRUCTOR
	function self = dae(args,data,labels)
		if notDefined('data');data=[];end
		if notDefined('labels');labels=[];end

		if ~nargin 
			self = self.defaultDAE();
		elseif strcmp(args,'empty'); 
		% PASS SKELETON AUTOENCODER 
		else
			self = self.train(args,data);
		end
	end

	function print(self)
		properties(self)
		methods(self)
	end
	
	function self = preTrain(self,args,data);
	% GREEDY, LAYERWISE PRETRAINING
	
		if notDefined('data');load('defaultData.mat','data');end
		if notDefined('labels');load('defaultData.mat','labels');end

		self.nLayers = parseArgs(args);

		for iL = 1:self.nLayers

			layerArgs = parseArgs(args,iL);
			% INPUT LAYER (BERNOUILLI OR GAUSSIAN)
			if iL == 1
				% ENSURE HIDDEN UNITS ARE BERNOULLI
				tmp = layerArgs{find(strcmp(layerArgs,'type'))+1};
				tmp{:}(2) = 'B';
				layerArgs(find(strcmp(layerArgs,'type'))+1)=tmp;
				self.layers{iL} = rbm(layerArgs,data);

			% TOP/MIDDLE LAYER ARE BERNOULLI-GAUSSIAN
			elseif (iL == self.nLayers)
				layerArgs{find(strcmp(layerArgs,'type'))+1} = 'BG';
				self.layers{iL} = rbm(layerArgs,self.layers{iL-1}.eHid);

			% INTERMEDIATE LAYERS ARE BERNOULLI-BERNOUILLI
			else
				layerArgs{find(strcmp(layerArgs,'type'))+1}='BB';
				self.layers{iL} = rbm(layerArgs,self.layers{iL-1}.eHid);
			end

			if self.verbose
				self.printProgress(iL)
			end
			self.layers{iL} = self.layers{iL}.train;
		end
	end

	function daeOut = unRoll(self)
	% 'UNROLL' PRETRAINED RBMS TO FORM THE DEEP AUTOENCODER
		if self.verbose
			self.printProgress(-1);
		end
		daeOut = dae('empty'); % ENSURE OUTPUT IS DAE OBJECT
		daeOut.X = self.layers{1}.X;
		daeOut.nLayers = self.nLayers*2;
		daeOut.verbose = self.verbose;
		daeOut.layers = cell(1,2*self.nLayers);
		daeOut.batchSz = self.batchSz;

		% NOTE:
		% HERE WE THROW AWAY ALL UNNEEDED METADATA
		% STORED DURING RBM PRETRAINING;
		% USE RBMS AS SKELETON FOR HOLDING WEIGHTS
		for iL = 1:self.nLayers;
			daeOut.layers{iL} = rbm('empty');
			daeOut.layers{iL}.W = self.layers{iL}.W;
			daeOut.layers{iL}.b = self.layers{iL}.b;
			daeOut.layers{iL}.c = self.layers{iL}.c;
			daeOut.layers{iL}.type = self.layers{iL}.type;
			
			daeOut.layers{2*self.nLayers-iL+1}=rbm('empty');
			daeOut.layers{2*self.nLayers-iL+1}.W = self.layers{iL}.W';
			daeOut.layers{2*self.nLayers-iL+1}.b = self.layers{iL}.c;
			daeOut.layers{2*self.nLayers-iL+1}.c = self.layers{iL}.b;
			daeOut.layers{2*self.nLayers-iL+1}.type = fliplr(self.layers{iL}.type);
		end
		
		if self.verbose
			printStr = sprintf('Autoencoder architecture:[%d-',self.layers{1}.nVis);
			for iL = 1:daeOut.nLayers;
				if iL == daeOut.nLayers
					printStr = [printStr, sprintf('%d]\n',numel(daeOut.layers{iL}.c))];
				else
					printStr = [printStr,	 sprintf('%d-',numel(daeOut.layers{iL}.c))];
				end
			end
			fprintf('%s',printStr);
		end
	end

	function self = fineTune(self);
	% FINETUNE AUTOENCODER WEIGHTS USING BACKPROP
		% # (minFunc)
		minOpt.maxlinsearch = 100;
		minOpt.maxIter = 10;
		minOpt.display = 'off';
%  		minOpt.DerivativeCheck = 'on'; % TO CHECK GRADIENT FUNCTION

		% MAKE 10 SUPER BATCHES
		batches = self.createBatches();
		nBatches = numel(batches);
		sBatchSz = ceil(nBatches/10);
		batchBins = 1:sBatchSz:nBatches;
		for iB = 1:numel(batchBins)
			if iB < numel(batchBins)
				batchIdx{iB} = [batches{batchBins(iB):batchBins(iB+1)-1}];
			else
				batchIdx{iB} = [batches{batchBins(iB):nBatches}];
			end
		end
		if self.verbose
			self.printProgress(0);
			errOld = Inf*ones(1,nBatches);
			errStr = '';
		end
		nSearch = self.nSearch;
		for iI = 1:self.nFineTune
			cnt = 1;
			E = 0;
			col = randcol;
			nBatches = numel(batchIdx);
			for iB = 1:nBatches;
				batchX = self.X(batchIdx{iB},:);

				if iI == 1
					% INITIALIZE VECTORIZED WEIGHTS FOR minimize.m
					vectW = [];
					for iL = 1:self.nLayers
						vectW = [vectW ;self.layers{iL}.W(:); ...
								self.layers{iL}.c(:)];
					end
				end
				
				% BACKPROP MAIN
				vectW = minFunc(@bpGradient,vectW(:),minOpt,self,batchX);
				
				% ADJUST NETWORK WEIGHTS
				self = self.updateWeights(vectW);
				
				if self.verbose
					recon = self.propDataUp(batchX);
					err = sum(sum((batchX - recon).^2)) / size(batchX,1);
					if errOld(iB) > err
						errStr = 'DOWN';
					else
						errStr = 'UP';
					end
					fprintf('Iteration %d/%d: Batch %d/%d: Recon error: %6.3f (%s)       \r',iI,self.nFineTune,iB,nBatches,err,errStr);
					errOld(iB) = err;
					cnt = cnt + 1;
					hold on
				end
			end
		end
	end

	function self = updateWeights(self,W);
		idx = 1;
		for iL = 1:self.nLayers
			% DISTRIBUTE CONNECTION WEIGHTS
			nW = numel(self.layers{iL}.W);
			self.layers{iL}.W = reshape(W(idx:idx-1 + ...
			nW),size(self.layers{iL}.W));
			idx = idx + nW;

			% DISTRIBUTE HIDDEN BIASES
			nc = numel(self.layers{iL}.c);
			self.layers{iL}.c = reshape(W(idx:idx+nc-1), ...
			size(self.layers{iL}.c));
			idx = idx + nc;
		end
	end

	function batchIdx = createBatches(self)
	% MAKE MINI BATCHES FOR FINETUNING
		nObs = size(self.X,1);
		nBatches = ceil(nObs/self.batchSz);
		tmp = repmat(1:nBatches, 1, self.batchSz);
		tmp = tmp(1:nObs);
		randIdx=randperm(nObs);
		tmp = tmp(randIdx);
		for iB=1:nBatches
		    batchIdx{iB} = find(tmp==iB);
		end
	end

	function outPut = propDataUp(self,X,outType)
	% PROPAGATE DATA UP THROUGH LAYERS
		if notDefined('outType'), outType = 'recon'; end
		nObs = size(X,1);
		probs = [X,ones(nObs,1)]; clear X;

		for iL = 1:self.nLayers
			switch self.layers{iL}.type(2)
			case 'B'
				probs=[1./(1 + exp(-(probs*[self.layers{iL}.W; ...
				self.layers{iL}.c]))),ones(nObs,1)];
			case 'G'
				probs = [probs*[self.layers{iL}.W;self.layers{iL}.c], ...
				ones(nObs,1)];
			end
			% OUTPUT IS MIDDLE/LINEAR LAYER ACTIVATION
			if strcmp(outType,'map') && iL == ceil(self.nLayers/2);
				outPut = probs(:,1:end-1);
				break
			end
		end
		% OUTPUT IS ACTIVATION AT TOP OF NETWORK
		if notDefined('outPut'),outPut = probs(:,1:end-1); end
	end

	function [] = printProgress(self,layer);
		if layer > 0
			fprintf('\nPretraining layer %d/%d (%d-%d):\n',layer, ...
			 self.nLayers,self.layers{layer}.nVis,self.layers{layer}.nHid);
		elseif layer < 0
			fprintf('\nUnrolling to form the autoencoder:\n');
		else
			fprintf('\nFine-tuning using backprop:\n');
		end
	end

	function self = defaultDAE(self)
		load('defaultData.mat','data','labels');
		args = defaultDAE();
		self = self.train(args,data);
	end

	function self = train(self,args,data)
		self = self.preTrain(args,data);
		self = self.unRoll();
		if self.nFineTune
			self = self.fineTune();
		end
		mapX = self.propDataUp(data,'map');
		self.mapX = mapX;
	end
end % END METHODS

end % END CLASSDEF