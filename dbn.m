classdef dbn
%-----------------------------------------------------------
% Deep Belief Network object class
%-----------------------------------------------------------
% DES

properties
	layers = [];		% FIT MODELS FOR EACH LAYER
	nLayers = [];		% # OF COMPRISING RBMS
	verbose = 1;		% DISPLAY OUTPUT
	classify = 1;		% CLASSIFY (MULTINOMIAL) AT TOP
end

methods
	% CONSTRUCTOR
	function self = dbn(args,data,labels)
		if notDefined('data');data=[];end
		if notDefined('labels');labels=[];end
		if ~nargin
			self = self.defaultDBN();
		else
			self = self.preTrain(args,data,labels);
		end
	end

	% MAIN
	function self = preTrain(self,args,data,labels);

		if notDefined('data');load('defaultData.mat','data');end
		if notDefined('labels');load('defaultData.mat','labels');end
		self.nLayers = parseArgs(args);

		for iL = 1:self.nLayers
			if self.verbose
				self.printProgress(iL)
			end
			layerArgs = parseArgs(args,iL);

			% INPUT LAYER
			if iL == 1
				self.layers{iL} = rbm(layerArgs,data);
				self.layers{iL} = self.layers{iL}.train;
			% CLASSIFIER LAYER
			elseif (iL == self.nLayers) & self.classify
				self.layers{iL} = rbmClassifier(layerArgs,self.layers{iL-1}.eHid,labels);
				self.layers{iL} = self.layers{iL}.train;
			% INTERMEDIATE LAYERS
			else
				self.layers{iL} = rbm(layerArgs,self.layers{iL-1}.eHid);
				self.layers{iL} = self.layers{iL}.train;
			end
		end
	end

	function samps = sample(self,data,nSamps,nIters)
		if notDefined('nSamps'),nSamps = 10; end
		if notDefined('nIters'),nIters = 1000; end
		
		if notDefined('data')
			for iS = 1:nSamps
				switch self.layers{end}.type(2)
				case 'B'
					data(iS,:) = binornd(1,.5,1,self.layers{end}.nVis);
				case 'G'
					data(iS,:) = normrnd(zeros(1,self.layers{end}.nVis), ...
								  ones(nSamps,self.layers{end}.nVis));
				end
			end
		end
		
		% SAMPLE THE TOP LAYER
		if self.verbose
			fprintf('\nSampling top layer (%d Gibbs Samples)\n',nIters);
		end
		for iS = 1:nSamps
			if self.verbose
				fprintf('sample (%d/%d)  \r',iS,nSamps);
			end
			samps0{iS}= self.layers{end}.sample(data,1,nIters);
		end
		
		if self.verbose
			fprintf('\nProjecting down \n');
		end
		
		% PROJECT DOWN NETWORK 
		for iS = 1:nSamps
			aHid = samps0{iS};
			for iL = self.nLayers-1:-1:1
				if strcmp(self.layers{iL}.type(2),'B')
					aHid = aHid > rand;
				end
				aHid = self.layers{iL}.HtoV(aHid,0);
			end
			samps{iS} = aHid;
		end
	end

	function [] = printProgress(self,layer);
		if layer > 0
			fprintf('\nTraining layer %d/%d:\n',layer, self.nLayers);
		else
			fprintf('\nStacking layer %d/%d:\n',layer, self.nLayers);
		end
	end
	
	function [pred,classErr,misClass]=predictClass(self,testData,testLabels)
		% PROPAGATE DATA THROUGH LAYERS
		testData0 = testData;
		for iL = 1:self.nLayers-1
			testData = self.layers{iL}.VtoH(testData);
		end
		if notDefined('testLabels')
			testLabels = [];
		end
		% PREDICT AT TOP/CLASSIFIER LAYER
		[pred,classErr,misClass] = self.layers{end}.predict(testData,testLabels);
	end

	function outPut = propDataUp(self,X)
	% PROPAGATE DATA UP THROUGH LAYERS
		nObs = size(X,1);
		probs = [X,ones(nObs,1)]; clear X;
		
		for iL = 1:self.nLayers
			switch self.layers{iL}.type(2)
			case 'B'
				probs=[1./(1 + exp(-(probs*[self.layers{iL}.W; ...
				self.layers{iL}.c]))),ones(nObs,1)]>rand;
			case 'G'
				probs = [probs*[self.layers{iL}.W;self.layers{iL}.c], ...
				ones(nObs,1)];
			end
		end
		% OUTPUT IS ACTIVATION AT TOP OF NETWORK
		if notDefined('outPut'),outPut = probs(:,1:end-1); end
	end
	
	function self = defaultDBN(self)
		load('defaultData.mat','data','labels');
		args = defaultDBN();
		self = self.preTrain(args,data,labels);
	end

end % END METHODS

end % END CLASSDEF