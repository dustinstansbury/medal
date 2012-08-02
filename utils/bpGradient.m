function [J, dJ] = bpGradient(vectW, net, X, targs)
%  [J dJ]=bpGradient(vectW,net,X,[targs])
%--------------------------------------------------------------------------
% Gradient function for use with Rassmussen's minFunc.m to perform
% backprobagation finetuning of network weights. The cost may be based on
% the cross-entropy or mean-squared error cost functions.
%
% INPUT:
% <vectW>:  - a seed point for the parameter values (a' la minFunc.m)
%
% <net>:    - a multilayer network <net> as input (see dbn.m dbm.m and dae.m)
%             The function assumes that each cell in <net.layers> is an rbm
%             object.
%
% <X>:      - values at which to calculate the gradient.
%
% <targs>:  - target variables for calculating the value of the cost function. %             If <targs> is not provided, the function assumes that the
%             targets are the prediction vectors <X>.
%---------------------------------------------------------------------------
% (Some code based on MNIST_CG.m from Salakhutidnov & Hinton, 2006)
% DES

if notDefined('targs')
	targs = X;
end
nObs = size(X, 1);


%  % UNPACK AND DISTRIBUTE WEIGHTS
if ~isempty(vectW)
	net = net.updateWeights(vectW);
end


% PROPAGATE DATA UP (STORE PROPAGATIONS)
prop = cell(1,net.nLayers+1);
prop{1} = [X,ones(nObs,1)]; clear X;

for iL = 1:net.nLayers
	switch net.layers{iL}.type(2)
	case 'B'
		prop{iL+1}=[1./(1 + exp(-(prop{iL}*[net.layers{iL}.W; ...
		net.layers{iL}.c]))),ones(nObs,1)];
	case 'G'
		prop{iL+1} = [prop{iL}*[net.layers{iL}.W;net.layers{iL}.c], ...
		ones(nObs,1)];
	end
end
outPut = prop{end}(:,1:end-1);

% COST FUNCTIONS
switch net.bpCost
case 'mse'   % MEAN-SQUARED ERROR
	J = -sum((outPut - targs).^2)/nObs;
case 'xent'  % CROSS-ENTROPY
	J = -sum(sum(targs.*log(outPut) + ...
	(1 - targs).*log(1 - outPut)))/nObs;
end

% COMPUTE COST GRADIENTS
dW = cell(1, net.nLayers); db = cell(1, net.nLayers);

% PROPAGATE ERROR DOWN
Err = (outPut - targs)/nObs;
for iL=net.nLayers:-1:1

    % PARITIALS (dE/dW)
	switch net.bpCost 
	case 'mse'
	    d = prop{iL}'*Err;
	case 'xent'
	    d = prop{iL}'*Err;
	end
     
	dW{iL} = d(1:end - 1,:);db{iL} = d(end,:);
    
	if iL > 1
	% ACTIVATION FUNCITON DERIVATIVES
		switch net.layers{iL-1}.type(2)
		case 'B'
			fPrime = prop{iL}.*(1 - prop{iL});
		case 'G'
			fPrime = 1;
		end        
		Err = (Err * [net.layers{iL}.W; net.layers{iL}.c]').*fPrime;
		Err = Err(:,1:end - 1);
	end
end

% VECTORIZE PARTIALS FOR minimize.m
dJ = zeros(size(vectW));
idx = 1;
for iL=1:net.nLayers
	nW = numel(dW{iL}); nb = numel(db{iL});
	dJ(idx:idx - 1 + nW) = dW{iL}(:);
	idx = idx + nW;
    
	dJ(idx:idx - 1 + nb) = db{iL}(:);
	idx = idx + nb;
end
