function [logZ, logZUB, logZLB, logW] = aisBB(RBM,M,nBeta);
%  [logZ, logZLB, logZLB, logW] = aisBB(RBM,[M],[beta]);
%-----------------------------------------------------------------------------
% Estimate the log partition function of a Bernoulli-Bernoulli RBM using
% Annealed Importance Sampling.
%-----------------------------------------------------------------------------
%INPUT:
% <RBM>:     - a trained RBM object
%
% <M>:       - the number of AIS runs to perform. Default = 100;
%
% <beta>:    - a an integer defining the resolution of the interpolation grid
%              define intermediate distributions. Default = 10000;
%
%
%OUTPUT:
% <logZ>:    - an estimate of the log partition function of the trained RBM
%
% <logZUB>:  - the upper bound (+ 2 standard deviations) of the estimate on
%              logZ
%
% <logzLB>:  - the lower bound (- 2 standard deviations) of the estimate on
%              logZ
%
% <logW>:    - the log importance weights
%-----------------------------------------------------------------------------
% Adapted from RBM_AIS.m ver 1.00 by Ruslan Salakhutdinov, and the paper
% "Learning and Evaluating Boltzmann Machines" (2008) by Ruslan Salakhutinov
%-----------------------------------------------------------------------------
% DES

if notDefined('M'), M = 100; end
if notDefined('nBeta'),nBeta = 10000; end

[nVis nHid]=size(RBM.W);
beta = linspace(0,1,nBeta+1);

% BIASES OF ESTIAMTED AND BASE-RATE MODELS
b = repmat(RBM.b,M,1);
c = repmat(RBM.c,M,1);
b0 = zeros(size(RBM.b));
bs0 = repmat(b0,M,1);

% SAMPLE FROM BASE MODEL AND CALC.
% INITIAL LOG-IMPORTANCE WEIGHTS
logW = zeros(M,1);
aVis = repmat(RBM.sigmoid(b0),M,1) > rand(M,nVis);
logW = logW - (aVis*b0' + nHid*log(2));
logZ0 = sum(log(1+exp(b0))) + (nHid)*log(2);

hidEnergy = aVis*RBM.W + c;	% CALC INNITIAL HIDDEN ENERGY STATE
visBFactorA = aVis*b0';		% INITIALIZE MODEL K
visBFactorB = aVis*b';		% INITIALIZE MODEL K+1

% MAIN AIS
for k = 2:(numel(beta)-1);
	% (????)
	expHidWFactor = exp(beta(k)*hidEnergy);
	
	% INTERPOLATE MODELS -- EQ 39
	Ek = (1-beta(k))*visBFactorA + beta(k)*visBFactorB;
	logW = logW + Ek + sum(log(1+expHidWFactor),2);

	% SAMPLE HIDDENS (???)
	pHid = expHidWFactor./(1 + expHidWFactor);
	aHid = pHid > rand(M,nHid);

	% SAMPLE VISIBLES -- p(v' | h) EQ 42
	pVis = RBM.sigmoid((1-beta(k))*bs0 + beta(k)*(aHid*RBM.W + b));
	aVis = pVis > rand(M,nVis);

	% UPDATE ENERGY AND MODELS GIVEN NEW SAMPLES
	hidEnergy = aVis*RBM.W' + c;
	visBFactorA = aVis*b0;
	visBFactorB = aVis*b;
	expHidWFactor = exp(beta(k)*hidEnergy);

	% CALCULATE LOG RATIO OF PARTITION
	% FUNCTIONS; ACCUMULATE IMPORTANCE WEIGHT
	logW = logW - ((1-beta(k))*visBFactorA + beta(k)*visBFactorB + sum(log(1+expHidWFactor),2));
end

expHidWFactor = exp(hidEnergy);
logW  = logW + aVis*b' + sum(log(1+expHidWFactor),2);

% RATIO OF INITIAL AND AND FINAL
% DISTRIBUTION PARTIION FUNCTIONS
% Z_K / Z_0 -- Eq. 23
rAIS = logsum(logW(:)) - log(M);

% ESTIMATE LOG PARTITION FUNCTION
% FROM THIS RATIO
logZ = rAIS + logZ0;

if nargout > 2	
	% COMPUTE ERROR BOUNDS (+/- 2 STANDARD DEVIATIONS).
	% CALCULATE LOG STANARD DEVIATION
	%  logstd_AIS = log(std(exp(logW(:)))/sqrt(M));
	stdDev = 2;
	mlW = mean(logW(:));
	logZSTD = log (std(exp(logW-mlW))) + mlW - log(M)/2;
	logZUB = logsum([log(stdDev)+logZSTD;rAIS]) + logZ0;
	logZLB = logdiff([(log(stdDev)+logZSTD);rAIS]) + logZ0;
end