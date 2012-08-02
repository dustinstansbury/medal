function act = softplus(x,upperBound)
%  act = softplus(x,upperBound)
%-----------------------------------------------------
% Softplus activation function
%-----------------------------------------------------

if notDefined('upperBound')
	upperBound = 20;
end
act = log(1 + exp(x));

upper = find(x > upperBound);
act(upper) = x(upper);
