function act = softRect(x)
%  act = softRect(x)
%-----------------------------------------------------
% Soft rectification activation function
%-----------------------------------------------------


act = log(1 + exp(x));
