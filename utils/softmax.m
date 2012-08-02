function x = softmax(X,dim);
%  x = softmax(X,[dim]);
%---------------------------------------------------------------------------
% Compute the softmax of the matrix <X> along dimension <dim>. Default
% <dim> is 1.
%--------------------------------------------------------------------------
% DES

if notDefined('dim'), dim = 1; end

maxX = max(X, [], dim);
tmp = exp(bsxfun(@minus,X,maxX));
x = bsxfun(@rdivide,tmp,sum(tmp,dim));
