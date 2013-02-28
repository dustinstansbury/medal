function out = dotdelay(kernel, in)
%function out = dotdelay(kernel, in)
%
% calculate linear responses of a system kernel to an input
%
% INPUT:
% [kernel] = kernel N x D matrix where N is number of channels and
%            D is number of delay lines.
%     [in] = input N x S matrix where S is the number of samples
%
% OUTPUT:
%    [out] = vector of kernel responses to input
%

ktsize = size(kernel,2);
ktsize2c = ceil(ktsize/2);
ktsize2f = floor(ktsize/2);
itsize = size(in, 2);

outs= (kernel'*in)';
for ii=1:ktsize2c
	outs(:,ii) = [outs(ktsize2c-ii+1:end,ii); zeros(ktsize2c-ii,1)];
end
for ii=1:ktsize2f
	ti = ii+ktsize2c;
	outs(:,ti) = [zeros(ii,1); outs(1:end-ii,ti)];
end
out = sum(outs,2);
