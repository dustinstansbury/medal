function m = gpuGather(m)
%  m = gpuGather(m)
%---------------------------------------------------------------------------
% Gather values of fields for model <m> that exist on the GPU Card.
% Loops over fields and subfields of model fo find any gpuArray objects, 
% then pulls the corresponding data to the workspace.
%---------------------------------------------------------------------------
% DES

