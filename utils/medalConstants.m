function c = medalConstants(query);
%  c = medalConstants(query);%--------------------------------------------------------------------------
% Helper function for MEDAL.
%--------------------------------------------------------------------------
% DES

rootDir = which('medalConstants.m');
tmp = fileparts(rootDir);
rootDir = fileparts(tmp);

cTmp = load(fullfile(rootDir,'.config.mat'),query);
f = fields(cTmp);
c = cTmp.(f{:});
