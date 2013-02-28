
learnLoc = which('startLearning.m');
parentDir = fileparts(learnLoc);
<<<<<<< HEAD
% ADD PATHS
addpath(genpath(parentDir));

% SAVE CONSTANTS (SEE medalConstants.m)
root = parentDir;
data = fullfile(parentDir,'data');
modules = fullfile(parentDir,'modules');
save(fullfile(parentDir,'.config.mat'));
clear parentDir learnLoc root data modules
=======
addpath(genpath(parentDir));
>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014
