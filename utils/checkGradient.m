function [dDiff,c] = checkGradient(gradFun,X,eps,input1,input2,input3,input4,input5)
%  function diff = checkGradient(gradFun,X,eps,[input1...inputN])
%----------------------------------------------------------------------------
% Checks that the gradients output by <gradFun> are correct by comparing to
% a numerical approximation calculated from finite differences
%
% INPUT:
% <gradFun>:   - a string indicating a function which must output function
%                 values as well as partial derivatives of the function.
%                 (ie [fX, dfX] = gradFun(X,input*)
%
% <X>:         - input data at which to evaluate
%
% <dX>:        - amount of perturbation to employ for numerical approximation
%
% <input*>:    - other inputs to the function <gradFun>
%
% OUTPUT:
% <diff>:      - the difference in the evaluated and approximated gradients
%                (scaled by their respective norms)
%----------------------------------------------------------------------------
% DES

funStr = sprintf('%s(X',gradFun);
funDStr = sprintf('%s(X+dX',gradFun);

nInputs = nargin - 3;
for iI = 1:nInputs
	funStr = sprintf('%s,input%d',funStr,iI);
	funDStr = sprintf('%s,input%d',funDStr,iI);
end

funStr = sprintf('%s)',funStr);
funDStr = sprintf('%s)',funDStr);

fprintf('\nChecking gradient function against numerical approximation...\n');
[fX, df] = eval(funStr);		% CALC GRADIENTS

dApprox = zeros(length(X),1);
for iD = 1:length(X)
	dX = zeros(size(dApprox));	% RESET
	dX(iD) = dX(iD) + eps;		% PERTURB A DIMENSION
	x2 = eval(funDStr);
	dX = -dX;					% PERTURB OTHER DIRECTION
	x1 = eval(funDStr);
	dApprox(iD) = (x2 - x1)/(2*eps);
	fprintf('Dimension %d/%d: Difference %4.2f\r',iD,length(X),abs(dApprox(iD)-df(iD)));
	try
		figure(2); scatter(dApprox(1:iD),df(1:iD)); drawnow
	catch
		keyboard
	end
end

fprintf('\ndone.');
c = [df,dApprox];
disp(c);
dDiff = norm(dApprox - df)/norm(dApprox + df);
