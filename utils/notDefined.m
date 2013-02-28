function nDefined = notDefined(varString)
%
% nDefined= notDefined( varString )
%
% Determines if a variable is defined in the workspace. 
% A variable is defined if (a) it exists and (b) it is not empty.
%
% nDefined: 1 (true) if the variable is not defined in the calling workspace 
%           0 (false) if the variable is defined in the calling workspace
%
%  This routine replaces calls of the form:
%    if ~exist('varname','var') || isempty(xxx) with the call
%    if notDefined('varname')
%

if (~ischar(varString)), error('Varible name must be a string'); end

nDefined = 1; 

str = sprintf('''%s''',varString);   
cmd1 = ['~exist(' str ',''var'') == 1'];
cmd2 = ['isempty(',varString ') == 1'];

% If either of these conditions holds, Defined == true
nDefined = evalin('caller',cmd1);     % Check that the variable exists in the caller space
if nDefined, return;                  % If it does not, return with a status of 0
else 
    nDefined = evalin('caller',cmd2); % Check if the variable is empty in the caller space
    if nDefined return;    
    else nDefined = 0;        
    
    end
end

return;
