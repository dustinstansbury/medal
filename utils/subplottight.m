function h=subplottight(rows, cols, ax, margin)
%  h=subplottight(rows, cols, ax, margin)
%-------------------------------------------------------------------------
% This function is like subplot but it removes the spacing between axes.
%-------------------------------------------------------------------------
% INPUT:
% <rows>:     - number of rows 
%
% <cols>:     - number of columns
%
% <ax>:       - requested subplot axis
%
% <margin>    - amount of margin (in percent) separating axes
%
% OUTPUT:
% <h>:        - handle to the axis
%-------------------------------------------------------------------------
% DES

if nargin < 4
    margin = 0.01;
end

ax = ax-1;
x = mod(ax,cols)/cols;
y = (rows-fix(ax/cols)-1)/rows;
h=axes('position', [x+margin/cols, ...
                    y+margin/rows, ...
                    1/cols-2*margin/cols, ...
                    1/rows-2*margin/rows]);

