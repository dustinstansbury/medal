function topoMat = createTopographyMatrix(nUnits,type,major)
%  topoMat = createTopographyMatrix(nUnits,type,major)
%----------------------------------------------------------------------------------
% Create a mask of values that can be used to induce a toroidal topography across a
% set of basis functions.%----------------------------------------------------------------------------------
% INPUT:
% <nUnits>: - the number of units. generally helpful if this is a square of an
%             integer.
%
% <type>:   - a string indicating the type of topography mask to create:
%             + 'independent': - units are independent (i.e. identity mask)
%             + '1d-gaussian'- 1D topography with Gaussian interaction function
%             + '1d-3  -   '1D topo., where only closest units interact
%             + '1d-5' -   '1D topo., where the closest 2 units interact
%             + '2d-gaussian' -   '2D topo. with Gaussian interaction function
%             + '2d-3x3  - '2D topo., where units interact 3 x 3 neighborhood
%             + '2d-5x5' - '2D topo., where units interact 5 x 5 neighborhood
%
% <major>:  - treat units alog rows or columns. Basis functions are along columns
%            for row major and along rows for column major. Can be 'row' or 'col'

%
% OUTPUT:
% <topoMat>:- an [nUnits x nUnits] matrix that defines a topogrphical interaction
%             function.
%----------------------------------------------------------------------------------
% DES
if notDefined('major'),	major = 'row'; end % WORK ALONG ROWS OR COLUMNS AS FIRST DIM

topoMat = zeros(nUnits,nUnits);
nPix = ceil(sqrt(nUnits));
switch type
	case 'independent'
		topoMat = eye(nUnits);
		return
		
	case '1d-gaussian'
		f0 = fspecial('gaussian',nPix,nPix*.075);
		rmIdxLeft = 1:floor(nPix/2)-1;
		rmIdxRight = floor(nPix/2)+1:nPix;
		f0(:,rmIdxLeft) = 0;
		f0(:,rmIdxRight) = 0;
%  		f0 = f0/max(f0(:));
		
	case '1d-3'
		f0 = zeros(nPix);
		idx0 = floor(nPix/2)-1:floor(nPix/2)+1;
%  		f0(idx0,floor(nPix/2)) = 1/3;
		f0(idx0,floor(nPix/2)) = [0.25 0.5 0.25];

	case '1d-5'
		f0 = zeros(nPix);
		idx0 = floor(nPix/2)-2:floor(nPix/2)+2;
		f0(idx0,floor(nPix/2)) = 1/5;
		
	case '2d-gaussian'
		f0 = fspecial('gaussian',nPix,nPix*.075);
%  		f0 = f0/max(f0(:));
		
	case '2d-3x3'
		f0 = zeros(nPix);
		idx0 = floor(nPix/2)-1:floor(nPix/2)+1;
%  		f0(idx0,idx0) = 1/9;
		f0(idx0,idx0) = [0.0625, 0.125, 0.0625;
                         0.125, 0.25, 0.125;
                         0.0625 0.125, 0.0625];
		
	case '2d-5x5'
		f0 = zeros(nPix);
		idx0 = floor(nPix/2)-2:floor(nPix/2)+2;
		f0(idx0,idx0) = 1/25;
end

for iU = 1:nUnits
	[xCenter,yCenter] = ind2sub([nPix,nPix],iU);
	switch major
		case 'col'
			f = circshift(f0,[floor(yCenter-nPix/2),floor(xCenter-nPix/2)]);
		case 'row'
		f = circshift(f0,[floor(xCenter-nPix/2),floor(yCenter-nPix/2)]);
	end
	topoMat(:,iU) = f(:);
end