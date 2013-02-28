function imOut = visPatchesRGB(W, invXForm, transIms, cLims)
% imOut = visPatchesRGB(W, invXForm, cLims)
%-------------------------------------------------------------------

if notDefined('invXForm'); invXForm = eye(size(W,1)); end
borderPix = 1;

W = gather(invXForm * W);

minW=min(W(:));
maxW=max(W(:));

[nDim,nUnits]=size(W);

% NORMALIZE BY L2 NORM (BOOST CONTRAST)
W = bsxfun(@rdivide,W,sqrt(dot(W,W)/nDim + .5));

nDPix = floor(sqrt(nDim/3)+0.999);
nCols = floor(sqrt(nUnits)+0.999);

imOut = zeros(((nDPix+borderPix)*nCols+borderPix),...
			  ((nDPix+borderPix)*nCols+borderPix),3);

scale = 127/max(abs(minW), abs(maxW));

try
	for iW=1:nUnits
		rowIdx = borderPix+floor((iW-1)/nCols)*(borderPix+nDPix)+1:borderPix+(1+floor((iW-1)/nCols))*(borderPix+nDPix)-1;
		
		colIdx = borderPix+mod(iW-1,nCols)*(borderPix+nDPix)+1: ...
				 borderPix + (1+mod(iW-1,nCols))*(borderPix+nDPix)-1;
				 
		imOut(rowIdx,colIdx,:) = reshape(W(:,iW),nDPix,nDPix,3)*scale + 128;
	end;
catch % IF ALL ELSE FAILS, JUST SHOW THE WEIGHTS
	fprintf('\nVisualization failed, displaying weights...\n')
	imOut = W;
end
if notDefined('cLims'),cLims = [minW,maxW]; end

imOut = imOut/255;

if ~nargout
	try
		imshow(double(imOut));
%  		set(gca,'clim',cLims);
		axis image;
		axis off;
	catch
	end

end
