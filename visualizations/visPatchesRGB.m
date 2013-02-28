function imOut = visPatchesRGB(W, invXForm, transIms, cLims)
% imOut = visPatchesRGB(W, invXForm, cLims)
%-------------------------------------------------------------------

if notDefined('invXForm'); invXForm = eye(size(W,1)); end
borderPix = 1;

<<<<<<< HEAD
W = gather(invXForm * W);
=======
W = invXForm * W;
>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014

minW=min(W(:));
maxW=max(W(:));

[nDim,nUnits]=size(W);

<<<<<<< HEAD
% NORMALIZE BY L2 NORM (BOOST CONTRAST)
W = bsxfun(@rdivide,W,sqrt(dot(W,W)/nDim + .5));

=======
>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014
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
<<<<<<< HEAD
	imOut = W;
=======
	imOut = W
>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014
end
if notDefined('cLims'),cLims = [minW,maxW]; end

imOut = imOut/255;

if ~nargout
	try
<<<<<<< HEAD
		imshow(double(imOut));
%  		set(gca,'clim',cLims);
=======
		imagesc(imOut);
		set(gca,'clim',cLims);
>>>>>>> 87b603f3cd257a31f0e649b9a1e396cabf5c6014
		axis image;
		axis off;
	catch
	end

end
