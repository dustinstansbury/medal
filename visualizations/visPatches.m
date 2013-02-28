function imOut = visPatches(W, invXForm, transIms, cLims, borderPix)
% imOut = visZCAWhitenedPatches(W, invXForm, transIms, cLims, borderPix)
%-------------------------------------------------------------------

if notDefined('invXForm'); invXForm = eye(size(W,1)); end
if notDefined('transIms'); transIms = 0; end
if notDefined('borderPix'); borderPix = 1; end

W = invXForm * W;

minW=min(W(:));
maxW=max(W(:));

[nDim,nUnits]=size(W);

nDPix = floor(sqrt(nDim)+0.999);
nUPix = floor(sqrt(nUnits)+0.999);

imOut = zeros(((nDPix+borderPix)*nUPix+borderPix));

if (nUnits/nUPix<=nUPix-1),
    imOut = imOut(:,1:(nDPix+borderPix)*(nUPix-1)+borderPix);
end

scale = 127/max(abs(minW), abs(maxW));

try
	for iW=1:nUnits;
	    if (transIms)
		    imOut(mod(iW-1,nUPix)*(nDPix+borderPix) + ...
		    borderPix+1:mod(iW-1,nUPix)*(nDPix+borderPix) + ...
		    borderPix+nDPix,floor((iW-1)/nUPix)*(nDPix+borderPix)+borderPix + ...
		    1:floor((iW-1)/nUPix)*(nDPix+borderPix)+borderPix+nDPix)...
		    = (reshape(W(:,iW),nDPix,nDPix)'*scale + 128);
	    else
		    imOut(mod(iW-1,nUPix)*(nDPix+borderPix) + ...
		    borderPix+1:mod(iW-1,nUPix)*(nDPix+borderPix) + ...
		    borderPix+nDPix,floor((iW-1)/nUPix)*(nDPix+borderPix)+borderPix + ...
		    1:floor((iW-1)/nUPix)*(nDPix+borderPix)+borderPix+nDPix)...
		    = reshape(W(:,iW),nDPix,nDPix)*scale + 128;
	    end
	end;
catch % IF ALL ELSE FAILS, JUST SHOW INVERSE-TRANSFORMED WEIGHTS
	fprintf('\nVisualization failed, displaying weights...\n')
	imOut = W;
end
if notDefined('cLims')cLims = [minW,maxW]; end

if ~nargout
	try
		imagesc(imOut/255); colormap(gray);
%  		set(gca,'clim',cLims)
		axis image;
		axis off;
	catch
	end

end
