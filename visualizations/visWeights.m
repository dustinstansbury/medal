function imOut = visWeights(W, transIms, cLims, sortNorms, nShow, borderPix)
%  imOut = visWeights(W, transIms, cLims, sortNorms, borderPix,nShow)
%-------------------------------------------------------------------
if notDefined('cLims'), cLims = 1; end; % SCALE BY L2 NORM BY DEFAULT
if notDefined('sortNorms'); sortNorms = 0; end
if notDefined('transIms'); transIms = 0; end
if notDefined('borderPix'); borderPix = 1; end
if notDefined('nShow'), nShow = size(W,2);end

wNorm = sum(W.^2,1)/size(W,1);

if sortNorms
    [wNorm,sortIdx] = sort(wNorm,'descend');
    W = W(:,sortIdx);
end

W = W(:,1:nShow);
wNorm = wNorm(1:nShow);

% SCALE EACH FEATURE BY ITS L2 NORM
if numel(cLims) < 2 & cLims
	W = bsxfun(@rdivide,W,wNorm);
end

minW=min(W(:));
maxW=max(W(:));

if numel(cLims) < 2,cLims = [minW,maxW]; end

[nDim,nUnits]=size(W);

nDPix = floor(sqrt(nDim)+0.999);
nUPix = floor(sqrt(nUnits)+0.999);

imOut = -(minW+maxW)/2*ones(((nDPix+borderPix)*nUPix+borderPix));

if (nUnits/nUPix<=nUPix-1),
    imOut = imOut(:,1:(nDPix+borderPix)*(nUPix-1)+borderPix);
end;

try
	for iW=1:nUnits;
	    if (transIms)
		    imOut(mod(iW-1,nUPix)*(nDPix+borderPix) + ...
		    borderPix+1:mod(iW-1,nUPix)*(nDPix+borderPix) + ...
		    borderPix+nDPix,floor((iW-1)/nUPix)*(nDPix+borderPix)+borderPix + ...
		    1:floor((iW-1)/nUPix)*(nDPix+borderPix)+borderPix+nDPix)...
		    = reshape(W(:,iW),nDPix,nDPix)';
	    else
		    imOut(mod(iW-1,nUPix)*(nDPix+borderPix) + ...
		    borderPix+1:mod(iW-1,nUPix)*(nDPix+borderPix) + ...
		    borderPix+nDPix,floor((iW-1)/nUPix)*(nDPix+borderPix)+borderPix + ...
		    1:floor((iW-1)/nUPix)*(nDPix+borderPix)+borderPix+nDPix)...
		    = reshape(W(:,iW),nDPix,nDPix);
	    end
	end;
catch % IF ALL FAILS, JUST SHOW THE WEIGHTS
	imOut = W;
end
imagesc(imOut); colormap(gray);
try
	set(gca,'clim',cLims)
catch
end
axis image;
axis off;
