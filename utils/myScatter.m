function h = myScatter(data,props,szScale,colScale,szLims,h0)
%  h = myScatter(data,props,sizeScale,colScale,szLims,h0)

if notDefined('props')
	props = {'Marker','o','MarkerEdgeColor','k'};
end

if notDefined('h0')
	h0 = gca;
end

[nMeasure,nDim] = size(data);

if nDim ~= 2
	data = data';
	[nMeasure, nDim] = size(data);
	if nDim ~= 2
		error('data incompatible');
	end
end

if notDefined('szLims')
    szLims = [10 200];
end

if notDefined('szScale')
	szScale = 25; %max(data(:))/10;
else
	if numel(szScale) == nMeasure
		szScale = rescaleData(szScale,szLims);
	else
		error('data size szScale incorrect size')
	end
end

if notDefined('colScale')
	h = scatter(h0,data(:,1),data(:,2),szScale,props{:});
else
	if length(colScale) == nMeasure
		h = scatter(h0,data(:,1),data(:,2),szScale,colScale,'filled','MarkerEdgeColor','none');
	else
		error('provided color szScale incorrect size')
	end
end

set(gcf,'color','w');
