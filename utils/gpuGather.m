function m = gpuGather(m)
%  m = gpuGather(m)
%---------------------------------------------------------------------------
% Gather values of fields for model <m> that exist on the GPU Card.
% Loops over fields and subfields of model fo find any gpuArray objects, 
% then pulls the corresponding data to the workspace.
%---------------------------------------------------------------------------
% DES

f = fields(m);

for iF = 1:numel(f)
	if strcmp(class(m.(f{iF})),'parallel.gpu.GPUArray')
		m.(f{iF}) = gather(m.(f{iF}));
	else
		if iscell(m.(f{iF}))
			cnt = 1;
			for jL = 1:numel(m.(f{iF}))
				try
					ff = fields(m.(f{iF}){jL});
					for kF = 1:numel(ff)
						if strcmp(class(m.(f{iF}){jL}.(ff{kF})),'parallel.gpu.GPUArray'),
							m.(f{iF}){jL}.(ff{kF}) = gather(m.(f{iF}){jL}.(ff{kF}));
						end
					end
				catch
					cnt = cnt+1;
				end
			end
%  			 (ASSUME NO MORE THAN 10 LAYERS TO SAVE UNEEDED LOOPS)
			if cnt > 10  % BIT OF A HACK
				break
			end
		elseif isstruct(m.(f{iF}))
			ff = fields(m.(f{iF}));
			try
				for kF = 1:numel(ff)
					if strcmp(class(m.(f{iF}).(ff{kF})),'parallel.gpu.GPUArray'),
						m.(f{iF}).(ff{kF}) = gather(m.(f{iF}).(ff{kF}));
					end
				end
			catch
				cnt = cnt+1;
			end
		end
		
	end
end