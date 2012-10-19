function m = gpuDistribute(m)
%  m = gpuDistribute(m)
%---------------------------------------------------------------------
% Distibutes the fields of the model <m> onto the GPU card.
%---------------------------------------------------------------------
% DES

try
	d = gpuDevice;
	if d.DeviceSupported
		m.gpuDevice = d;

		switch m.class
		case 'mlnn'
			for iL = 1:numel(m.layers)
				m.layers{iL}.W = gpuArray(single(m.layers{iL}.W));
				m.layers{iL}.dW = gpuArray(single(m.layers{iL}.dW));
				m.layers{iL}.bias = gpuArray(single(m.layers{iL}.bias));
				m.layers{iL}.dBias = gpuArray(single(m.layers{iL}.dBias));
			end
			m.X = gpuArray(single(m.X));
			m.trainError = gpuArray(single(m.trainError));
			m.xValError = gpuArray(single(m.xValError));
			m.netOutput = gpuArray(m.netOutput);

		otherwise
			% GENERAL INITIALIZATIONS
			m.X = gpuArray(single(m.X));
			m.W = gpuArray(single(m.W));
			m.dW = gpuArray(single(m.dW));
			m.b = gpuArray(single(m.b));
			m.db = gpuArray(single(m.db));
			m.c = gpuArray(single(m.c));
			m.dc = gpuArray(single(m.dc));
			m.log.err = gpuArray(single(m.log.err));
			m.log.eta = gpuArray(single(m.log.eta));


			% CLASS-SPECIFIC INITS
			switch m.class
			case 'rbm'
				if strcmp(lower(m.type),'gb')
					m.sigma2 = gpuArray(single(m.sigma2));
				end


			case 'rbmClassifier'

			case 'crbm'
				m.eVis = gpuArray(single(m.eVis));
				m.eHid = gpuArray(single(m.eHid));
				m.eHid0 = gpuArray(single(m.eHid0));
				m.ePool = gpuArray(single(m.ePool));
				m.hidI = gpuArray(single(m.hidI));
				m.visI = gpuArray(single(m.visI));

			case 'mlnn'

			end
		end
	else
		error('');
	end

catch
	fprintf('\nNo CUDA Capability on current host.\n');
	m.useGPU = 0;
end