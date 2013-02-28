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
		
		try
			switch m.class
			case 'mlnn' % MULTI-LAYER NEURAL NET
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
				
			case 'mcrbm' % MEAN-COVARIANCE RBM
					m.C = gpuArray(single(m.C));
					m.P = gpuArray(single(m.P));
				m.W = gpuArray(single(m.W));
				m.bC = gpuArray(single(m.bC));
				m.bM = gpuArray(single(m.bM));
				m.bV = gpuArray(single(m.bV));
					m.dC = gpuArray(single(m.dC));
					m.dP = gpuArray(single(m.dP));
				m.dbC = gpuArray(single(m.dbC));
				m.dbV = gpuArray(single(m.dbV));
				m.dW = gpuArray(single(m.dW));
			m.dbM = gpuArray(single(m.dbM));
				m.wPenalty = gpuArray(single(m.wPenalty));
					m.eta0 = gpuArray(single(m.eta0));
					m.topoMask = gpuArray(single(m.topoMask));

			case 'rbm'  % STANDARD RBMS
			
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
				if strcmp(lower(m.type),'gb')
					m.sigma2 = gpuArray(single(m.sigma2));
				end


			case 'rbmClassifier'

			case 'crbm' % CONVLUTIONAL RBM
				m.eVis = gpuArray(single(m.eVis));
				m.eHid = gpuArray(single(m.eHid));
				m.eHid0 = gpuArray(single(m.eHid0));
				m.ePool = gpuArray(single(m.ePool));
				m.hidI = gpuArray(single(m.hidI));
				m.visI = gpuArray(single(m.visI));

			otherwise  % GENERAL DATASTRUCT (E.G. TEMP VARIABLES)
				fprintf('Unkown model type.')
			end
		catch
			f = fields(m);
			for iF = 1:numel(f);
				try
					m.(f{iF}) = gpuArray(single(m.(f{iF})));
				catch
				end
			end
		end
	else
		error('');
	end

catch
	fprintf('\nNo CUDA Capability on current host.\n');
	m.useGPU = 0;
	reset(m.gpuDevice);
	m.gpuDevice = [];
end