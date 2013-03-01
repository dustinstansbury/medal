function checkGradients(model, input, targets)
    epsilon = 1e-6;
    tol = 1e-6;
    switch model.class
    case {'mlnn','ae'}
    
    % CALCULATE ACTUAL NET GRADIENTS
    rng(12345);
    model = model.fProp(input,targets);
    model = model.bProp;
    
    for lL = 1:(model.nLayers - 1) % LOOP OVER LAYERS
	    fprintf('Layers %d / %d',lL,model.nLayers-1);
	    fprintf('\nChecking Weight Gradients\n');
        for r = 1 : size(model.layers{lL}.W, 1) % LOOP OVER ROWS
            for c = 1 : size(model.layers{lL}.W, 2) % LOOP OVER COLS

	            % PLUS AND MINUS NETS
                minusNet = model;
                plusNet = model;

                % PERTURB WEIGHTS RESPECTIVELY
                minusNet.layers{lL}.W(r, c) = model.layers{lL}.W(r, c) - epsilon;
                plusNet.layers{lL}.W(r, c) = model.layers{lL}.W(r, c) + epsilon;

                % FORWARD THROUGH PLUS NET
                rng(12345);
                minusNet = minusNet.fProp(input, targets);

                % FORWARD THROUGH MINUS NET
                rng(12345);
                plusNet = plusNet.fProp(input, targets);

                % COMPUTE NUMERICAL GRADIENT
                dW = (plusNet.J - minusNet.J) / (2 * epsilon);

                % COMPARE NUMERICAL AND ACTUAL
                err = abs(dW - model.layers{lL}.dW(r, c));
				
				model.auxVars.numGrad = dW;
				model.auxVars.netGrad = model.layers{lL}.dW(r, c);
				model.auxVars.gradFailed = err > tol;
				model.printProgress('gradCheck');
            end
            
        end

        fprintf('\nChecking Bias Gradients\n');
		% REPEAT FOR BIASES...
        for r = 1:size(model.layers{lL}.b, 1)
	        minusNet = model;
            plusNet = model;
            
            minusNet.layers{lL}.b(r) = model.layers{lL}.b(r) - epsilon;
            plusNet.layers{lL}.b(r) = model.layers{lL}.b(r) + epsilon;
            
            rng(12345);
            minusNet = minusNet.fProp(input, targets);
            
            rng(12345);            
            plusNet = plusNet.fProp(input, targets);
            
            db = (plusNet.J - minusNet.J) / (2 * epsilon);
            err= abs(db - model.layers{lL}.db(r));
			model.printProgress('gradCheck');
        end
    end
    case 'mlcnn'
	rng(12345);
    model = model.fProp(input,targets);
    model = model.bProp;

    for lL = 1:model.nLayers % LOOP OVER LAYERS
	    switch model.layers{lL}.type

		case 'conv'
			fprintf('\nChecking Layer %d Filter Gradients\n',lL);
			for jM = 1:model.layers{lL}.nFM
				for iM = 1:model.layers{lL-1}.nFM
					for r = 1 : size(model.layers{lL}.filter, 1) % LOOP OVER ROWS
			            for c = 1 : size(model.layers{lL}.filter, 2) % LOOP OVER COLS
				            % PLUS AND MINUS NETS
			                minusNet = model;
			                plusNet = model;

			                % PERTURB FILTER WEIGHTS RESPECTIVELY
			                minusNet.layers{lL}.filter(r,c,iM,jM) = model.layers{lL}.filter(r,c,iM,jM) - epsilon;
			                
			                plusNet.layers{lL}.filter(r,c,iM,jM) = model.layers{lL}.filter(r,c,iM,jM) + epsilon;

			                % FORWARD THROUGH PLUS NET
			                rng(12345);
			                minusNet = minusNet.fProp(input, targets);

			                % FORWARD THROUGH MINUS NET
			                rng(12345);
			                plusNet = plusNet.fProp(input, targets);
							
			                % COMPUTE NUMERICAL GRADIENT
			                dW = (plusNet.J - minusNet.J) / (2 * epsilon);

			                % COMPARE NUMERICAL AND ACTUAL
			                err = abs(dW - model.layers{lL}.dFilter(r, c,iM,jM));

							model.auxVars.numGrad = dW;
							model.auxVars.netGrad = model.layers{lL}.dFilter(r, c,iM,jM);
							model.auxVars.gradFailed = err > tol;
							if model.auxVars.gradFailed
%  								keyboard
							end	
							model.printProgress('gradCheck');
			            end
			        end
		        end
			end

			fprintf('\nChecking Layer %d Bias Gradients\n',lL);
			for r = 1:numel(model.layers{lL}.b)
		        minusNet = model;
	            plusNet = model;

	            minusNet.layers{lL}.b(r) = model.layers{lL}.b(r) - epsilon;
	            plusNet.layers{lL}.b(r) = model.layers{lL}.b(r) + epsilon;

	            rng(12345);
	            minusNet = minusNet.fProp(input, targets);

	            rng(12345);
	            plusNet = plusNet.fProp(input, targets);

	            % NUMERICAL GRADIENT
	            db = (plusNet.J - minusNet.J) / (2 * epsilon);
	            err = abs(db - model.layers{lL}.db(r));
				model.printProgress('gradCheck');
	        end
	    
	    case 'output'
		    fprintf('\nChecking Output Layer Weight Gradients\n');
	        for r = 1 : size(model.layers{end}.W, 1) % LOOP OVER ROWS
	            for c = 1 : size(model.layers{end}.W, 2) % LOOP OVER COLS
		            % PLUS AND MINUS NETS
	                minusNet = model;
	                plusNet = model;

	                % PERTURB WEIGHTS RESPECTIVELY
	                minusNet.layers{lL}.W(r,c) = model.layers{lL}.W(r,c) - epsilon;
	                plusNet.layers{lL}.W(r,c) = model.layers{lL}.W(r,c) + epsilon;

	                % FORWARD THROUGH PLUS NET
	                rng(12345);
	                minusNet = minusNet.fProp(input, targets);

	                % FORWARD THROUGH MINUS NET
	                rng(12345);
	                plusNet = plusNet.fProp(input, targets);

	                % COMPUTE NUMERICAL GRADIENT
	                dW = (plusNet.J - minusNet.J) / (2 * epsilon);

	                % COMPARE NUMERICAL AND ACTUAL
	                err = abs(dW - model.layers{lL}.dW(r, c));

					model.auxVars.numGrad = dW;
					model.auxVars.netGrad = model.layers{lL}.dW(r, c);
					model.auxVars.gradFailed = err > tol;
					model.printProgress('gradCheck');
	            end
	        end

	        fprintf('\nChecking Output Layer Bias Gradients\n');
	        for r = 1:numel(model.layers{lL}.b)
		        minusNet = model;
	            plusNet = model;

	            minusNet.layers{lL}.b(r) = model.layers{lL}.b(r) - epsilon;
	            plusNet.layers{lL}.b(r) = model.layers{lL}.b(r) + epsilon;

	            rng(12345);
	            minusNet = minusNet.fProp(input, targets);

	            rng(12345);
	            plusNet = plusNet.fProp(input, targets);

	            % NUMERICAL GRADIENT
	            db = (plusNet.J - minusNet.J) / (2 * epsilon);
	            err = abs(db - model.layers{lL}.db(r));
				model.printProgress('gradCheck');
	        end
        end
    end
    
	end % END SWITCH