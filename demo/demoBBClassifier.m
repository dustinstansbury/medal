% INITIALIZE AND TRAIN DEFAULT RBM CLASSIFIER

clear rc; rc = rbmClassifier(); rc = rc.train();

load('defaultData.mat');

figure(2)
subplot(121);
plot(rc.e); axis square;
title(sprintf('Reconstruction Error\nCD[%d]',rc.nGibbs));
xlabel('Iteration');drawnow
subplot(122);
rc.vis();title('Learned Weights');

% PREDICT CLASSES OF HOLD-OUT SET
[pred,classError,misClass]=rc.predict(testdata,testlabels);

figure(3);
rc.vis(testdata(misClass,:)');
title(sprintf('Misclassifications\n(Error rate = %2.2f%%)',classError*100));

