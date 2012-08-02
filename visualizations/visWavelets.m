function [out, strf] = visWavelets(strf, visopt)
%function [out, strf] = preprocWavelets3dVis(strf, visopt)
%
%  A visualizer of STRF preprocessed by preprocWavelets3d
%
% INPUT:
%          [strf] = strf model that was preprocessed by preprocWavelets3d
%        [visopt] = structure of options for visualizer
%      .vis_ssize = Number of spatial pixels to visualize
%                   (default: 40)
% .show_st_panels = A vector to specify whether or not to show space-time RFs.
%                   Each elements in the vector represents
%                   [excitatory suppressive linear gaussian_envelope]
%                   (default: [0 0 0 0]);
%     .show_bests = A flag to determine whether or not showing best 10 wavelets
%                   (default: 0)
%       .show_fig = A flag to determine whether or not to show figures. If not set,
%                   it will not show any figure and just return space-time matrix.
%
% OUTPUT:
%           [out] = a structure that contains space-time matrix.
%          [strf] = a strf structure. The .valid_w_index may be modified.
%
if ~exist('visopt', 'var'), visopt = []; end

vis_ssize = 40;
if isfield(visopt, 'vis_ssize')
	vis_ssize = visopt.vis_ssize;
end

moviereps = 10;
if isfield(visopt, 'moviereps')
	moviereps = visopt.moviereps;
end


if isfield(strf, 'physics')
	physics = strf.physics;
else
	physics = [];
end

vis_tsize = strf.params.tsize;

delays = strf.delays;
params = strf.params;
w = squeeze(strf.w1);

if isfield(params, 'normalize') & params.normalize
	for ii=1:size(w, 2)
		w(:,ii) = w(:,ii)./params.stds';
	end
end

numdelays = length(strf.delays);

valid_w_index = find(sum(abs(w),2));

%%% make Gabor wavelets
params.show_or_preprocess = 0;
params.valid_w_index = valid_w_index;
dummy = zeros(vis_ssize, vis_ssize, vis_tsize);
[gaborbank thisparams] = preprocWavelets3d(dummy, params);
gparams = thisparams.gaborparams;

gaborsize = size(gaborbank);
vis_t_size = gaborsize(3)+max(delays)  - min(delays);
vis_t_start = min(delays);

exs.ind = find(w(:)>0);
sps.ind = find(w(:)<0);

exs.wvalue = w(exs.ind);
sps.wvalue = w(sps.ind);

[exs.wvalue si] = sort(exs.wvalue, 'descend');
exs.ind = exs.ind(si);

[sps.wvalue si] = sort(sps.wvalue,'ascend');
sps.ind = sps.ind(si);

[exs.wch exs.wdelay] = ind2sub(size(w), exs.ind);
[sps.wch sps.wdelay] = ind2sub(size(w), sps.ind);

exs.wdelay = delays(exs.wdelay);
sps.wdelay = delays(sps.wdelay);

exs.gabors = zeros([gaborsize(1:3) length(exs.ind)], 'single');
for ii=1:length(exs.ind)
	thisind = find(valid_w_index == exs.wch(ii));
	exs.gabors(:,:,:,ii) = gaborbank(:,:,:,thisind);
end

sps.gabors = zeros([gaborsize(1:3) length(sps.ind)], 'single');
for ii=1:length(sps.ind)
	thisind = find(valid_w_index == sps.wch(ii));
	sps.gabors(:,:,:,ii) = gaborbank(:,:,:,thisind);
end

bestgabors_ex = zeros([gaborsize(1:3) 10]);
bestgabors_sp = zeros([gaborsize(1:3) 10]);

thisnum = min([10 length(exs.ind)]);
bestgabors_ex(:,:,:,1:thisnum) = exs.gabors(:,:,:,1:thisnum);
thisnum = min([10 length(sps.ind)]);
bestgabors_sp(:,:,:,1:thisnum) = sps.gabors(:,:,:,1:thisnum);

im_ex = zeros([gaborsize(1:2) vis_t_size]);
im_sp = im_ex;
im_linear = im_ex;
im_ex_gauss = im_ex;
im_sp_gauss = im_ex;


%% weight sum
for ex_or_sp = 1:2
	if ex_or_sp == 1
		exsp = exs;
		thisim = im_ex;
		thisim_g = im_ex_gauss;
	else
		exsp = sps;
		thisim = im_sp;
		thisim_g = im_sp_gauss;
	end
	for ii=1:length(exsp.ind)
		thisgabor = exsp.gabors(:,:,:,ii);
		thisgparam = gparams(:,exsp.wch(ii));
		thisdelays = (1:gaborsize(3)) + (exsp.wdelay(ii) - vis_t_start);
		switch thisgparam(8)
			case {0,3,4,5,6}  % spectra or half-rectified channels
				thisim(:,:,thisdelays) = thisim(:,:,thisdelays) + thisgabor*exsp.wvalue(ii);
			case {1,2} % linear channels
				im_linear(:,:,thisdelays) = im_linear(:,:,thisdelays) + thisgabor*exsp.wvalue(ii);
		end
		%% calculate gaussian envelopes
		thisgparam(4) = 0;  % zero spatial frequency
		thisgparam(5) = 0;  % zero temporal frequency
		[dummy thisgauss] = make3dgabor([vis_ssize vis_ssize vis_tsize], thisgparam);
		thisim_g(:,:,thisdelays) = thisim_g(:,:,thisdelays) + abs(thisgauss*exsp.wvalue(ii));
	end
	if ex_or_sp == 1
		im_ex = thisim;
		im_ex_gauss = thisim_g;
	else
		im_sp = thisim;
		im_sp_gauss = thisim_g;
	end

end

%% inverse time
im_ex = flipdim(im_ex,3);
im_sp = flipdim(im_sp,3);
im_linear = flipdim(im_linear,3);
im_ex_gauss = flipdim(im_ex_gauss,3);
im_sp_gauss = flipdim(im_sp_gauss,3);
bestgabors_ex = flipdim(bestgabors_ex,3);
bestgabors_sp = flipdim(bestgabors_sp,3);


%% space and time integration
g_ex_space = flipud(mean(im_ex_gauss,3));
g_sp_space = flipud(mean(im_sp_gauss,3));
g_space = sum(cat(3,g_ex_space, g_sp_space), 3);
g_ex_t = squeeze(mean(mean(im_ex_gauss,1),2));
g_sp_t = squeeze(mean(mean(im_sp_gauss,1),2));
g_t = sum([g_ex_t g_sp_t],2);

out.im_ex = im_ex;
out.im_sp = im_sp;
out.im_linear = im_linear;
out.im_ex_gauss = im_ex_gauss;
out.im_sp_gauss = im_sp_gauss;
out.space = cat(3, g_space, g_ex_space, g_sp_space);
out.time = [g_t g_ex_t -g_sp_t];


if isfield(visopt, 'figshow') & ~figshow
	return;
end

%--------------------------
% show images for different delays
%--------------------------

ShowParam.imdim = 3;
ShowParam.axisimage = 1;

maxim_ex = max([10e-10; abs(im_ex(:))]);
maxim_sp = max([10e-10; abs(im_sp(:))]);
maxim_linear = max([10e-10; abs(im_linear(:))]);

if isfield(visopt, 'show_st_panels') & visopt.show_st_panels(1)
	ShowParam.clim =[-maxim_ex maxim_ex]; 
	ShowParam.title = 'excitatory';
	figure(1);
	ndimages(im_ex, ShowParam);
end

if isfield(visopt, 'show_st_panels') & visopt.show_st_panels(2)
	ShowParam.clim =[-maxim_sp maxim_sp]; 
	ShowParam.title = 'suppressive';
	figure(2);
	ndimages(im_sp, ShowParam);
end

if isfield(visopt, 'show_st_panels') & visopt.show_st_panels(3)
	ShowParam.clim =[-maxim_linear maxim_linear]; 
	ShowParam.title = 'linear';
	figure(3);
	ndimages(im_linear, ShowParam);
end


if isfield(visopt, 'show_st_panels') & visopt.show_st_panels(4)
    ShowParam.title = 'excitatory';
	figure(4);
	ndimages(im_ex_gauss, ShowParam);
end

if isfield(visopt, 'show_st_panels') & visopt.show_st_panels(5)
    ShowParam.title = 'suppressive';
	figure(5);
	ndimages(im_sp_gauss, ShowParam);
end

%%% display space- and time-integrated plots
tt = -ceil(gaborsize(3)/2) + vis_t_start + (1:vis_t_size);
tt = fliplr(-tt);
if isfield(physics, 'binwidth_t')
	tt = tt*physics.binwidth_t;
end
tunit = '';
if isfield(physics, 'unit_t'), tunit = ['(' physics.unit_t ')']; end

tx = linspace(-0.5, 0.5, vis_ssize);
ty = linspace(-0.5, 0.5, vis_ssize);
if isfield(physics, 'binwidth_s')
	tx = tx*physics.binwidth_s*vis_ssize;
	ty = ty*physics.binwidth_s*vis_ssize;
end
if isfield(physics, 'centerx'), tx = tx + physics.centerx; end
if isfield(physics, 'centery'), ty = ty + physics.centery; end

sunit = '';
if isfield(physics, 'unit_s'), sunit = ['(' physics.unit_s ')']; end


%  spgauss = out.space;
%  spgauss(:,:,1) = spgauss(:,:,3);
%  spgauss(:,:,3) = spgauss(:,:,2);
%  spgauss(:,:,2) = spgauss(:,:,3)*0;
%  spgauss = spgauss./max(spgauss(:));


spg = out.space;
spg = spg./max(spg(:));

spgauss(:,:,1) = 1 - spg(:,:,2);
spgauss(:,:,3) = 1 - spg(:,:,3);
spgauss(:,:,2) = spgauss(:,:,3)*0 + 1 - sum(spg(:,:,[2 3]),3);
spgauss(spgauss<0) = 0;
spgauss = spgauss./max(spgauss(:));

figure(6); clf;
subplot(1,2,1);
imagesc(tx, ty, spgauss);
axis xy; axis image;
xlabel(['x ' sunit]);
ylabel(['y ' sunit]);

subplot(1,2,2);
h = plot(tt, out.time, '.'); hold on
h2 = plot(tt, out.time, '-');
set(h(1), 'color', [0 0 0]); set(h2(1), 'color', [0 0 0]);
set(h(2), 'color', [0 0 1]); set(h2(2), 'color', [0 0 1]);
set(h(3), 'color', [1 0 0]); set(h2(3), 'color', [1 0 0]);
xlabel(['time ' tunit ]);
xlim([tt(1) tt(end)+0.000001]);


%% show kernels in a Fourier domain
if isfield(visopt, 'show_fft') & visopt.show_fft
	Show_FFT3d(w, gaborbank, valid_w_index);
end


%--------------------------
% show movie
%--------------------------

totalmax = max([maxim_ex maxim_sp maxim_linear]);

im_exsp = cat(2, im_ex, ones(size(im_ex,1), 5, size(im_ex, 3)), im_sp);

ShowParam.clim = [-1 1];
ShowParam.imdim = 6;
figure(7); clf;

for ii=1:vis_t_size*moviereps
	tic;
	t = mod(ii-1, gaborsize(3))+1;
	t_kernel = mod((ii-1), vis_t_size)+1;

	figure(7);
    subplot(2,2,1), imagesc(im_ex(:,:,t_kernel), [-totalmax totalmax]);
	axis off; axis square;
	title(sprintf('ex: %.3f', maxim_ex/totalmax));

	subplot(2,2,2), imagesc(im_sp(:,:,t_kernel), [-totalmax totalmax]);
	axis off; axis square;
	title(sprintf('sp: %.3f', maxim_sp/totalmax));

	subplot(2,2,3), imagesc(im_linear(:,:,t_kernel), [-totalmax totalmax]);
	axis off; axis square;
	title(sprintf('linear: %.3f', maxim_linear/totalmax));

	colormap gray;

	if isfield(visopt, 'show_bests') & visopt.show_bests
		ShowParam.show = 0;
		thisim = squeeze(bestgabors_ex(:,:,t,:));
		thisim = reshape(thisim, [size(thisim,1) size(thisim,2) 5 2]);
		thisim_ex = ndimages(thisim, ShowParam);
		thisim = squeeze(bestgabors_sp(:,:,t,:));
		thisim = reshape(thisim, [size(thisim,1) size(thisim,2) 5 2]);
		thisim_sp = ndimages(thisim, ShowParam);
	
		figure(7);
		ShowParam.show = 1;
		ndimages(cat(6, thisim_ex, thisim_sp), ShowParam);
		colormap gray;
	end

	drawnow;
	while toc<0.05
		drawnow;
	end
end

strf.params = params;

