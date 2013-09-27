X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covSum, {{@covNoise}, {@covSEiso}, {@covChangeBurstTanh, {{@covSum, {{@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}}}, {@covSum, {{@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}}}}}}};
kernel_params = [ -5.89196 14.43781 7.632901 1680.160148 -2.125895 4.337788 -0.770772 -1.275594 -2.017506 5.992869 1869.885546 0.84583 -3.795428 2.531315 7.196563 2082.784684 5.180441 -7.949922 -1.463931 2.302064 2011.831444 7.525079 -3.202374 0.442442 3.632285 1259.455679 ];
noise = [-inf];

cov_func = {@covSum, {{@covNoise}, {@covSEiso}, {@covChangeBurstTanh, {{@covSum, {{@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covFourier}, {@covSEiso}}}, {@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}}}, {@covSum, {{@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}}}}}}};
hyp.cov = [ -5.89196 14.43781 7.632901 1680.160148 -2.125895 4.337788 -0.770772 -1.275594 -2.017506 5.992869 1869.885546 0, log(11), 0, 2.84583 -3.795428 2.84583 -3.795428 2.531315 7.196563 2082.784684 5.180441 -7.949922 -1.463931 2.302064 2011.831444 7.525079 -3.202374 0.442442 3.632285 1259.455679 ];

% cov_func = kernel_family;
% hyp.cov = kernel_params;

hyp.lik  = [];
lik_func = @likDelta;

hyp.mean = [];
mean_func = @meanZero;

hyp_opt = minimize(hyp, @gp, -100, @infDelta, mean_func, cov_func, lik_func, X, y);