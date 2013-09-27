X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covSum, {{@covNoise}, {@covSEiso}, {@covChangeBurstTanh, {{@covSum, {{@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}}}, {@covSum, {{@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}}}}}}};
kernel_params = [ -5.89196 14.43781 7.632901 1680.160148 -2.125895 4.337788 -0.770772 -1.275594 -2.017506 5.992869 1869.885546 0.84583 -3.795428 2.531315 7.196563 2082.784684 5.180441 -7.949922 -1.463931 2.302064 2011.831444 7.525079 -3.202374 0.442442 3.632285 1259.455679 ];
kernel_family_list = { {@covNoise},{@covSEiso},{@covChangeBurstTanh, {{@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covZero}}},{@covChangeBurstTanh, {{@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covZero}}},{@covChangeBurstTanh, {{@covZero}, {@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}}},{@covChangeBurstTanh, {{@covZero}, {@covProd, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}}}}} };
kernel_params_list = { [ -5.89196 ],[ 14.43781 7.632901 ],[ 1680.160148 -2.125895 4.337788 -0.770772 -1.275594 -2.017506 5.992869 1869.885546 ],[ 1680.160148 -2.125895 4.337788 0.84583 -3.795428 2.531315 7.196563 2082.784684 ],[ 1680.160148 -2.125895 4.337788 5.180441 -7.949922 -1.463931 2.302064 2011.831444 ],[ 1680.160148 -2.125895 4.337788 7.525079 -3.202374 0.442442 3.632285 1259.455679 ] };
noise = [-inf];
figname = '/scratch/home/Research/GPs/gpss-research/analyses/2013-09-26/figures/02-solar/02-solar';
latex_names = { ' WN','SE','CBT\left( SE \times Lin , NIL \right)','CBT\left( SE \times Lin , NIL \right)','CBT\left( NIL , SE \times Lin \right)','CBT\left( NIL , SE \times Lin \right) ' };
full_kernel_name = { '\left( WN + SE + CBT\left( \left( SE \times Lin + SE \times Lin \right) , \left( SE \times Lin + SE \times Lin \right) \right) \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, kernel_family, kernel_params, kernel_family_list, kernel_params_list, noise, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)
