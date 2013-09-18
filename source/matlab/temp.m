load '/tmp/tmpPH1WLa.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covProd, {{@covMaterniso, 5}, {@covSum, {{@covProd, {{@covPeriodicCentre}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covConst}, {@covSum, {{@covPeriodicCentre}, {@covConst}}}}}}}}};
kernel_params = [ 3.217148 6.007081 -0.423756 0.001793 -5.933056 0.950227 -1.562857 1946.454477 0.807736 0.196506 1.447892 -4.044675 -0.761862408406 ];
kernel_family_list = { {@covProd, {{@covMaterniso, 5}, {@covProd, {{@covPeriodicCentre}, {@covSum, {@covConst, @covLINscaleshift}}}}}},{@covProd, {{@covMaterniso, 5}, {@covProd, {{@covConst}, {@covPeriodicCentre}}}}},{@covProd, {{@covMaterniso, 5}, {@covProd, {{@covConst}, {@covConst}}}}} };
kernel_params_list = { [ 3.217148 6.007081 -0.423756 0.001793 -5.933056 0.950227 -1.562857 1946.454477 ],[ 3.217148 6.007081 0.807736 0.196506 1.447892 -4.044675 ],[ 3.217148 6.007081 0.807736 -0.761862408406 ] };
noise = [1.84979287];
figname = '/scratch/home/Research/GPs/gpss-research/analyses/2013-09-17-summary/figures/01-airline/01-airline';
latex_names = { ' MT5 \times CenPer \times Lin','MT5 \times CS \times CenPer','MT5 \times CS \times CS ' };
full_kernel_name = { 'MT5 \times \left( CenPer \times Lin + CS \times \left( CenPer + CS \right) \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, kernel_family, kernel_params, kernel_family_list, kernel_params_list, noise, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)