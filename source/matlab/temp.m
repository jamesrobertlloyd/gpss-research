load '/var/folders/ss/chbkrr6d6rdgn0w3848wmh040000gn/T/tmpwfpt5Q.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/gpml'));
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covProd, {{@covMaterniso, 5}, {@covSum, {{@covProd, {{@covPeriodicCentre}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covConst}, {@covSum, {{@covPeriodicCentre}, {@covConst}}}}}}}}};
kernel_params = [ 3.217148 6.007081 -0.423756 0.001793 -5.933056 0.950227 -1.562857 1946.454477 0.807736 0.196506 1.447892 -4.044675 -0.761862408406 ];
kernel_family_list = { {@covProd, {{@covPeriodicCentre}, {@covSum, {@covConst, @covLINscaleshift}}, {@covMaterniso, 5}}},{@covProd, {{@covPeriodicCentre}, {@covConst}, {@covMaterniso, 5}}},{@covProd, {{@covConst}, {@covConst}, {@covMaterniso, 5}}} };
kernel_params_list = { [ -0.423756 0.001793 -5.933056 0.950227 -1.562857 1946.454477 3.217148 6.007081 ],[ 0.196506 1.447892 -4.044675 0.807736 3.217148 6.007081 ],[ -0.761862408406 0.807736 3.217148 6.007081 ] };
noise = [1.84979287];
figname = '/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/analyses/2013-09-18-human/figures/01-airline/01-airline';
latex_names = { ' CenPer \times Lin \times MT5','CenPer \times CS \times MT5','CS \times CS \times MT5 ' };
full_kernel_name = { 'MT5 \times \left( CenPer \times Lin + CS \times \left( CenPer + CS \right) \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, kernel_family, kernel_params, kernel_family_list, kernel_params_list, noise, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)