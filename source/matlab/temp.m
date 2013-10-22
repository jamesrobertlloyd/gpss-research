load '/tmp/tmpG92bbn.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covSum, {{@covSEiso}, {@covProd, {{@covNoise}, {@covLINscaleshift}}}, {@covProd, {{@covSEiso}, {@covFourier}, {@covLINscaleshift}}}, {@covProd, {{@covFourier}, {@covLINscaleshift}}}}};
kernel_params = [ 2.660738 5.9716 -3.309317 -2.619361 1942.378721 3.119187 -0.171976 -0.321892 0.002416 4.971314 3.270519 1945.19997 -0.092149 1.450734 3.809601 3.967303 1945.781619 ];
kernel_family_list = { {@covSEiso},{@covProd, {{@covNoise}, {@covLINscaleshift}}},{@covProd, {{@covSEiso}, {@covFourier}, {@covLINscaleshift}}},{@covProd, {{@covFourier}, {@covLINscaleshift}}} };
kernel_params_list = { [ 2.660738 5.9716 ],[ -3.309317 -2.619361 1942.378721 ],[ 3.119187 -0.171976 -0.321892 0.002416 4.971314 3.270519 1945.19997 ],[ -0.092149 1.450734 3.809601 3.967303 1945.781619 ] };
noise = [-inf];
figname = '/scratch/home/Research/GPs/gpss-research/analyses/2013-10-22-summary/figures/01-airline/01-airline';
latex_names = { ' SE','WN \times PureLin','SE \times Fourier \times PureLin','Fourier \times PureLin ' };
full_kernel_name = { '\left( SE + WN \times PureLin + SE \times Fourier \times PureLin + Fourier \times PureLin \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, kernel_family, kernel_params, kernel_family_list, kernel_params_list, noise, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)
