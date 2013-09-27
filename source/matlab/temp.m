load '/tmp/tmpv3QnIz.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covSum, {{@covSEiso}, {@covSum, {@covConst, @covLINscaleshift}}, {@covProd, {{@covNoise}, {@covSum, {@covConst, @covLINscaleshift}}}}, {@covProd, {{@covSEiso}, {@covFourier}, {@covSum, {@covConst, @covLINscaleshift}}}}}};
kernel_params = [ -0.421213 2.534102 -0.628703 -3.794036 1946.100163 2.083379 -3.311197 2.651618 1943.871657 3.151642 -0.547919 -0.31074 0.002349 1.105826 0.077946 -1.044053 1945.527471 ];
kernel_family_list = { {@covSEiso},{@covSum, {@covConst, @covLINscaleshift}},{@covProd, {{@covNoise}, {@covSum, {@covConst, @covLINscaleshift}}}},{@covProd, {{@covSEiso}, {@covFourier}, {@covSum, {@covConst, @covLINscaleshift}}}} };
kernel_params_list = { [ -0.421213 2.534102 ],[ -0.628703 -3.794036 1946.100163 ],[ 2.083379 -3.311197 2.651618 1943.871657 ],[ 3.151642 -0.547919 -0.31074 0.002349 1.105826 0.077946 -1.044053 1945.527471 ] };
noise = [-inf];
figname = '/scratch/home/Research/GPs/gpss-research/analyses/2013-09-26/figures/01-airline/01-airline';
latex_names = { ' SE','Lin','WN \times Lin','SE \times Fourier \times Lin ' };
full_kernel_name = { '\left( SE + Lin + WN \times Lin + SE \times Fourier \times Lin \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, kernel_family, kernel_params, kernel_family_list, kernel_params_list, noise, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)
