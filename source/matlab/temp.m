load '/var/folders/ss/chbkrr6d6rdgn0w3848wmh040000gn/T/tmpbDtuB2.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/gpml'));
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covSum, {{@covSEiso}, {@covLINscaleshift}, {@covProd, {{@covNoise}, {@covLINscaleshift}}}, {@covProd, {{@covSEiso}, {@covFourier}, {@covLINscaleshift}}}}};
kernel_params = [ -0.38573 2.627733 -3.445314 1946.074222 -2.485397 -1.840854 1942.998574 3.112678 0.745437 -0.332958 0.002295 1.078193 0.296651 1945.488578 ];
kernel_family_list = { {@covSEiso},{@covLINscaleshift},{@covProd, {{@covNoise}, {@covLINscaleshift}}},{@covProd, {{@covSEiso}, {@covFourier}, {@covLINscaleshift}}} };
kernel_params_list = { [ -0.38573 2.627733 ],[ -3.445314 1946.074222 ],[ -2.485397 -1.840854 1942.998574 ],[ 3.112678 0.745437 -0.332958 0.002295 1.078193 0.296651 1945.488578 ] };
noise = [-inf];
figname = '/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/analyses/2013-10-01-pure-lin/figures/01-airline/01-airline';
latex_names = { ' SE','PureLin','WN \times PureLin','SE \times Fourier \times PureLin ' };
full_kernel_name = { '\left( SE + PureLin + WN \times PureLin + SE \times Fourier \times PureLin \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, kernel_family, kernel_params, kernel_family_list, kernel_params_list, noise, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)