load '/var/folders/ss/chbkrr6d6rdgn0w3848wmh040000gn/T/tmpWxz6E1.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/gpml'));
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/matlab'));

mean_family = {@meanZero};
mean_params = [  ];
kernel_family = {@covSum, {{@covSEiso}, {@covSEiso}, {@covSEiso}}};
kernel_params = [ -1.89161214047 3.65727340956 -2.72225953721 3.00777665568 2.78412165003 6.05459267497 ];
lik_family = {@likDelta};
lik_params = [  ];
kernel_family_list = { {@covSEiso},{@covSEiso},{@covSEiso} };
kernel_params_list = { [ -1.89161214047 3.65727340956 ],[ -2.72225953721 3.00777665568 ],[ 2.78412165003 6.05459267497 ] };
inference = '@infDelta';
figname = '/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/analyses/debug/01-airline/01-airline';
latex_names = { ' {\sc SE}','{\sc SE}','{\sc SE} ' };
full_kernel_name = { '\left( {\sc SE} + {\sc SE} + {\sc SE} \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, mean_family, mean_params, kernel_family, kernel_params, kernel_family_list, kernel_params_list, lik_family, lik_params, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)