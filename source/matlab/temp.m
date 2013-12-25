load '/var/folders/ss/chbkrr6d6rdgn0w3848wmh040000gn/T/tmpSTyuXG.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/gpml'));
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/matlab'));

mean_family = {@meanZero};
mean_params = [  ];
kernel_family = {@covSum, {{@covNoise}, {@covSEiso}, {@covLinear}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}, {@covLinear}}}}};
kernel_params = [ 1.70386265331 -0.500063816447 2.49989084099 3.42242275086 1945.75848682 2.75990985139 3.79692041008 -0.403397379226 0.00274899616587 -1.95132739066 -0.380566371851 1945.59698616 ];
lik_family = {@likDelta};
lik_params = [  ];
kernel_family_list = { {@covNoise},{@covSEiso},{@covLinear},{@covProd, {{@covSEiso}, {@covPeriodicNoDC}, {@covLinear}}} };
kernel_params_list = { [ 1.70386265331 ],[ -0.500063816447 2.49989084099 ],[ 3.42242275086 1945.75848682 ],[ 2.75990985139 3.79692041008 -0.403397379226 0.00274899616587 -1.95132739066 -0.380566371851 1945.59698616 ] };
figname = '/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/analyses/2013-12-10-extrap-GPSS/figures/01-airline/01-airline';
idx = '[2 3 1 0]';

component_stats_and_plots(X, y, mean_family, mean_params, kernel_family, kernel_params, kernel_family_list, kernel_params_list, lik_family, lik_params, figname, idx)