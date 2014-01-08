load '/tmp/tmpKLuxjQ.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

mean_family = {@meanZero};
mean_params = [  ];
kernel_family = {@covSum, {{@covNoise}, {@covSEiso}, {@covSEiso}, {@covSEiso}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}}}}};
kernel_params = [ -1.66464384598 -0.0440676317821 -0.551114423072 -1.93529780802 -1.67883681466 3.91656713552 3.63287360892 5.35253861999 -0.260252196287 0.35486612623 -0.000357361590088 1.02805636032 ];
lik_family = {@likDelta};
lik_params = [  ];
kernel_family_list = { {@covNoise},{@covSEiso},{@covSEiso},{@covSEiso},{@covProd, {{@covSEiso}, {@covPeriodicNoDC}}} };
kernel_params_list = { [ -1.66464384598 ],[ -0.0440676317821 -0.551114423072 ],[ -1.93529780802 -1.67883681466 ],[ 3.91656713552 3.63287360892 ],[ 5.35253861999 -0.260252196287 0.35486612623 -0.000357361590088 1.02805636032 ] };
figname = '/scratch/home/Research/GPs/gpss-research/analyses/2014-01-08-collated/03-mauna/03-mauna';
idx = [ 4 5 2 1 3 ];
plot = true;

checking_stats(X, y, mean_family, mean_params, kernel_family, kernel_params, kernel_family_list, kernel_params_list, lik_family, lik_params, figname, idx, plot);