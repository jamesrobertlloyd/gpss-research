load '/tmp/tmpbpzVcI.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

mean_family = {@meanZero};
mean_params = [  ];
kernel_family = {@covSum, {{@covConst}, {@covChangeWindowMultiD, {1, {@covSum, {{@covSEiso}, {@covSEiso}, {@covProd, {{@covNoise}, {@covLinear}}}, {@covProd, {{@covNoise}, {@covLinear}}}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}}}}}, {@covSum, {{@covNoise}, {@covConst}}}}}}};
kernel_params = [ 7.47963232653 1679.42006046 -2.04572905004 4.29176527936 0.47177787047 -2.13632411486 3.13981937815 -1.41109409712 -1.68843730228 -5.7015024374 1837.35420854 -3.13078956004 -5.0654645828 1952.8962968 3.60703992141 -1.94343767603 3.82783498823 2.37551297407 0.249203973137 -5.85037372055 -0.628268751886 ];
lik_family = {@likDelta};
lik_params = [  ];
kernel_family_list = { {@covConst},{@covChangeWindowMultiD, {1, {@covProd, {{@covNoise}, {@covLinear}}}, {@covZero}}},{@covChangeWindowMultiD, {1, {@covProd, {{@covNoise}, {@covLinear}}}, {@covZero}}},{@covChangeWindowMultiD, {1, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}}}, {@covZero}}},{@covChangeWindowMultiD, {1, {@covSEiso}, {@covZero}}},{@covChangeWindowMultiD, {1, {@covSEiso}, {@covZero}}},{@covChangeWindowMultiD, {1, {@covZero}, {@covConst}}},{@covChangeWindowMultiD, {1, {@covZero}, {@covNoise}}} };
kernel_params_list = { [ 7.47963232653 ],[ 1679.42006046 -2.04572905004 4.29176527936 -1.68843730228 -5.7015024374 1837.35420854 ],[ 1679.42006046 -2.04572905004 4.29176527936 -3.13078956004 -5.0654645828 1952.8962968 ],[ 1679.42006046 -2.04572905004 4.29176527936 3.60703992141 -1.94343767603 3.82783498823 2.37551297407 0.249203973137 ],[ 1679.42006046 -2.04572905004 4.29176527936 0.47177787047 -2.13632411486 ],[ 1679.42006046 -2.04572905004 4.29176527936 3.13981937815 -1.41109409712 ],[ 1679.42006046 -2.04572905004 4.29176527936 -0.628268751886 ],[ 1679.42006046 -2.04572905004 4.29176527936 -5.85037372055 ] };
figname = '/scratch/home/Research/GPs/gpss-research/analyses/2014-05-19-GPSS-add-mmd/02-solar/02-solar';
idx = [ 1 7 6 4 5 2 3 8 ];
plot = true;

checking_stats(X, y, mean_family, mean_params, kernel_family, kernel_params, kernel_family_list, kernel_params_list, lik_family, lik_params, figname, idx, plot)
