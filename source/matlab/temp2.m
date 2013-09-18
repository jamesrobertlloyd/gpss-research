load '/tmp/tmpj10h5d.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covProd, {{@covSum, {{@covPeriodicCentre}, {@covConst}}}, {@covSum, {{@covMaterniso, 5}, {@covProd, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covSum, {{@covSum, {@covConst, @covLINscaleshift}}, {@covProd, {{@covMaterniso, 5}, {@covSum, {{@covPeriodicCentre}, {@covConst}}}}}}}}}}}}};
kernel_params = [ 2.314972 -0.008012 -0.573749 0.241566722026 0.745465 4.607419 4.494469 -2.520251 5.866723 5.492876 -1.617521 -1.360697 1982.38242 4.922308 4.333409 -1.06271 -2.56839 -3.502612 0.925316 ];
kernel_family_list = { {@covProd, {{@covPeriodicCentre}, {@covMaterniso, 5}}},{@covProd, {{@covPeriodicCentre}, {@covProd, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covSum, {@covConst, @covLINscaleshift}}}}}},{@covProd, {{@covPeriodicCentre}, {@covProd, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covProd, {{@covMaterniso, 5}, {@covPeriodicCentre}}}}}}},{@covProd, {{@covPeriodicCentre}, {@covProd, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covProd, {{@covMaterniso, 5}, {@covConst}}}}}}},{@covProd, {{@covConst}, {@covMaterniso, 5}}},{@covProd, {{@covConst}, {@covProd, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covSum, {@covConst, @covLINscaleshift}}}}}},{@covProd, {{@covConst}, {@covProd, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covProd, {{@covMaterniso, 5}, {@covPeriodicCentre}}}}}}},{@covProd, {{@covConst}, {@covProd, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covProd, {{@covMaterniso, 5}, {@covConst}}}}}}} };
kernel_params_list = { [ 2.314972 -0.008012 -0.573749 0.745465 4.607419 ],[ 2.314972 -0.008012 -0.573749 4.494469 -2.520251 5.866723 5.492876 -1.617521 -1.360697 1982.38242 ],[ 2.314972 -0.008012 -0.573749 4.494469 -2.520251 5.866723 5.492876 4.922308 4.333409 -1.06271 -2.56839 -3.502612 ],[ 2.314972 -0.008012 -0.573749 4.494469 -2.520251 5.866723 5.492876 4.922308 4.333409 0.925316 ],[ 0.241566722026 0.745465 4.607419 ],[ 0.241566722026 4.494469 -2.520251 5.866723 5.492876 -1.617521 -1.360697 1982.38242 ],[ 0.241566722026 4.494469 -2.520251 5.866723 5.492876 4.922308 4.333409 -1.06271 -2.56839 -3.502612 ],[ 0.241566722026 4.494469 -2.520251 5.866723 5.492876 4.922308 4.333409 0.925316 ] };
noise = [3.06507713];
figname = '/scratch/home/Research/GPs/gpss-research/analyses/2013-09-17-summary/figures/number-of-daily-births-in-quebec/number-of-daily-births-in-quebec';
latex_names = { ' CenPer \times MT5','CenPer \times MT5 \times MT5 \times Lin','CenPer \times MT5 \times MT5 \times MT5 \times CenPer','CenPer \times MT5 \times MT5 \times MT5 \times CS','CS \times MT5','CS \times MT5 \times MT5 \times Lin','CS \times MT5 \times MT5 \times MT5 \times CenPer','CS \times MT5 \times MT5 \times MT5 \times CS ' };
full_kernel_name = { '\left( CenPer + CS \right) \times \left( MT5 + MT5 \times MT5 \times \left( Lin + MT5 \times \left( CenPer + CS \right) \right) \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, kernel_family, kernel_params, kernel_family_list, kernel_params_list, noise, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)
