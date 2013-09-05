
load '/tmp/tmpZqC4iJ.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('gpml/'));
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/matlab'));

plot_decomp_projective(X, y, {@covProd, {{@covSum, {@covConst, @covLINscaleshift}}, {@covSum, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covProd, {{@covPeriodic}, {@covSum, {{@covSEiso}, {@covConst}}}}}}}}}, [ -7.868232 -1.556342 1944.558458 -3.157494 -2.006594 0.161321 -0.896747 0.078534 0.002477 3.496241 3.705114 -2.419997 -2.527989 ], { {@covProd, {{@covSum, {@covConst, @covLINscaleshift}}, {@covMaterniso, 5}}},{@covProd, {{@covSum, {@covConst, @covLINscaleshift}}, {@covMaterniso, 5}}},{@covProd, {{@covSum, {@covConst, @covLINscaleshift}}, {@covProd, {{@covPeriodic}, {@covSEiso}}}}},{@covProd, {{@covSum, {@covConst, @covLINscaleshift}}, {@covProd, {{@covPeriodic}, {@covConst}}}}} }, { [ -7.868232 -1.556342 1944.558458 -3.157494 -2.006594 ],[ -7.868232 -1.556342 1944.558458 0.161321 -0.896747 ],[ -7.868232 -1.556342 1944.558458 0.078534 0.002477 3.496241 3.705114 -2.419997 ],[ -7.868232 -1.556342 1944.558458 0.078534 0.002477 3.496241 -2.527989 ] }, [-3.46137894], '/scratch/home/Research/GPs/gpss-research/analyses/2013-09-02-projective/figures/01-airline/01-airline', { ' Lin \times MT5','Lin \times MT5','Lin \times Per \times SE','Lin \times Per \times CS ' }, { 'Lin \times \left( MT5 + MT5 + Per \times \left( SE + CS \right) \right)' }, 0.000000, 1.000000, 0.000000, 1.000000)