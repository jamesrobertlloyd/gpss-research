load '/var/folders/ss/chbkrr6d6rdgn0w3848wmh040000gn/T/tmp0wpsoA.mat'  % Load the data, it should contain X and y.
X = double(X);
y = double(y);

addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/gpml'));
addpath(genpath('/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/source/matlab'));

kernel_family = {@covSum, {{@covNoise}, {@covConst}, {@covChangeBurstTanh, {{@covChangeBurstTanh, {{@covSum, {{@covSEiso}, {@covSEiso}, {@covChangePointTanh, {{@covSEiso}, {@covProd, {{@covSEiso}, {@covLINscaleshift}}}}}, {@covProd, {{@covSEiso}, {@covFourier}}}}}, {@covConst}}}, {@covConst}}}}};
kernel_params = [ -5.807507 7.216893 1679.078644 -1.855858 4.233405 1713.568641 -1.253981 2.520475 0.164456 -2.250492 3.087954 -1.337912 1751.055402 -2.757152 -1.337813 -2.127402 -1.357407 -2.780108 4.55908 1843.489208 3.502061 -1.273383 1.9888 2.379065 -0.479717 -1.000562 -1.025533 ];
kernel_family_list = { {@covNoise},{@covConst},{@covChangeBurstTanh, {{@covChangeBurstTanh, {{@covSEiso}, {@covZero}}}, {@covZero}}},{@covChangeBurstTanh, {{@covChangeBurstTanh, {{@covSEiso}, {@covZero}}}, {@covZero}}},{@covChangeBurstTanh, {{@covChangeBurstTanh, {{@covChangePointTanh, {{@covSEiso}, {@covZero}}}, {@covZero}}}, {@covZero}}},{@covChangeBurstTanh, {{@covChangeBurstTanh, {{@covChangePointTanh, {{@covZero}, {@covProd, {{@covSEiso}, {@covLINscaleshift}}}}}, {@covZero}}}, {@covZero}}},{@covChangeBurstTanh, {{@covChangeBurstTanh, {{@covProd, {{@covSEiso}, {@covFourier}}}, {@covZero}}}, {@covZero}}},{@covChangeBurstTanh, {{@covChangeBurstTanh, {{@covZero}, {@covConst}}}, {@covZero}}},{@covChangeBurstTanh, {{@covZero}, {@covConst}}} };
kernel_params_list = { [ -5.807507 ],[ 7.216893 ],[ 1679.078644 -1.855858 4.233405 1713.568641 -1.253981 2.520475 0.164456 -2.250492 ],[ 1679.078644 -1.855858 4.233405 1713.568641 -1.253981 2.520475 3.087954 -1.337912 ],[ 1679.078644 -1.855858 4.233405 1713.568641 -1.253981 2.520475 1751.055402 -2.757152 -1.337813 -2.127402 ],[ 1679.078644 -1.855858 4.233405 1713.568641 -1.253981 2.520475 1751.055402 -2.757152 -1.357407 -2.780108 4.55908 1843.489208 ],[ 1679.078644 -1.855858 4.233405 1713.568641 -1.253981 2.520475 3.502061 -1.273383 1.9888 2.379065 -0.479717 ],[ 1679.078644 -1.855858 4.233405 1713.568641 -1.253981 2.520475 -1.000562 ],[ 1679.078644 -1.855858 4.233405 -1.025533 ] };
noise = [-inf];
figname = '/Users/JamesLloyd/Documents/Cambridge/MachineLearning/Research/GPs/gpss-research/analyses/2013-10-01-pure-lin/figures/02-solar/02-solar';
latex_names = { ' WN','CS','CBT\left( CBT\left( SE , NIL \right) , NIL \right)','CBT\left( CBT\left( SE , NIL \right) , NIL \right)','CBT\left( CBT\left( CPT\left( SE , NIL \right) , NIL \right) , NIL \right)','CBT\left( CBT\left( CPT\left( NIL , SE \times PureLin \right) , NIL \right) , NIL \right)','CBT\left( CBT\left( SE \times Fourier , NIL \right) , NIL \right)','CBT\left( CBT\left( NIL , CS \right) , NIL \right)','CBT\left( NIL , CS \right) ' };
full_kernel_name = { '\left( WN + CS + CBT\left( CBT\left( \left( SE + SE + CPT\left( SE , SE \times PureLin \right) + SE \times Fourier \right) , CS \right) , CS \right) \right)' };
X_mean = 0.000000;
X_scale = 1.000000;
y_mean = 0.000000;
y_scale = 1.000000;

plot_decomp(X, y, kernel_family, kernel_params, kernel_family_list, kernel_params_list, noise, figname, latex_names, full_kernel_name, X_mean, X_scale, y_mean, y_scale)