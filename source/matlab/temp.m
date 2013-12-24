%%

rand('twister', 1809726962);
randn('state', 1809726962);

X = double(X)
y = double(y)

X_full = X;
y_full = y;

if true & (250 < size(X, 1))
    subset = randsample(size(X, 1), 250, false)
    X = X_full(subset,:);
    y = y_full(subset);
end

% Set up model.
meanfunc = {@meanZero}
hyp.mean = [  ]

covfunc = {@covChangePointMultiD, {1, {@covSum, {{@covNoise}, {@covChangeWindowMultiD, {1, {@covSum, {{@covSEiso}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}, {@covPeriodicNoDC}}}}}, {@covConst}}}}}, {@covNoise}}}
hyp.cov = [ 1688.18275358 -3.23579381547 -2.42172246393 1702.96490795 -1.25795850012 3.25396776726 9.78926180719 7.26092694985 3.49808224898 5.51701594419 -1.85676982564 4.79398098638 -5.43973154469 1.58781442467 3.1549955952 -1.50228630314 0.650179584169 7.37398966171 ]

likfunc = {@likDelta}
hyp.lik = [  ]

inference = @infDelta

% Optimise on subset
[hyp_opt, nlls] = minimize(hyp, @gp, -50, inference, meanfunc, covfunc, likfunc, X, y);

% Optimise on full data
if 10 > 0
    hyp_opt = minimize(hyp_opt, @gp, -10, inference, meanfunc, covfunc, likfunc, X_full, y_full);
end

% Evaluate the nll on the full data
best_nll = gp(hyp_opt, inference, meanfunc, covfunc, likfunc, X_full, y_full)

%%

rand('twister', 1282554292);
randn('state', 1282554292);

a='Load the data, it should contain X and y.'
load 'tmpMDlshP.mat'
X = double(X)
y = double(y)

X_full = X;
y_full = y;

if true & (250 < size(X, 1))
    subset = randsample(size(X, 1), 250, false)
    X = X_full(subset,:);
    y = y_full(subset);
end

% Load GPML
addpath(genpath('/users/jrl44/GPML'));

% Set up model.
meanfunc = {@meanZero}
hyp.mean = [  ]

covfunc = {@covChangeWindowMultiD, {1, {@covSum, {{@covNoise}, {@covConst}}}, {@covConst}}}
hyp.cov = [ 1688.18275358 -3.23579381547 2.19933455752 -2.41097122659 0.631435313908 7.07677174125 ]

likfunc = {@likDelta}
hyp.lik = [  ]

inference = @infDelta

% Optimise on subset
[hyp_opt, nlls] = minimize(hyp, @gp, -int32(250 * 3 / 3), inference, meanfunc, covfunc, likfunc, X, y);

% Optimise on full data
if 10 > 0
    hyp_opt = minimize(hyp_opt, @gp, -10, inference, meanfunc, covfunc, likfunc, X_full, y_full);
end

% Evaluate the nll on the full data
best_nll = gp(hyp_opt, inference, meanfunc, covfunc, likfunc, X_full, y_full)

save( '/scratch/tmprrDRxT.out', 'hyp_opt', 'best_nll', 'nlls');
% exit();

system('scp -o ConnectTimeout=300 -i /users/jrl44/.ssh/jrl44fear2sagarmatha /scratch/tmprrDRxT.out jrl44@sagarmatha:/tmp; rm /scratch/tmprrDRxT.out')

addpath(genpath('/users/jrl44/matlab'))

rand('twister', 1804674652);
randn('state', 1804674652);

a='Load the data, it should contain X and y.'
load 'tmpMDlshP.mat'
X = double(X)
y = double(y)

X_full = X;
y_full = y;

if true & (250 < size(X, 1))
    subset = randsample(size(X, 1), 250, false)
    X = X_full(subset,:);
    y = y_full(subset);
end

% Load GPML
addpath(genpath('/users/jrl44/GPML'));

% Set up model.
meanfunc = {@meanZero}
hyp.mean = [  ]

covfunc = {@covChangeWindowMultiD, {1, {@covSum, {{@covNoise}, {@covSEiso}, {@covProd, {{@covPeriodicNoDC}, {@covPeriodicNoDC}}}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}, {@covPeriodicNoDC}}}}}, {@covConst}}}
hyp.cov = [ 1688.18275358 -3.23579381547 2.19933455752 -2.50414818315 9.8213610959 7.39111430809 -1.10650914277 5.7020004081 -1.11021841494 -1.67825669961 4.71152394908 -5.44387202108 3.42140871457 5.50165280861 -1.89414823407 4.77904051548 -5.28179092623 1.43571400277 3.13516905328 -1.36378161432 7.37552538282 ]

likfunc = {@likDelta}
hyp.lik = [  ]

inference = @infDelta

% Optimise on subset
[hyp_opt, nlls] = minimize(hyp, @gp, -int32(250 * 3 / 3), inference, meanfunc, covfunc, likfunc, X, y);

% Optimise on full data
if 10 > 0
    hyp_opt = minimize(hyp_opt, @gp, -10, inference, meanfunc, covfunc, likfunc, X_full, y_full);
end

% Evaluate the nll on the full data
best_nll = gp(hyp_opt, inference, meanfunc, covfunc, likfunc, X_full, y_full)

save( '/scratch/tmpR0F_cL.out', 'hyp_opt', 'best_nll', 'nlls');
% exit();

system('scp -o ConnectTimeout=300 -i /users/jrl44/.ssh/jrl44fear2sagarmatha /scratch/tmpR0F_cL.out jrl44@sagarmatha:/tmp; rm /scratch/tmpR0F_cL.out')

addpath(genpath('/users/jrl44/matlab'))

rand('twister', 1027899619);
randn('state', 1027899619);

a='Load the data, it should contain X and y.'
load 'tmpMDlshP.mat'
X = double(X)
y = double(y)

X_full = X;
y_full = y;

if true & (250 < size(X, 1))
    subset = randsample(size(X, 1), 250, false)
    X = X_full(subset,:);
    y = y_full(subset);
end

% Load GPML
addpath(genpath('/users/jrl44/GPML'));

% Set up model.
meanfunc = {@meanZero}
hyp.mean = [  ]

covfunc = {@covChangeWindowMultiD, {1, {@covSum, {{@covNoise}, {@covSEiso}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}, {@covPeriodicNoDC}}}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}, {@covPeriodicNoDC}, {@covLinear}}}, {@covProd, {{@covSEiso}, {@covLinear}}}}}, {@covConst}}}
hyp.cov = [ 1688.18275358 -3.23579381547 2.19933455752 -2.44382893961 9.69507152266 7.07589193426 3.4342620207 3.06832941595 -1.64547869751 4.76304074628 -5.52500904465 1.46514288707 3.12626470964 -1.20159020811 3.3491966881 5.46968342113 -1.85366898507 4.76616034263 -5.39652649444 1.56263161342 3.26485172408 -1.54524662551 -4.24443712505 2270.10281168 9.70329589937 7.06412478193 -6.09844149578 1807.07107071 7.24880915861 ]

likfunc = {@likDelta}
hyp.lik = [  ]

inference = @infDelta

% Optimise on subset
[hyp_opt, nlls] = minimize(hyp, @gp, -int32(250 * 3 / 3), inference, meanfunc, covfunc, likfunc, X, y);

% Optimise on full data
if 10 > 0
    hyp_opt = minimize(hyp_opt, @gp, -10, inference, meanfunc, covfunc, likfunc, X_full, y_full);
end

% Evaluate the nll on the full data
best_nll = gp(hyp_opt, inference, meanfunc, covfunc, likfunc, X_full, y_full)

save( '/scratch/tmpeOfiO9.out', 'hyp_opt', 'best_nll', 'nlls');
% exit();

system('scp -o ConnectTimeout=300 -i /users/jrl44/.ssh/jrl44fear2sagarmatha /scratch/tmpeOfiO9.out jrl44@sagarmatha:/tmp; rm /scratch/tmpeOfiO9.out')

addpath(genpath('/users/jrl44/matlab'))

rand('twister', 242525551);
randn('state', 242525551);

a='Load the data, it should contain X and y.'
load 'tmpMDlshP.mat'
X = double(X)
y = double(y)

X_full = X;
y_full = y;

if true & (250 < size(X, 1))
    subset = randsample(size(X, 1), 250, false)
    X = X_full(subset,:);
    y = y_full(subset);
end

% Load GPML
addpath(genpath('/users/jrl44/GPML'));

% Set up model.
meanfunc = {@meanZero}
hyp.mean = [  ]

covfunc = {@covChangeWindowMultiD, {1, {@covSum, {{@covNoise}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}, {@covPeriodicNoDC}}}, {@covProd, {{@covSEiso}, {@covPeriodicNoDC}}}}}, {@covConst}}}
hyp.cov = [ 1688.18275358 -3.23579381547 2.19933455752 -2.67630966926 3.37709112005 5.40983447647 -1.66455742834 4.8570675275 -5.26283671931 1.51822076784 3.18422601924 -1.42757774385 9.81627178855 7.28100328345 -1.62379349138 2.62367383012 -0.805925209134 7.18298343858 ]

likfunc = {@likDelta}
hyp.lik = [  ]

inference = @infDelta

% Optimise on subset
[hyp_opt, nlls] = minimize(hyp, @gp, -int32(250 * 3 / 3), inference, meanfunc, covfunc, likfunc, X, y);

% Optimise on full data
if 10 > 0
    hyp_opt = minimize(hyp_opt, @gp, -10, inference, meanfunc, covfunc, likfunc, X_full, y_full);
end

% Evaluate the nll on the full data
best_nll = gp(hyp_opt, inference, meanfunc, covfunc, likfunc, X_full, y_full)
