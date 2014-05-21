rand('twister', 1);
randn('state', 1);

a='Load the data, it should contain X and y.'
load '/tmp/tmpA4jfIQ.mat'
X = double(X)
y = double(y)
Xtest = double(Xtest)
ytest = double(ytest)

%% Load GPML
addpath(genpath('/scratch/home/Research/GPs/gpss-research/source/gpml'));

%% Set up model.

meanfunc = {@meanZero}
hyp.mean = [  ]

covfunc = {@covProd, {{@covSum, {{@covConst}, {@covSEiso}, {@covPeriodicNoDC}}}, {@covSum, {{@covNoise}, {@covProd, {{@covSEiso}, {@covLinear}}}}}}}
hyp.cov = [ 1.45699285257 -0.704577132093 -1.83729240455 -0.299179164141 -0.000152961057915 -0.330389056015 0.0949193604473 3.02440453042 -1.69998056044 3.4377865315 1943.65539738 ]

likfunc = {@likDelta}
hyp.lik = [  ]

inference = @infDelta

model.hypers = hyp;

%% Evaluate at test points.
[ymu, ys2, predictions, fs2, loglik] = gp(model.hypers, inference, meanfunc, covfunc, likfunc, X, y, Xtest, ytest)

actuals = ytest;
timestamp = now

'%(output_file)s'

save('%(output_file)s', 'loglik', 'predictions', 'actuals', 'model', 'timestamp', 'ymu', 'ys2');

a='Supposedly finished writing file'