

%a='Load the data, it should contain X and y.'
%load '%(datafile)s'
X = double(X)
y = double(y)
Xtest = double(Xtest)
ytest = double(ytest)

%% Load GPML
%addpath(genpath('%(gpml_path)s'));

%% Set up model.
meanfunc = {@meanZero}
hyp.mean = []

covfunc = {@covSum, {@covSEiso, @covNoise}}
hyp.cov = [0,0,0]

likfunc = @likDelta
hyp.lik = []

%% Optimize a little anyways.
[hyp_opt, nlls] = minimize(hyp, @gp, -100, @infDelta, meanfunc, covfunc, likfunc, X, y);
%%best_nll = nlls(end)

model.hypers = hyp_opt;

%% Evaluate a test points.
[ymu, ys2, predictions, fs2, loglik] = gp(model.hypers, @infDelta, meanfunc, covfunc, likfunc, X, y, Xtest, ytest)

actuals = ytest;
timestamp = now

%save('%(writefile)s', 'loglik', 'predictions', 'actuals', 'model', 'timestamp');

%% Set up model.
meanfunc = {@meanZero}
hyp.mean = []

covfunc = {@covSum, {@covSEiso}}
hyp.cov = [0,0]

likfunc = @likGauss
hyp.lik = [0]

%% Optimize a little anyways.
[hyp_opt, nlls] = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, X, y);
%%best_nll = nlls(end)

model.hypers = hyp_opt;

%% Evaluate a test points.
[ymu, ys2, predictions, fs2, loglik] = gp(model.hypers, @infExact, meanfunc, covfunc, likfunc, X, y, Xtest, ytest)

actuals = ytest;
timestamp = now

%save('%(writefile)s', 'loglik', 'predictions', 'actuals', 'model', 'timestamp');