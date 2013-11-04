%% Generate data from SE

X = linspace(0,1,100)';
cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-1,0,-2];

K = feval(cov_func{:}, hyp.cov, X);

y = chol(K)' * randn(size(X));

%% Plot

plot(X, y, 'o');

%% Fit SE with lik Delta

cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-2,0,log(std(y) / 10)];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likDelta;
hyp.lik = [];

hyp = minimize(hyp, @gp, -1000, @infDelta, mean_func, cov_func, lik_func, X, y);

%% Fit SE with infExact

cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-2,0,-1];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = [-Inf];

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, X, y);

%% Predict

X_test = linspace(0, 1, 1000)';
y_test = gp(hyp, @infExact, mean_func, cov_func, lik_func, X, y, X_test);

plot(X, y, 'o');
hold on;
plot(X_test, y_test);
hold off;