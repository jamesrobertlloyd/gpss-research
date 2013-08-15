%% Cosine draw

x = linspace(-5, 5, 100)';

cov_func = {@covCos};
hyp.cov = [0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-5*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Cosine fit

x = linspace(-5, 5, 100)';
%y = cos(1.8*pi*x) + 0.1*randn(size(x));
y = cos(1*pi*x) + 0.1*randn(size(x)) + x;

cov_func = {@covCos};
hyp.cov = [0,0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y) / 10);

hyp = minimize(hyp, @gp, -10000, @infExact, mean_func, cov_func, lik_func, x, y);

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, x);

plot(x, y, 'o');
hold on;
plot(x, fit);
hold off;

%% Spectral draw

x = linspace(-5, 5, 100)';

cov_func = {@covProd, {@covSEiso, @covCos}};
hyp.cov = [1,0,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-5*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Spectral fit

x = linspace(-5, 5, 100)';
x_extrap = linspace(5, 15, 100)';
y = cos(1*pi*x) + 0.1*randn(size(x)) + x;

cov_func = {@covProd, {@covSEiso, @covCosUnit}};
hyp.cov = [0,0,0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, x);
fit_extrap = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, x_extrap);

plot(x, y, 'o');
hold on;
plot(x, fit);
plot(x_extrap, fit_extrap);
hold off;

%% Spectral mixture fit

x = linspace(-5, 5, 100)';
x_extrap = linspace(5, 15, 100)';
y = cos(1*pi*x) + 0.1*randn(size(x)) + x;

cov_func = {@covSum, {{@covProd, {@covSEiso, @covCosUnit}}, {@covProd, {@covSEiso, @covCosUnit}}}};
hyp.cov = [0,0,0,2,0,2];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, x);
fit_extrap = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, x_extrap);

plot(x, y, 'o');
hold on;
plot(x, fit);
plot(x_extrap, fit_extrap);
hold off;

%% SE x Per draw

x = linspace(-5, 5, 1000)';

cov_func = {@covProd, {@covSEiso, @covPeriodic}};
hyp.cov = [1,0,-1,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-5*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);