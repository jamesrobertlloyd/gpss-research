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

%% IBM draw

x = linspace(-20, 20, 1000)';

cov_func = {@covIBMLin};
hyp.cov = [-2, -0, 4, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% IBM fit


%%%% Check the derivative w.r.t location - might just be v. tricky to
%%%% optimise

x = linspace(-15,-5, 100)';
x_extrap = linspace(-6, 5, 100)';
y = cos(1*pi*x) + 0.1*randn(size(x)) + x + 0.2*(x+5).*(x+5) + 150;

% cov_func = {@covIBM};
% hyp.cov = [0,0];
cov_func = {@covSum, {@covCos, @covIBMLin}};
hyp.cov = [0, -0.33, -1.6,+-10,5,0];

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

%% IBM fit

%%%% Check the derivative w.r.t location - might just be v. tricky to
%%%% optimise

load 01-airline

x = X;

% cov_func = {@covIBM};
% hyp.cov = [0,0];
% cov_func = {@covProd, {@covPeriodic, @covIBMLin}};
% hyp.cov = [0, 0, 0, 0, mean(x), 0, 0];
cov_func = {@covProd, {@covPeriodic, @covIBM}};
hyp.cov = [0, 0, 0, 0, mean(x)];
% cov_func = {@covSum, {{@covProd, {@covPeriodic, @covLINscaleshift}}, @covConst}};
% hyp.cov = [0, 0, 0, 0, mean(x), 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x), max(x)+20, 1000)';

%xrange = x + 30;
%xrange = x;

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% Check IBM grad

x = linspace(-10, 10, 100)';

delta = 0.00001;
i = 2;

cov_func = {@covIBM};
hyp1.cov = [0.4, -5, 0, 0];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% Burst draw

x = linspace(-20, 20, 1000)';

cov_func = {@covBurst, {@covSEiso}};
hyp.cov = [-2, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Quick step draw

x = linspace(-10, 10, 1000)';

cov_func = {@covQuickStep, {@covPeriodic}};
hyp.cov = [0, 0, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);