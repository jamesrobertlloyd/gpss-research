%% Change window draw

x = linspace(-10, 10, 100)';

cov_func = {@covChangeWindowMultiD, {1, @covSEiso, @covSEiso}};
%cov_func = {@covSum, {@covChangeWindowMultiD, 1, {@covSEiso, @covSEiso}}};
cov_func = {@covSum, {cov_func, {@covNoise}}};
%cov_func = {@covSum, {cov_func}};
hyp.cov = [0, 0, 2, 0, 0, 2, 0, -2];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y, 'o');

%% Change point draw

x = linspace(-10, 10, 100)';
[X,Y] = meshgrid(x,x);
x = [X(:), Y(:)];

cov_func = {@covChangeWindowMultiD, 2, {{@covMask, {[1, 0], @covSEiso}}, {@covMask, {[1, 1], @covSEiso}}}};
hyp.cov = [0, 0, 2, 0, 0, 0, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x, 1),1);

surf(reshape(y, 100, 100));

%% Check CP tanh grad

x = linspace(-10, 10, 50)';
[X,Y] = meshgrid(x,x);
x = [X(:), Y(:)];

delta = 0.00000001;
i = 4;

cov_func = {@covChangePointMultiD, {1, {@covMask, {[1, 0], @covSEiso}}, {@covMask, {[1, 1], @covSEiso}}}};
hyp.cov = [0, 0, 0, 3, 1, 1];
hyp1.cov = hyp.cov;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% ChangePointTanh fit

%load 01-airline

x = linspace(-15, 15, 250)';
y = cos(1*pi*x) .* (1 - max(0,x-5)>0) .* (max(0,x+5)>0) + cos(2*pi*x) .* ((max(0,x-5)>0) + (1 - (max(0,x+5)>0))) + 0.1*randn(size(x)) + max(0,x);

cov_func = {@covChangeWindowMultiD, 1, {{@covProd, {@covPeriodic, @covSEiso}}, {@covProd, {@covPeriodic, @covSEiso}}}};
hyp.cov = [-0.3227 -1.4131 1.8835 3.4256 -7.4231e-05 1.3844 3.4111 1.3844 -0.2244 0.7251 0.3409 0.4526 0.3409];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

repeats = 1;
total_iters = 200;
for i = 1:repeats
  hyp = minimize(hyp, @gp, -floor(total_iters/repeats), @infExact, mean_func, cov_func, lik_func, x, y);
end

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% Lin

x = linspace(-1, 1, 1000)';

cov_func = {@covLinear};
hyp.cov = [1, 0.5];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check grad

x = linspace(-1, 1, 100)';

delta = 0.0000001;
i = 1;

hyp1.cov = hyp.cov;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% ChangePointTanh fit

%load 01-airline

x = linspace(-15, 15, 250)';
y = x + 0.01*randn(size(x));

cov_func = {@covLinear};
hyp.cov = [0, -15];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

repeats = 1;
total_iters = 500;
for i = 1:repeats
  hyp = minimize(hyp, @gp, -floor(total_iters/repeats), @infExact, mean_func, cov_func, lik_func, x, y);
end

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;