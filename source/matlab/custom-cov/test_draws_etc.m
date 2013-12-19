%% SE draw

x = linspace(0, 1, 1000)';

cov_func = {@covSEiso};
hyp.cov = [log(3),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);
ylim([min(y)-1, max(y)+1]);

%% SE * Lin draw

x = linspace(0, 1, 1000)';

cov_func = {@covProd, {@covSEiso, @covLINscaleshift}};
hyp.cov = [log(3),0,-1,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);
ylim([min(y)-1, max(y)+1]);

%% MT1 draw

x = linspace(0, 1, 1000)';

cov_func = {@covMaterniso, 1};
hyp.cov = [-3,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-5*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);
ylim([min(y)-1, max(y)+1]);

%% MT3 draw

x = linspace(0, 1, 1000)';

cov_func = {@covMaterniso, 3};
hyp.cov = [-2.5,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-5*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% MT5 draw

x = linspace(0, 1, 1000)';

cov_func = {@covMaterniso, 5};
hyp.cov = [0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Many materns

x = linspace(0, 1, 1000)';

cov_func = {@covSum, {{@covMaterniso, 5}, {@covMaterniso, 3}, {@covMaterniso, 1}}};
hyp.cov = [0,0,0,0,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Cosine draw

x = linspace(-0, 1, 1000)';

cov_func = {@covCos};
hyp.cov = [-log(2),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check Cos grad

x = linspace(-10, 10, 100)';

delta = 0.000000001;
i = 2;

cov_func = {@covCos};
hyp1.cov = [-2, -2];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

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

%% Fourier fit

%x = linspace(-5, 5, 100)';
%y = cos(1.8*pi*x) + 0.1*randn(size(x));
%y = cos(2.2*pi*x) + 0.1*randn(size(x)) + 0*x;

cov_func = {@covSum, {@covFourier, @covNoise}};
hyp.cov = [4.9,0,0,-5];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likDelta;
hyp.lik = [];

hyp = minimize(hyp, @gp, -2000, @infDelta, mean_func, cov_func, lik_func, x, y);

fit = gp(hyp, @infDelta, mean_func, cov_func, lik_func, x, y, x);

plot(x, y, 'o');
hold on;
plot(x, fit);
hold off;

%% Centered periodic no DC draw

x = linspace(0, 1, 1000)';

cov_func = {@covPeriodicNoDC};
hyp.cov = [10,log(max(x)/10),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);
hold on;
plot(x, 0.5*(max(y)-min(y))*sin(x*4*pi) + mean(y), 'r');
hold off;
%figure;
%plot(K(:,1));

%% noDC vs Fourier

x = linspace(0, 1, 1000)';

hyp.cov = [-2,log(max(x)/20),1];

cov_func = {@covPeriodicNoDC};
K1 = feval(cov_func{:}, hyp.cov, x);

cov_func = {@covFourier};
K2 = feval(cov_func{:}, hyp.cov, x);

max(max(abs(K2 - K1)))

%% No DC grad check

x = linspace(-10, 10, 1000)';

delta = 0.00000000001;
i = 2;

cov_func = {@covPeriodicNoDC};
hyp1.cov = [-2, -3, 1];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% Centered periodic draw

x = linspace(0, 1, 1000)';

cov_func = {@covPeriodicCentre};
hyp.cov = [log(100),log(max(x)/2),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);
hold on;
plot(x, 0.5*(max(y)-min(y))*sin(x*4*pi) + mean(y), 'r');
hold off;
%figure;
%plot(K(:,1));

%% periodic draw

x = linspace(0, 1000, 1000)';

cov_func = {@covPeriodicCentre};
hyp.cov = [log(10),log(max(x)/10),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

figure;
plot(x, y);

%% Fourier draw

x = linspace(0, 10, 1000)';

cov_func = {@covFourier};
hyp.cov = [3,log(max(x))-3.2,log(4)];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-6*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

figure;
plot(x, y);

%% Check Fourier grad

x = linspace(-5, 5, 1000)';

delta = 0.000001;
i = 2;

cov_func = {@covFourier};
hyp1.cov = [1, 1, 1] + 3.99;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% periodic draw

x = linspace(0, 1000, 1000)';

cov_func = {@covPeriodic};
hyp.cov = [log(1),log(max(x)/2),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

figure;
plot(x, y);

%% Check centered periodic grad

x = linspace(-5, 5, 1000)';

delta = 0.000000001;
i = 3;

cov_func = {@covPeriodicCenter};
hyp1.cov = [0, 0, 0] - 1;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% centered periodic fit

x = linspace(-5, 5, 100)';
y = cos(1.8*pi*x) + 2*cos(3.6*pi*x) + 0.1*randn(size(x));

cov_func = {@covPeriodicCenter};
hyp.cov = [0,0,0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-10, max(x)+10, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
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

x = linspace(0, 1, 1000)';

cov_func = {@covIBMLin};
hyp.cov = [100, 0, -1000, -1000];

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

x = linspace(-10, 10, 100)';

cov_func = {@covBurst, {@covSEiso}};
hyp.cov = [0, 1, 2, -2, 2];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check Burst grad

delta = 0.00001;
i = 3;

cov_func = {@covBurst, {@covSEiso}};
hyp1.cov = [2, 0, 2, 2, 2];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% Burst fit

%y = y + 0.05 * randn(size(y));

cov_func = {@covBurst, {@covSEiso}};
hyp.cov = [-2, 0, 0, -3, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% BurstLin draw

x = linspace(-10, 10, 1000)';

cov_func = {@covBurstLin, {@covSEiso}};


K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check BurstLin grad

delta = 0.0000001;
i = 3;

cov_func = {@covBurstLin, {@covSEiso}};
hyp1.cov = [1, 3, 1, 1, 1];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% Burst fit

%y = y + 0.05 * randn(size(y));

cov_func = {@covBurstLin, {@covSEiso}};
hyp.cov = [0, 0, 2, -2, 2];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% BurstTanh draw

x = linspace(-10, 10, 1000)';

cov_func = {@covBurstTanh, {@covSEiso}};
hyp.cov = [0, 1, 2, -2, 2];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check BurstTanh grad

delta = 0.0000001;
i = 3;

cov_func = {@covBurstTanh, {@covSEiso}};
hyp1.cov = [2, 1, 2, -2, 2];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% Quick step draw

x = linspace(-100, 100, 1000)';

cov_func = {@covQuickStep, {@covSEiso}};
hyp.cov = [-0, -2];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Change point draw

x = linspace(-10, 10, 1000)';

cov_func = {@covChangePoint, {@covSEiso, @covSEiso}};
hyp.cov = [0, 0, -2, 0, 2, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check CP grad

x = linspace(-10, 10, 100)';

delta = 0.00000001;
i = 2;

cov_func = {@covChangePoint, {@covSEiso, @covSEiso}};
hyp1.cov = [1, 5, 1, 2, 3, 4] -2;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% ChangePoint fit

%%%% Check the derivative w.r.t location - might just be v. tricky to
%%%% optimise

%load 01-airline

x = linspace(-15, 15, 250)';
y = cos(1*pi*x) .* (1- (max(0,x)>0)) + cos(3*pi*x) .* ((max(0,x)>0)) + 0.1*randn(size(x)) + max(0,x);

cov_func = {@covChangePoint, {{@covProd, {@covPeriodic, @covSEiso}}, {@covProd, {@covPeriodic, @covSEiso}}}};
hyp.cov = [-0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

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

%% Change point Lin draw

x = linspace(-10, 10, 1000)';

cov_func = {@covChangePointLin, {@covSEiso, @covSEiso}};
hyp.cov = [0, 0, -2, 0, 2, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check CPLin grad

x = linspace(-10, 10, 1000)';

delta = 0.00000001;
i = 2;

cov_func = {@covChangePoint, {@covSEiso, @covSEiso}};
hyp1.cov = [1, 1, 1, 2, 3, 4] -2;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% ChangePointLin fit

%load 01-airline

x = linspace(-15, 15, 250)';
y = cos(1*pi*x) .* (1- (max(0,x)>0)) + cos(3*pi*x) .* ((max(0,x)>0)) + 0.1*randn(size(x)) + max(0,x);

cov_func = {@covChangePointLin, {{@covProd, {@covPeriodic, @covSEiso}}, {@covProd, {@covPeriodic, @covSEiso}}}};
hyp.cov = [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

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

%% Change point tanh draw

x = linspace(-10, 10, 1000)';

cov_func = {@covChangePointTanh, {@covSEiso, @covSEiso}};
hyp.cov = [0, 0, 0, 0, 2, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check CP tanh grad

x = linspace(-10, 10, 1000)';

delta = 0.00000001;
i = 2;

cov_func = {@covChangePointTanh, {@covSEiso, @covSEiso}};
hyp1.cov = hyp.cov;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% ChangePointTanh fit

%load 01-airline

x = linspace(-15, 15, 250)';
y = cos(1*pi*x) .* (1- (max(0,x)>0)) + cos(3*pi*x) .* ((max(0,x)>0)) + 0.1*randn(size(x)) + max(0,x);

cov_func = {@covChangePointLin, {{@covProd, {@covPeriodic, @covSEiso}}, {@covProd, {@covPeriodic, @covSEiso}}}};
%hyp.cov = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

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

%% Change point burst draw

x = linspace(-10, 10, 1000)';

cov_func = {@covChangeBurstTanh, {@covSEiso, @covSEiso}};
hyp.cov = [0, 1, 2, -1, 0, 1, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check CP tanh grad

x = linspace(-15, 15, 1000)';

delta = 0.00001;
i = 6;

hyp.cov = [0, 0, 0, -2, 0, 2, 0];

cov_func = {@covChangeBurstTanh, {@covSEiso, @covSEiso}};
hyp1.cov = hyp.cov;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% ChangeBurstTanh fit

cov_func = {@covChangeBurstTanh, {@covSEiso, @covSEiso}};
%hyp.cov = [0, 1, 2, -1, 0, 1, 0];
hyp.cov = [0, 0, 0, 0, 0, 0, 0];

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

%% Burst and step draw

x = linspace(-10, 10, 250)';

cov_func = {@covSum, {{@covChangePoint, {@covConst, @covConst}}, {@covBurst, {@covSEiso}}}};
hyp.cov = [-5, 2, 2, 2, 5, 1, 1, -2, 2];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

y = y + 0.5*randn(size(y));

plot(x, y, 'o');

X = x;

save('step_and_burst', 'X', 'y');

%% Blackout draw

x = linspace(-10, 10, 100)';

cov_func = {@covBlackout, {@covSEiso}};
hyp.cov = [0, 1, 2, -2, -1, 2];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check Blackout grad

delta = 0.00001;
i = 3;

cov_func = {@covBlackout, {@covSEiso}};
hyp1.cov = [2, 3, 1, 1, 1, 1] + 1;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% Blackout fit

%y = y + 0.05 * randn(size(y));

cov_func = {@covBlackout, {@covSEiso}};
hyp.cov = [0, 0, 0, 0, 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% Blackout fit

load '02-solar.mat'

%X = X - mean(X);
%X = X / std(X);
y = y - min(y);
y = y / std(y);

x = X;

cov_func = {@covBlackout, {@covSEiso}};
hyp.cov = [1695, -2.5, 4.8, -2, 4, 1];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -500, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% Blackout Lin draw

x = linspace(-10, 10, 500)';

cov_func = {@covBlackoutLin, {@covSEiso}};
hyp.cov = [0, 1, 2, -2, -1, 2];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check Blackout Lin grad

delta = 0.000000001;
i = 2;

cov_func = {@covBlackoutLin, {@covSEiso}};
hyp1.cov = [ 0.0229   -0.2586    1.6875   -0.1492   -0.9620    1.9555];
hyp1.cov = hyp.cov;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% Blackout Lin fit

y = y + 0.05 * randn(size(y));

cov_func = {@covBlackoutLin, {@covSEiso}};
hyp.cov = [0, 0, 0, 0, 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% Blackout Lin fit

load '02-solar.mat'

%X = X - mean(X);
%X = X / std(X);
y = y - min(y);
y = y / std(y);

x = X;

cov_func = {@covBlackoutLin, {@covSEiso}};
hyp.cov = [1695, -2.5, 4.8, -2, 4, 1];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -500, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% Blackout Lin draw

x = linspace(-10, 10, 500)';

cov_func = {@covBlackoutTanh, {@covSEiso}};
hyp.cov = [0, 1, 2, -2, -1, 2];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check Blackout Lin grad

delta = 0.00001;
i = 3;

cov_func = {@covBlackoutTanh, {@covSEiso}};
hyp1.cov = [ 0.0229   -0.2586    1.6875   -0.1492   0.9620    1.9555];
hyp1.cov = hyp.cov;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% Blackout Lin fit

y = y + 0.05 * randn(size(y));

cov_func = {@covBlackoutTanh, {@covSEiso}};
hyp.cov = [0, 0, 0, 0, 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -1000, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% Blackout Lin fit

load '02-solar.mat'

%X = X - mean(X);
%X = X / std(X);
y = y - min(y);
y = y / std(y);

x = X;

cov_func = {@covBlackoutTanh, {@covProd, {{@covSEiso}, {covPeriodic}}}};
hyp.cov = [1695, -2.5, 4.8, -2, 4, 1, 0, 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -500, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% IMT1 draw

x = linspace(-1, 1, 1000)';

cov_func = {@covIMT1};
hyp.cov = [0,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check IMT1 grad

delta = 0.00001;
i = 3;

cov_func = {@covIMT1};
hyp1.cov = [0, 0, 0];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% IMT1 fit

load '02-solar.mat'

%X = X - mean(X);
%X = X / std(X);
y = y - min(y);
y = y / std(y);

x = X;

cov_func = {@covSum, {@covIMT1, @covConst}};
hyp.cov = [10, mean(x), 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -50, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-100, max(x)+100, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% IMT1 fit

x = linspace(-1, 1, 100)';
y = x.^2 + 0.1*randn(size(x));

cov_func = {@covSum, {@covIMT1, @covConst}};
hyp.cov = [0, 0, 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -500, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-10, max(x)+10, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% IMT1Lin draw

x = linspace(-1, 1, 1000)';

cov_func = {@covIMT1Lin};
hyp.cov = [0,0,0,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check IMT1Lin grad

delta = 0.00001;
i = 5;

cov_func = {@covIMT1Lin};
hyp1.cov = [0, 0, 0, 0, 0];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% IMT1Lin fit

load '02-solar.mat'

%X = X - mean(X);
%X = X / std(X);
y = y - min(y);
y = y / std(y);

x = X;

cov_func = {@covSum, {@covIMT1, @covConst, @covRQiso, @covRQiso}};
hyp.cov = [12.6716 1.8114e+03 -1.7924 0.0402 0.7728 -0.5099 8.7169 2.6716 -0.5099 8.7169];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = -1.8044;

hyp = minimize(hyp, @gp, -100, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-10000, max(x)+10000, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% IMT1Lin fit

x = linspace(-1, 1, 100)';
y = x.^2 + 0.1*randn(size(x));

cov_func = {@covSum, {@covIMT1Lin, @covConst}};
hyp.cov = [0, 0, 0, 0, 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -500, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-10, max(x)+10, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% IMT3 draw

x = linspace(0, 1, 1000)';

cov_func = {@covIMT3};
hyp.cov = [-2.5,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check IMT3 grad

delta = 0.000001;
i = 3;

cov_func = {@covIMT3};
hyp1.cov = [0.1, 0.1, 0];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% IMT1 fit

load '02-solar.mat'

%X = X - mean(X);
%X = X / std(X);
y = y - min(y);
y = y / std(y);

x = X;

cov_func = {@covSum, {@covIMT3, @covIMT3, @covConst}};
hyp.cov = [0, mean(x), 0, 2, mean(x), 2, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -500, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-100, max(x)+100, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% IMT3 fit

x = linspace(-1, 1, 100)';
y = x.^2 + 0.1*randn(size(x));

cov_func = {@covSum, {@covIMT1, @covConst}};
hyp.cov = [0, 0, 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -500, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-10, max(x)+10, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;

%% IMT3Lin draw

x = linspace(-1, 1, 1000)';

cov_func = {@covIMT3Lin};
hyp.cov = [0.1,0,0,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check IMT3 grad

delta = 0.000001;
i = 5;

cov_func = {@covIMT3Lin};
hyp1.cov = [0.1, 0.1, 0, 0, 0];
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% IMT5 draw

x = linspace(0, 1, 1000)';

cov_func = {@covIMT5};
hyp.cov = [-2,0.5];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% IMT5 * SE draw

x = linspace(0, 100, 1000)';

cov_func = {@covProd, {@covSEiso, @covIMT5}};
hyp.cov = [5,0,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% exp draw

x = linspace(-5, 5, 1000)';

cov_func = {@covExp};
hyp.cov = [0.02,0,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Check exp grad

delta = 0.00000001;
i = 1;

cov_func = {@covExp};
hyp1.cov = [0.1, 0.1, 0]+0.3;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x, x) - feval(cov_func{:}, hyp2.cov, x, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% exp x Fourier draw

l = 10;
x = linspace(0, l, 1000)';

cov_func = {@covProd, {@covExp, @covFourier}};
hyp.cov = [2/l,0*l/2,0,0,log(l)-log(20),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Change point draw

x = linspace(-10, 10, 1000)';

cov_func = {@covChangePointMultiD, 1, {@covSEiso, @covSEiso}};
hyp.cov = [0, 0, 0, 0, 2, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Change point draw

x = linspace(-10, 10, 100)';
[X,Y] = meshgrid(x,x);
x = [X(:), Y(:)];

cov_func = {@covChangePointMultiD, 2, {{@covMask, {[1, 0], @covSEiso}}, {@covMask, {[1, 1], @covSEiso}}}};
hyp.cov = [0, 0, 0, 0, 0, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x, 1),1);

surf(reshape(y, 100, 100));

%% Check CP tanh grad

x = linspace(-10, 10, 50)';
[X,Y] = meshgrid(x,x);
x = [X(:), Y(:)];

delta = 0.0000001;
i = 2;

%cov_func = {@covChangePointMultiD, 1, {{@covMask, {[1, 0], @covSEiso}}, {@covMask, {[1, 1], @covSEiso}}}};
%hyp.cov = [0, 1, 2, 3, 1, 1];
hyp1.cov = hyp.cov;
hyp2.cov = hyp1.cov;
hyp2.cov(i) = hyp2.cov(i) + delta;

diff = -(feval(cov_func{:}, hyp1.cov, x) - feval(cov_func{:}, hyp2.cov, x)) / delta;
deriv = feval(cov_func{:}, hyp1.cov, x, x, i);

max(max(abs(diff - deriv)))

%% ChangePointTanh fit

%load 01-airline

x = linspace(-15, 15, 250)';
y = cos(1*pi*x) .* (1- (max(0,x)>0)) + cos(2*pi*x) .* ((max(0,x)>0)) + 0.1*randn(size(x)) + max(0,x);

cov_func = {@covChangePointMultiD, 1, {{@covProd, {@covPeriodic, @covSEiso}}, {@covProd, {@covPeriodic, @covSEiso}}}};
hyp.cov = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

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

%% Change window draw

x = linspace(-10, 10, 1000)';

cov_func = {@covChangeWindowMultiD, 1, {@covSEiso, @covSEiso}};
hyp.cov = [0, 0, 2, 0, 0, 2, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

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

delta = 0.0000001;
i = 1;

%cov_func = {@covChangePointMultiD, 1, {{@covMask, {[1, 0], @covSEiso}}, {@covMask, {[1, 1], @covSEiso}}}};
%hyp.cov = [0, 1, 2, 3, 1, 1];
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