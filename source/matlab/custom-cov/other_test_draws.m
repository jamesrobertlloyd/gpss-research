%% Centered periodic draw - with long lengthscale it becomes sinusoidal

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

figure;
plot(x, K(1,:));
hold on;
plot(x, 0.5*(max(K(1,:))-min(K(1,:)))*cos(x*4*pi) + mean(K(1,:)), 'r');
hold off;

%% Product of one sinusoids

x = linspace(0, 1, 1000)';

cov_func = {@covProd, {@covCos}};
% hyp.cov = [-log(2),0];
hyp.cov = [-log(6),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Product of two sinusoid kernels

x = linspace(0, 1, 1000)';

cov_func = {@covProd, {@covCos, @covCos}};
hyp.cov = [-log(3),0,-log(6),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);
hold on;
plot(x, 0.5*(max(y)-min(y))*sin(x*12*pi).*sin(x*6*pi) + mean(y), 'r');
hold off;

%% Product of two sinusoids

x = linspace(0, 1, 1000)';

cov_func = {@covProd, {@covCos}};
hyp.cov = [-log(3),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y1 = chol(K)' * randn(size(x));

cov_func = {@covProd, {@covCos}};
hyp.cov = [-log(6),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y2 = chol(K)' * randn(size(x));

plot(x, y1.*y2);

%% Product of two periodic kernels

x = linspace(0, 1, 1000)';

cov_func = {@covProd, {@covPeriodicCentre, @covPeriodicCentre}};
hyp.cov = [-1,-log(5),0,1,-log(25),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Product of two Matern

x = linspace(1977, 1991, 1000)';

cov_func = {@covProd, {{@covMaterniso, 5}, {@covMaterniso, 5}, {@covMaterniso, 5}}};
hyp.cov = [4.494469 -2.520251 5.866723 5.492876 4.922308 4.333409];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

imagesc(K);

%% SE * Per

x = linspace(0, 1, 1000)';

cov_func = {@covProd, {@covSEiso, @covPeriodicCentre}};
hyp.cov = [log(1/(5*10)), 0, log(1),log(max(x)/5),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% SE * Cos

x = linspace(0, 1, 1000)';

cov_func = {@covProd, {@covSEiso, @covCos}};
hyp.cov = [-3,0,-3,0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x));

plot(x, y);

%% Cosine fit

x = linspace(-5, 5, 250)';
%y = cos(1.8*pi*x) + 0.1*randn(size(x));
y = cos(2*pi*x) + 0.01*randn(size(x));

cov_func = {@covCos};
hyp.cov = [0,0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y) / 10);

hyp = minimize(hyp, @gp, -100, @infExact, mean_func, cov_func, lik_func, x, y);

x_test = linspace(-5, 10, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, x_test);

plot(x, y, 'o');
hold on;
plot(x_test, fit);
hold off;

%% SE * Cosine fit

x = linspace(-5, 5, 500)';
y = 3*(cos(2*pi*(x+0.00*x.*x)).*(x>=0)+cos(1.0*2*pi*(x+0.00*x.*x)).*(x<0)) + 0.01*randn(size(x));
%x = x + 0.01*sin(2*pi*x/1.5);
%x = 2*x;

cov_func = {@covProd, {@covSEiso, @covCos}};
hyp.cov = [0,0,0,0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y) / 10);

hyp = minimize(hyp, @gp, -100, @infExact, mean_func, cov_func, lik_func, x, y);

%%

x_test = linspace(-10, 10, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, x_test);

K = feval(cov_func{:}, hyp.cov, x) + eye(length(x))*exp(2*hyp.lik);
K = K + 1e-9*max(max(K))*eye(size(K));
K_star = feval(cov_func{:}, hyp.cov, x, x_test);
K_starstar = feval(cov_func{:}, hyp.cov, x_test, x_test);
K = K_starstar - K_star'*(K \ K_star);
K = K + 1e-9*max(max(K))*eye(size(K));

plot(x, y, 'o');
hold on;
plot(x_test, fit);
plot(x_test, fit + chol(K)' * randn(size(x_test)), 'g');
plot(x_test, fit + chol(K)' * randn(size(x_test)), 'r');
hold off;

%% Flexible periodic function with fixed period

x = linspace(-20, 20, 1000)';
y = zeros(size(x));
y1 = y;
y2 = y;

cov_func = {@covSEiso};
hyp.cov = [log(3),0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));
chol_K = chol(K);

for i = 1:1
    y1 = y1 + (chol_K' * randn(size(x))).*cos(2*pi*i*x)./(i*i) + ...
              (chol_K' * randn(size(x))).*sin(2*pi*i*x)./(i*i);
    y2 = y2 + (chol_K' * randn(size(x))).*cos(2*pi*i*x)./(i*i) + ...
              (chol_K' * randn(size(x))).*sin(2*pi*i*x)./(i*i);
end
plot(x, y1, 'r');
hold on;
plot(x, y2, 'g');
hold off;