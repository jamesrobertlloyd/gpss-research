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