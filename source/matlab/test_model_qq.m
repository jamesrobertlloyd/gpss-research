%% Generate data from SE

X = linspace(0,1,250)';
cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-2,0,-2];

temp_K = feval(cov_func{:}, hyp.cov, X);

y = chol(temp_K)' * randn(size(X));

%% Generate data from SE with linearly growing noise

X = linspace(0,1,250)';
cov_func = {@covSum, {@covSEiso, {@covProd, {@covLIN, @covNoise}}}};
hyp.cov = [-2,0,0,0,-2];

temp_K = feval(cov_func{:}, hyp.cov, X);

y = chol(temp_K)' * randn(size(X));

%% Generate data from SE with heavy tailed noise

X = linspace(0,1,250)';
cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-2,0,-2];

temp_K = feval(cov_func{:}, hyp.cov, X);

y = chol(temp_K)' * randn(size(X));
y = y + trnd(3, length(y), 1);

%% Generate data from periodic

X = linspace(0,1,250)';
cov_func = {@covSum, {@covPeriodicCentre, @covNoise}};
hyp.cov = [2,-1,1,-3];

temp_K = feval(cov_func{:}, hyp.cov, X);

y = chol(temp_K)' * randn(size(X));

%% Generate data from nearly periodic

X = linspace(0,1,250)';
y = sin(10*(X+0.5*X.*X));
y = y + 0.1*randn(size(y));

%% Load airline

load('01-airline.mat');

%% Load solar

load('02-solar.mat');

%% Plot

plot(X, y, 'o');

%% Fit SE

cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-2,0,log(std(y)/10)];

cov_func_1 = @covSEiso;
cov_func_2 = @covNoise;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likDelta;
hyp.lik = [];

hyp = minimize(hyp, @gp, -1000, @infDelta, mean_func, cov_func, lik_func, X, y);

K = feval(cov_func{:}, hyp.cov, X);
K1 = feval(cov_func_1, hyp.cov(1:2), X);
K2 = feval(cov_func_2, hyp.cov(3), X);

%% Fit Periodic

cov_func = {@covSum, {@covPeriodicCentre, @covNoise}};
hyp.cov = [2,-2,1,-3];

cov_func_1 = @covPeriodicCentre;
cov_func_2 = @covNoise;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likDelta;
hyp.lik = [];

hyp = minimize(hyp, @gp, -1000, @infDelta, mean_func, cov_func, lik_func, X, y);

K = feval(cov_func{:}, hyp.cov, X);
K1 = feval(cov_func_1, hyp.cov(1:3), X);
K2 = feval(cov_func_2, hyp.cov(4), X);

%% Fit SE*Periodic

cov_func = {@covSum, {{@covProd, {@covSEiso, @covPeriodicCentre}}, @covNoise}};
hyp.cov = [0,0,2,-1,1,-3];

cov_func_1 = {@covProd, {@covSEiso, @covPeriodicCentre}};
cov_func_2 = @covNoise;

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likDelta;
hyp.lik = [];

hyp = minimize(hyp, @gp, -1000, @infDelta, mean_func, cov_func, lik_func, X, y);

K = feval(cov_func{:}, hyp.cov, X);
K1 = feval(cov_func_1{:}, hyp.cov(1:5), X);
K2 = feval(cov_func_2, hyp.cov(6), X);

%% Plot from posterior and prior

figure(1);
prior_sample = chol(non_singular(K1))' * randn(size(y));
plot(X, prior_sample, 'b');
hold on;
prior_sample = chol(non_singular(K1))' * randn(size(y));
plot(X, prior_sample, 'g');
prior_sample = chol(non_singular(K1))' * randn(size(y));
plot(X, prior_sample, 'r');
hold off;
    
post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
figure(2);
post_sample = post_mean + chol(post_cov)' * randn(size(y));
plot(X, post_sample, 'b');
hold on;
post_sample = post_mean + chol(post_cov)' * randn(size(y));
plot(X, post_sample, 'g');
post_sample = post_mean + chol(post_cov)' * randn(size(y));
plot(X, post_sample, 'r');
hold off;

%% Plot from posterior and prior

figure(1);
prior_sample = chol(non_singular(K2))' * randn(size(y));
plot(X, prior_sample, 'b');
hold on;
prior_sample = chol(non_singular(K2))' * randn(size(y));
plot(X, prior_sample, 'g');
prior_sample = chol(non_singular(K2))' * randn(size(y));
plot(X, prior_sample, 'r');
hold off;
    
post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
figure(2);
post_sample = post_mean + chol(post_cov)' * randn(size(y));
plot(X, post_sample, 'b');
hold on;
post_sample = post_mean + chol(post_cov)' * randn(size(y));
plot(X, post_sample, 'g');
post_sample = post_mean + chol(post_cov)' * randn(size(y));
plot(X, post_sample, 'r');
hold off;

%% Plot prior and posterior qq

prior_samples = chol(non_singular(K1))' * ...
                randn(length(y), 100);
prior_quantiles = quantile(prior_samples(:), linspace(0,1,length(y))');
% prior_samples = chol(non_singular(K1))' * randn(length(y), 100) ./ ...
%                 repmat(sqrt(diag(K1)), 1, 100);
prior_samples = chol(non_singular(K1))' * randn(length(y), 100);
prior_qq = zeros(length(y), 100);
for i = 1:100
    a = normcdf(sort(prior_samples(:,i)));
    prior_qq(:,i) = a;
end

figure(1);
plot(linspace(0, 1, length(y)), mean(prior_qq, 2), 'b');
hold on;
plot(linspace(0, 1, length(y)), quantile(prior_qq, 0.95, 2), 'b--');
plot(linspace(0, 1, length(y)), quantile(prior_qq, 0.05, 2), 'b--');
% hold off;

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = (repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100)) ./ ...
               repmat(sqrt(diag(K1)), 1, 100);
post_qq = zeros(length(y), 100);
for i = 1:100
    a = normcdf(sort(post_samples(:,i)));
    post_qq(:,i) = a;
end

%figure(2);
plot(linspace(0, 1, length(y)), mean(post_qq, 2), 'g');
%hold on;
plot(linspace(0, 1, length(y)), quantile(post_qq, 0.95, 2), 'g--');
plot(linspace(0, 1, length(y)), quantile(post_qq, 0.05, 2), 'g--');
hold off;

%% Plot prior and posterior qq

prior_samples = chol(non_singular(K2))' * randn(length(y), 100) ./ ...
                repmat(sqrt(diag(K2)), 1, 100);
prior_qq = zeros(length(y), 100);
for i = 1:100
    a = normcdf(sort(prior_samples(:,i)));
    prior_qq(:,i) = a;
end

figure(1);
plot(linspace(0, 1, length(y)), mean(prior_qq, 2), 'b');
hold on;
plot(linspace(0, 1, length(y)), quantile(prior_qq, 0.95, 2), 'b--');
plot(linspace(0, 1, length(y)), quantile(prior_qq, 0.05, 2), 'b--');
% hold off;

post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
post_samples = (repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100)) ./ ...
               repmat(sqrt(diag(K2)), 1, 100);
post_qq = zeros(length(y), 100);
for i = 1:100
    a = normcdf(sort(post_samples(:,i)));
    post_qq(:,i) = a;
end

%figure(2);
plot(linspace(0, 1, length(y)), mean(post_qq, 2), 'g');
%hold on;
plot(linspace(0, 1, length(y)), quantile(post_qq, 0.95, 2), 'g--');
plot(linspace(0, 1, length(y)), quantile(post_qq, 0.05, 2), 'g--');
hold off;

%% Plot distribution of location of maximum deviation

prior_samples = chol(non_singular(K1))' * randn(length(y), 100) ./ ...
                repmat(sqrt(diag(K1)), 1, 100);
prior_qq_d_loc = zeros(100, 1);
for i = 1:100
    a = normcdf(sort(prior_samples(:,i)));
    b = linspace(0, 1, length(y))';
    max_d = max(abs(a - b));
    prior_qq_d_loc(i) = find(abs(a-b) == max_d);
end

figure(1);
hist(prior_qq_d_loc);

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = (repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100)) ./ ...
               repmat(sqrt(diag(K1)), 1, 100);
post_qq_d_loc = zeros(100, 1);
for i = 1:100
    a = normcdf(sort(post_samples(:,i)));
    b = linspace(0, 1, length(y))';
    max_d = max(abs(a - b));
    post_qq_d_loc(i) = find(abs(a-b) == max_d);
end

figure(2);
hist(post_qq_d_loc);

%% Is the value of the deviation extreme

prior_samples = chol(non_singular(K1))' * randn(length(y), 100) ./ ...
                repmat(sqrt(diag(K1)), 1, 100);
prior_qq_d = zeros(100, 1);
for i = 1:100
    a = normcdf(sort(prior_samples(:,i)));
    b = linspace(0, 1, length(y))';
    max_d = max(abs(a - b));
    prior_qq_d(i) = max_d;
end

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = (repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100)) ./ ...
               repmat(sqrt(diag(K1)), 1, 100);
post_qq_d = zeros(100, 1);
for i = 1:100
    a = normcdf(sort(post_samples(:,i)));
    b = linspace(0, 1, length(y))';
    max_d = max(abs(a - b));
    post_qq_d(i) = max_d;
end

% The random normals break ties
p_value = sum(prior_qq_d > post_qq_d + 0.001*randn(size(post_qq_d))) / length(post_qq_d);
p_value

%% Is the value of the deviation extreme

prior_samples = chol(non_singular(K2))' * randn(length(y), 100) ./ ...
                repmat(sqrt(diag(K2)), 1, 100);
prior_qq_d = zeros(100, 1);
for i = 1:100
    a = normcdf(sort(prior_samples(:,i)));
    b = linspace(0, 1, length(y))';
    max_d = max(abs(a - b));
    prior_qq_d(i) = max_d;
end

post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
post_samples = (repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100)) ./ ...
               repmat(sqrt(diag(K2)), 1, 100);
post_qq_d = zeros(100, 1);
for i = 1:100
    a = normcdf(sort(post_samples(:,i)));
    b = linspace(0, 1, length(y))';
    max_d = max(abs(a - b));
    post_qq_d(i) = max_d;
end

% The random normals break ties
p_value = sum(prior_qq_d > post_qq_d + 0.001*randn(size(post_qq_d))) / length(post_qq_d);
p_value
