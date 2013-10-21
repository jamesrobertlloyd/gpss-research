%% Generate data from SE

X = linspace(0,1,100)';
cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-1,0,-2];

K = feval(cov_func{:}, hyp.cov, X);

y = chol(K)' * randn(size(X));

%% Generate data from SE with linearly growing noise

X = linspace(0,1,250)';
cov_func = {@covSum, {@covSEiso, {@covProd, {@covLIN, @covNoise}}}};
hyp.cov = [-2,0,2,0,-2];

temp_K = feval(cov_func{:}, hyp.cov, X);

y = chol(temp_K)' * randn(size(X));

%% Generate data from periodic

X = linspace(0,1,250)';
cov_func = {@covSum, {@covPeriodicCentre, @covNoise}};
hyp.cov = [2,-2,1,-3];

K = feval(cov_func{:}, hyp.cov, X);

y = chol(K)' * randn(size(X));

%% Generate data from nearly periodic

X = linspace(0,1,250)';
y = sin(10*(X+1*X.*X));
y = y + 0.1*randn(size(y));

%% Load airline

load('01-airline.mat');

%% Load solar

load('02-solar.mat');

%% Plot

plot(X, y, 'o');

%% Fit SE

cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-2,0,log(std(y) / 10)];

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

%% Plot prior and posterior acf

prior_samples = chol(non_singular(K1))' * randn(length(y), 100);
prior_acf = autocorr(prior_samples(:,1), length(y) -1);
prior_acf = zeros(length(prior_acf), 100);
for i = 1:100
    prior_acf(:,i) = autocorr(prior_samples(:,i), length(y) -1);
end

figure(1);
plot(mean(prior_acf, 2), 'b');
hold on;
plot(quantile(prior_acf, 0.95, 2), 'b--');
plot(quantile(prior_acf, 0.05, 2), 'b--');
%hold off;

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_acf = autocorr(post_samples(:,1), length(y) -1);
post_acf = zeros(length(post_acf), 100);
for i = 1:100
    post_acf(:,i) = autocorr(post_samples(:,i), length(y) -1);
end

%figure(2);
plot(mean(post_acf, 2), 'g');
%hold on;
plot(quantile(post_acf, 0.95, 2), 'g--');
plot(quantile(post_acf, 0.05, 2), 'g--');
hold off;

figure(2);
two_band_plot((1:size(post_acf, 1))', ...
              mean(prior_acf, 2), ...
              quantile(prior_acf, 0.95, 2), ...
              quantile(prior_acf, 0.05, 2), ...
              mean(post_acf, 2), ...
              quantile(post_acf, 0.95, 2), ...
              quantile(post_acf, 0.05, 2));

%% Plot prior and posterior acf

prior_samples = chol(non_singular(K2))' * randn(length(y), 100);
prior_acf = autocorr(prior_samples(:,1), length(y) -1);
prior_acf = zeros(length(prior_acf), 100);
for i = 1:100
    prior_acf(:,i) = autocorr(prior_samples(:,i), length(y) -1);
end

figure(1);
plot(mean(prior_acf, 2), 'b');
hold on;
plot(quantile(prior_acf, 0.95, 2), 'b--');
plot(quantile(prior_acf, 0.05, 2), 'b--');
%hold off;

post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_acf = autocorr(post_samples(:,1), length(y) -1);
post_acf = zeros(length(post_acf), 100);
for i = 1:100
    post_acf(:,i) = autocorr(post_samples(:,i), length(y) -1);
end

%figure(2);
plot(mean(post_acf, 2), 'g');
%hold on;
plot(quantile(post_acf, 0.95, 2), 'g--');
plot(quantile(post_acf, 0.05, 2), 'g--');
hold off;

%% Plot distribution of minimum of acf

prior_samples = chol(non_singular(K1))' * randn(length(y), 100);
prior_troughs = zeros(100,1);
for i = 1:100
    prior_acf = autocorr(prior_samples(:,i), length(y) -1);
    prior_troughs(i) = find(prior_acf == min(prior_acf));
end

figure(1);
hist(prior_troughs);
%hold on;

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_troughs = zeros(100,1);
for i = 1:100
    post_acf = autocorr(post_samples(:,i), length(y) -1);
    post_troughs(i) = find(post_acf == min(post_acf));
end

figure(2);
hist(post_troughs);
%hold off;

%% Is the posterior acf trough too large?

prior_samples = chol(non_singular(K1))' * randn(length(y), 100);
prior_troughs = zeros(100,1);
for i = 1:100
    prior_acf = autocorr(prior_samples(:,i), length(y) -1);
    prior_troughs(i) = find(prior_acf == min(prior_acf));
end

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_troughs = zeros(100,1);
for i = 1:100
    post_acf = autocorr(post_samples(:,i), length(y) -1);
    post_troughs(i) = find(post_acf == min(post_acf));
end

% The random normals break ties
p_value = sum(prior_troughs > post_troughs + 0.001*randn(size(post_troughs))) / length(prior_troughs);
p_value

%% Is the posterior acf trough too large?

prior_samples = chol(non_singular(K2))' * randn(length(y), 100);
prior_troughs = zeros(100,1);
for i = 1:100
    prior_acf = autocorr(prior_samples(:,i), length(y) -1);
    prior_troughs(i) = find(prior_acf == min(prior_acf));
end

post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_troughs = zeros(100,1);
for i = 1:100
    post_acf = autocorr(post_samples(:,i), length(y) -1);
    post_troughs(i) = find(post_acf == min(post_acf));
end

% The random normals break ties
p_value = sum(prior_troughs > post_troughs + 0.001*randn(size(post_troughs))) / length(prior_troughs);
p_value

%% Is the posterior acf location too large?

prior_samples = chol(non_singular(K1))' * randn(length(y), 100);
prior_troughs = zeros(100,1);
for i = 1:100
    prior_acf = autocorr(prior_samples(:,i), length(y) -1);
    prior_troughs(i) = min(prior_acf);
end

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_troughs = zeros(100,1);
for i = 1:100
    post_acf = autocorr(post_samples(:,i), length(y) -1);
    post_troughs(i) = min(post_acf);
end

% The random normals break ties
p_value = sum(prior_troughs > post_troughs + 0.001*randn(size(post_troughs))) / length(prior_troughs);
p_value

%% Is the posterior acf trough too large?

prior_samples = chol(non_singular(K2))' * randn(length(y), 100);
prior_troughs = zeros(100,1);
for i = 1:100
    prior_acf = autocorr(prior_samples(:,i), length(y) -1);
    prior_troughs(i) = min(prior_acf);
end

post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_troughs = zeros(100,1);
for i = 1:100
    post_acf = autocorr(post_samples(:,i), length(y) -1);
    post_troughs(i) = min(post_acf);
end

% The random normals break ties
p_value = sum(prior_troughs > post_troughs + 0.001*randn(size(post_troughs))) / length(prior_troughs);
p_value