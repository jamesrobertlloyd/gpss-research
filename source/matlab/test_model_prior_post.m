%% Generate data from SE

X = linspace(0,1,100)';
cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-1,0,-2];

K = feval(cov_func{:}, hyp.cov, X);

y = chol(K)' * randn(size(X));

%% Generate data from periodic

X = linspace(0,1,250)';
cov_func = {@covSum, {@covPeriodicCentre, @covNoise}};
hyp.cov = [2,-2,1,-3];

K = feval(cov_func{:}, hyp.cov, X);

y = chol(K)' * randn(size(X));

%% Generate data from nearly periodic

X = linspace(0,1,250)';
y = sin(10*(X+2*X.*X));
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

%% Plot prior and posterior periodograms

prior_samples = chol(non_singular(K1))' * randn(length(y), 100);
prior_pxx = 10 * log10(periodogram(prior_samples(:,1)));
prior_pxx = zeros(length(prior_pxx), 100);
for i = 1:100
    prior_pxx(:,i) = 10 * log10(periodogram(prior_samples(:,i)));
end

figure(1);
plot(mean(prior_pxx, 2), 'b');
hold on;
plot(quantile(prior_pxx, 0.95, 2), 'b--');
plot(quantile(prior_pxx, 0.05, 2), 'b--');
%hold off;

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_pxx = 10 * log10(periodogram(post_samples(:,1)));
post_pxx = zeros(length(post_pxx), 100);
for i = 1:100
    post_pxx(:,i) = 10 * log10(periodogram(post_samples(:,i)));
end

%figure(2);
plot(mean(post_pxx, 2), 'g');
%hold on;
plot(quantile(post_pxx, 0.95, 2), 'g--');
plot(quantile(post_pxx, 0.05, 2), 'g--');
hold off;

%% Plot prior and posterior periodograms

prior_samples = chol(non_singular(K2))' * randn(length(y), 100);
prior_pxx = 10 * log10(periodogram(prior_samples(:,1)));
prior_pxx = zeros(length(prior_pxx), 100);
for i = 1:100
    prior_pxx(:,i) = 10 * log10(periodogram(prior_samples(:,i)));
end

figure(1);
plot(mean(prior_pxx, 2), 'b');
hold on;
plot(quantile(prior_pxx, 0.95, 2), 'b--');
plot(quantile(prior_pxx, 0.05, 2), 'b--');
%hold off;

post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_pxx = 10 * log10(periodogram(post_samples(:,1)));
post_pxx = zeros(length(post_pxx), 100);
for i = 1:100
    post_pxx(:,i) = 10 * log10(periodogram(post_samples(:,i)));
end

%figure(2);
plot(mean(post_pxx, 2), 'g');
%hold on;
plot(quantile(post_pxx, 0.95, 2), 'g--');
plot(quantile(post_pxx, 0.05, 2), 'g--');
hold off;

%% Plot distribution of peak of periodogram

prior_samples = chol(non_singular(K1))' * randn(length(y), 100);
prior_peaks = zeros(100,1);
for i = 1:100
    prior_pxx = 10 * log10(periodogram(prior_samples(:,i)));
    prior_peaks(i) = find(prior_pxx == max(prior_pxx));
end

figure(1);
hist(prior_peaks);
%hold on;

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = repmat(post_mean, 1, 100) + chol(post_cov)' * randn(length(y), 100);
post_peaks = zeros(100,1);
for i = 1:100
    post_pxx = 10 * log10(periodogram(post_samples(:,i)));
    post_peaks(i) = find(post_pxx == max(post_pxx));
end

figure(2);
hist(post_peaks);
%hold off;

%% Is the posterior periodogram peak too large?

prior_samples = chol(non_singular(K1))' * randn(length(y), 1000);
prior_peaks = zeros(100,1);
for i = 1:1000
    prior_pxx = 10 * log10(periodogram(prior_samples(:,i)));
    prior_peaks(i) = find(prior_pxx == max(prior_pxx));
end

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = repmat(post_mean, 1, 1000) + chol(post_cov)' * randn(length(y), 1000);
post_peaks = zeros(1000,1);
for i = 1:1000
    post_pxx = 10 * log10(periodogram(post_samples(:,i)));
    post_peaks(i) = find(post_pxx == max(post_pxx));
end

% The random normals break ties
p_value = sum(prior_peaks > post_peaks + 0.001*randn(size(post_peaks))) / length(prior_peaks);
p_value

%% Is the posterior periodogram peak too large? - other component

prior_samples = chol(non_singular(K2))' * randn(length(y), 1000);
prior_peaks = zeros(100,1);
for i = 1:1000
    prior_pxx = 10 * log10(periodogram(prior_samples(:,i)));
    prior_peaks(i) = find(prior_pxx == max(prior_pxx));
end

post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
post_samples = repmat(post_mean, 1, 1000) + chol(post_cov)' * randn(length(y), 1000);
post_peaks = zeros(1000,1);
for i = 1:1000
    post_pxx = 10 * log10(periodogram(post_samples(:,i)));
    post_peaks(i) = find(post_pxx == max(post_pxx));
end

% The random normals break ties
p_value = sum(prior_peaks > post_peaks + 0.001*randn(size(post_peaks))) / length(prior_peaks);
p_value

%% Is the value of the posterior periodogram peak too large?

prior_samples = chol(non_singular(K1))' * randn(length(y), 1000);
prior_peaks = zeros(100,1);
for i = 1:1000
    prior_pxx = 10 * log10(periodogram(prior_samples(:,i)));
    prior_peaks(i) = max(prior_pxx);
end

post_mean = K1 * (K \ y);
post_cov  = non_singular(K1 - K1 * (K \ K1));
post_samples = repmat(post_mean, 1, 1000) + chol(post_cov)' * randn(length(y), 1000);
post_peaks = zeros(1000,1);
for i = 1:1000
    post_pxx = 10 * log10(periodogram(post_samples(:,i)));
    post_peaks(i) = max(post_pxx);
end

% The random normals break ties
p_value = sum(prior_peaks > post_peaks + 0.001*randn(size(post_peaks))) / length(prior_peaks);
p_value

%% Is the value of the posterior periodogram peak too large?

prior_samples = chol(non_singular(K2))' * randn(length(y), 1000);
prior_peaks = zeros(100,1);
for i = 1:1000
    prior_pxx = 10 * log10(periodogram(prior_samples(:,i)));
    prior_peaks(i) = max(prior_pxx);
end

post_mean = K2 * (K \ y);
post_cov  = non_singular(K2 - K2 * (K \ K2));
post_samples = repmat(post_mean, 1, 1000) + chol(post_cov)' * randn(length(y), 1000);
post_peaks = zeros(1000,1);
for i = 1:1000
    post_pxx = 10 * log10(periodogram(post_samples(:,i)));
    post_peaks(i) = max(post_pxx);
end

% The random normals break ties
p_value = sum(prior_peaks > post_peaks + 0.001*randn(size(post_peaks))) / length(prior_peaks);
p_value