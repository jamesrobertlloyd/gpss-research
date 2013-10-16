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

%% Load airline

load('01-airline.mat');

%% Load solar

load('02-solar.mat');

%% Plot

plot(X, y, 'o');

%% Fit SE

cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-2,0,log(std(y) / 10)];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likDelta;
hyp.lik = [];

hyp = minimize(hyp, @gp, -1000, @infDelta, mean_func, cov_func, lik_func, X, y);

K = feval(cov_func{:}, hyp.cov, X);

%% Fit SE + SE

cov_func = {@covSum, {@covSEiso, @covSEiso, @covNoise}};
hyp.cov = [3,6,0,6,1];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likDelta;
hyp.lik = [];

hyp = minimize(hyp, @gp, -1000, @infDelta, mean_func, cov_func, lik_func, X, y);

K = feval(cov_func{:}, hyp.cov, X);

%% Load good airline covariance

cov_func = {@covSum, {{@covSEiso}, {@covSEiso}, {@covProd, {{@covNoise}, {@covLINscaleshift}}}, {@covProd, {{@covSEiso}, {@covFourier}, {@covLINscaleshift}}}}};
hyp.cov = [ -0.590119 2.197834 2.86931 6.282442 -0.293687 0.492275 1941.423324 3.153432 3.460612 -0.315508 0.002278 -0.032842 1.838454 1945.443724 ];

K = feval(cov_func{:}, hyp.cov, X);

%% Load good solar covariance

cov_func = {@covSum, {{@covNoise}, {@covChangeBurstTanh, {{@covSum, {{@covSEiso}, {@covChangePointTanh, {{@covSum, {{@covSEiso}, {@covSEiso}}}, {@covSum, {{@covSEiso}, {@covProd, {{@covSEiso}, {@covExp}}}}}}}}}, {@covConst}}}}};
hyp.cov = [ -5.868626 1679.988852 -2.12758 4.326001 13.016431 7.061076 1746.837124 -2.190126 -1.479298 -1.946564 1.073387 -1.477947 0.853083 -1.169796 -1.544589 -6.475691 0.002149 1879.320319 3.546089 7.254741 ];

K = feval(cov_func{:}, hyp.cov, X);

%% Were any points unlikely under the prior?

vars = diag(K);
standard = y ./ sqrt(vars);
p_point_prior = normcdf(standard);

plot(X, p_point_prior, 'o');

%% Plot qq plot of p values

qqplot(p_point_prior, linspace(0,1,10000));

%% Were any points unlikely under the LOO posterior

p_point_LOO = nan(size(X));
for i = 1:length(p_point_LOO)
    K_i = K([1:(i-1),(i+1):length(p_point_LOO)],:);
    K_ii = K_i(:,[1:(i-1),(i+1):length(p_point_LOO)]);
    K_i = K(i,[1:(i-1),(i+1):length(p_point_LOO)]);
    y_i = y([1:(i-1),(i+1):length(p_point_LOO)]);
    mean = K_i * (K_ii \ y_i);
    var = K(i,i) - K_i * (K_ii \ K_i');
    standard = (y(i) - mean) ./ sqrt(var);
    
    p_point_LOO(i) = normcdf(standard);
end

plot(X, p_point_LOO, 'o');

%% Plot qq plot of p values

qqplot(p_point_LOO, linspace(0,1,10000));

%% Overlay

plot(X, y, 'go');
hold on;
plot(X(p_point_LOO>0.95), y(p_point_LOO>0.95), 'ro');
plot(X(p_point_LOO<0.05), y(p_point_LOO<0.05), 'bo');
hold off;

%% Were any squared points unlikely under the LOO posterior

p_point_square_LOO = nan(size(X));
for i = 1:length(p_point_square_LOO)
    K_i = K([1:(i-1),(i+1):length(p_point_square_LOO)],:);
    K_ii = K_i(:,[1:(i-1),(i+1):length(p_point_square_LOO)]);
    K_i = K(i,[1:(i-1),(i+1):length(p_point_square_LOO)]);
    y_i = y([1:(i-1),(i+1):length(p_point_square_LOO)]);
    mean = K_i * (K_ii \ y_i);
    var = K(i,i) - K_i * (K_ii \ K_i');
    standard = ((y(i) - mean) ./ sqrt(var)).^2;
    
    p_point_square_LOO(i) = chi2cdf(standard, 1);
end

plot(X, p_point_square_LOO, 'o');

%% Plot qq plot of p values

qqplot(p_point_square_LOO, linspace(0,1,10000));

%% Overlay

plot(X, y, 'go');
hold on;
plot(X(p_point_square_LOO>0.95), y(p_point_square_LOO>0.95), 'ro');
plot(X(p_point_square_LOO<0.05), y(p_point_square_LOO<0.05), 'bo');
hold off;

%% Were any differences unlikely under the prior

p_dif_prior = nan(size(K));
for i = 1:length(X)
    for j = 1:length(X)
        var = K(i,i) + K(j,j) - 2*K(i,j);
        standard = (y(i) - y(j)) ./ sqrt(var);
        p_dif_prior(i,j) = normcdf(standard);
    end
end

imagesc(p_dif_prior);
colorbar;

%% Plot qq plot of p values

qqplot(p_dif_prior(:), linspace(0,1,10000));

%% Were any squared differences unlikely under the prior

p_dif_square_prior = nan(size(K));
for i = 1:length(X)
    for j = 1:length(X)
        var = K(i,i) + K(j,j) - 2*K(i,j);
        standard = ((y(i) - y(j)) ./ sqrt(var)) .^ 2;
        p_dif_square_prior(i,j) = chi2cdf(standard, 1);
    end
end

imagesc(p_dif_square_prior);
colorbar;

%% Plot qq plot of p values

qqplot(p_dif_square_prior(:), linspace(0,1,10000));

%% Were any differences unlikely under the LTO posterior

p_diff_LTO = nan(size(X));
p_diff_LTO_list = [];
for i = 1:length(X)
    i
    for j = (i+1):length(X)
        not_ij = [1:(i-1),(i+1):(j-1),(j+1):length(X)];
        K_ijij = K(not_ij,not_ij);
        K_i = K(i,not_ij);
        K_j = K(j,not_ij);
        K_ij = [K(i,i), K(i,j);
                K(j,i), K(j,j)];
        something_else = [K_i;K_j];
        y_ij = y(not_ij);
        mean_i = K_i * (K_ijij \ y_ij);
        mean_j = K_j * (K_ijij \ y_ij);
        mean = mean_i - mean_j;
        var = K_ij - something_else * (K_ijij \ something_else');
        var = var(1,1) + var(2,2) - 2*var(1,2);
        standard = ((y(i) - y(j)) - mean) ./ sqrt(var);

        p_diff_LTO(i, j) = normcdf(standard);
        p_diff_LTO_list = [p_diff_LTO_list; p_diff_LTO(i, j)];
    end
end

imagesc(p_diff_LTO);
colorbar;

%% Plot qq plot of p values

qqplot(p_diff_LTO_list(:), linspace(0,1,10000));

%% Plot LOO mean and variance on data

p_point_LOO = nan(size(X));
mean_LOO = nan(size(X));
var_LOO = nan(size(X));
for i = 1:length(p_point_LOO)
    K_i = K([1:(i-1),(i+1):length(p_point_LOO)],:);
    K_ii = K_i(:,[1:(i-1),(i+1):length(p_point_LOO)]);
    K_i = K(i,[1:(i-1),(i+1):length(p_point_LOO)]);
    y_i = y([1:(i-1),(i+1):length(p_point_LOO)]);
    mean_LOO(i) = K_i * (K_ii \ y_i);
    var_LOO(i) = K(i,i) - K_i * (K_ii \ K_i');
    standard = (y(i) - mean_LOO(i)) ./ sqrt(var_LOO(i));
    
    p_point_LOO(i) = normcdf(standard);
end

plot(X,y,'o');
hold on;
plot(X,mean_LOO,'b-');
plot(X,mean_LOO+2*sqrt(var_LOO),'b--');
plot(X,mean_LOO-2*sqrt(var_LOO),'b--');
hold off;

%% Plot LCO mean and variance on data

chunk_size = 0.1;

p_point_LCO = nan(size(X));
mean_LCO = nan(size(X));
var_LCO = nan(size(X));
for i = 1:length(X)
    not_close = abs(X-X(i)) > ((max(X) - min(X)) * chunk_size * 0.5);
    
    K_ii = K(not_close,not_close);
    K_i = K(i,not_close);
    y_i = y(not_close);
    
    mean_LCO(i) = K_i * (K_ii \ y_i);
    var_LCO(i) = K(i,i) - K_i * (K_ii \ K_i');
    standard = (y(i) - mean_LCO(i)) ./ sqrt(var_LCO(i));
    
    p_point_LCO(i) = normcdf(standard);
end

plot(X,y,'o');
hold on;
plot(X,mean_LCO,'b-');
plot(X,mean_LCO+2*sqrt(var_LCO),'b--');
plot(X,mean_LCO-2*sqrt(var_LCO),'b--');
hold off;

%% Plot p values

plot(X, p_point_LCO, 'o');

%% Plot qq plot of p values

qqplot(p_point_LCO, linspace(0,1,10000));

%% Overlay

plot(X, y, 'go');
hold on;
plot(X(p_point_LCO>0.95), y(p_point_LCO>0.95), 'ro');
plot(X(p_point_LCO<0.05), y(p_point_LCO<0.05), 'bo');
hold off;

%% Is the radius an interesting statistic in high d Gaussians?

p = nan(100,1);
r2s = nan(100,1);
for i = 1:length(p)

    X = linspace(0,1,100)';
    cov_func = {@covSum, {@covSEiso, @covNoise}};
    hyp.cov = [-1,0,-2];

    K = feval(cov_func{:}, hyp.cov, X);

    L = chol(K);

    y = L' * randn(size(X));

    z = L' \ y;

    r2 = sum(z.*z);
    r2s(i) = r2;

    p(i) = chi2cdf(r2, length(z));
end

%% Plot qq plot of p values

qqplot(p(:), linspace(0,1,10000));

%% What is the radius on real data? Answer: not interesting

L = chol(K);
z = L' \ y;
r2 = sum(z.*z);
p = chi2cdf(r2, length(z));

p

plot(normcdf(z), 'o')

%% Does marginal likelihood show anything interesting - nope, just radius

iters = 100;
lmls = nan(iters,1);
L = chol(K);
for i = 1:iters
    y_null = L' * randn(size(y));
    lmls(i) = y_null' * (K \ y_null);
end

hist(lmls);

lml = y'*(K\y);

p = sum(lmls < lml) / length(lmls)

%% Leave two next to each other out statistic?

p_point_LTO = nan(length(X)-1,1);
mean_LTO = nan(length(X)-1,1);
var_LTO = nan(length(X)-1,1);
X_LTO = nan(length(X)-1,1);
y_LTO = nan(length(X)-1,1);

for i = 1:(length(X) - 1)
    not_i = [1:(i-1),(i+2):length(X)];
    
    K_ii = K(not_i,not_i);
    K_i = K([i;i+1],not_i);
    y_i = y(not_i);
    
    little_mean = K_i * (K_ii \ y_i);
    mean_LTO(i) = little_mean(2) - little_mean(1);
    little_K = K([i,i+1],[i,i+1]) - K_i * (K_ii \ K_i');
    var_LTO(i) =little_K(1,1) + little_K(2,2) - 2 * little_K(1,2);
    standard = ((y(i+1)-y(i)) - mean_LTO(i)) ./ sqrt(var_LTO(i));
    
    p_point_LTO(i) = normcdf(standard);
    
    X_LTO(i) = mean(X(i:i+1));
    y_LTO(i) = y(i+1) - y(i);
end

plot(X_LTO,y_LTO,'o');
hold on;
plot(X_LTO,mean_LTO,'b-');
plot(X_LTO,mean_LTO+2*sqrt(var_LTO),'b--');
plot(X_LTO,mean_LTO-2*sqrt(var_LTO),'b--');
hold off;

%% Plot p values

plot(X_LTO, p_point_LTO, 'o');

%% Plot qq plot of p values

qqplot(p_point_LTO, linspace(0,1,10000));

%% Leave two next to each other out statistic - squared?

p_point_LTO = nan(length(X)-1,1);
mean_LTO = nan(length(X)-1,1);
var_LTO = nan(length(X)-1,1);
X_LTO = nan(length(X)-1,1);
y_LTO = nan(length(X)-1,1);

for i = 1:(length(X) - 1)
    not_i = [1:(i-1),(i+2):length(X)];
    
    K_ii = K(not_i,not_i);
    K_i = K([i;i+1],not_i);
    y_i = y(not_i);
    
    little_mean = K_i * (K_ii \ y_i);
    mean_LTO(i) = little_mean(2) - little_mean(1);
    little_K = K([i,i+1],[i,i+1]) - K_i * (K_ii \ K_i');
    var_LTO(i) =little_K(1,1) + little_K(2,2) - 2 * little_K(1,2);
    standard = (((y(i+1)-y(i)) - mean_LTO(i)) ./ sqrt(var_LTO(i))) .^ 2;
    
    p_point_LTO(i) = chi2cdf(standard,1);
    
    X_LTO(i) = mean(X(i:i+1));
    y_LTO(i) = y(i+1) - y(i);
end

plot(X_LTO,y_LTO,'o');
hold on;
plot(X_LTO,mean_LTO,'b-');
plot(X_LTO,mean_LTO+2*sqrt(var_LTO),'b--');
plot(X_LTO,mean_LTO-2*sqrt(var_LTO),'b--');
hold off;

%% Plot p values

plot(X_LTO, p_point_LTO, 'o');

%% Plot qq plot of p values

qqplot(p_point_LTO, linspace(0,1,10000));

%% Some correlated Gaussians

sigma = 0.9;
K = [1, sigma; sigma, 1];
L = chol(K);

samples = L' * randn(2, 100);
X = samples(1, :)';
Y = samples(2, :)';

plot(X, Y, 'o');

%% Distirbution of standardised variables

X_standard = X - sigma*Y;
Y_standard = Y - sigma*X;

plot(X_standard, Y, 'o');
plot(X_standard, X_standard, 'o');

%% One step ahead residuals

p_point_OSA = nan(size(X));
mean_OSA = nan(size(X));
var_OSA = nan(size(X));
for i = 1:length(X)
    not_close = X < X(i);
    
    if i > 1
        K_ii = K(not_close,not_close);
        K_i = K(i,not_close);
        y_i = y(not_close);

        mean_OSA(i) = K_i * (K_ii \ y_i);
        var_OSA(i) = K(i,i) - K_i * (K_ii \ K_i');
    else
        mean_OSA(i) = 0;
        var_OSA(i) = K(i,i);
    end
    standard = (y(i) - mean_OSA(i)) ./ sqrt(var_OSA(i));
    
    p_point_OSA(i) = normcdf(standard);
end

plot(X,y,'o');
hold on;
plot(X,mean_OSA,'b-');
plot(X,mean_OSA+2*sqrt(var_OSA),'b--');
plot(X,mean_OSA-2*sqrt(var_OSA),'b--');
ylim([min(y)-0.1*(max(y)-min(y)), max(y)+0.1*(max(y)-min(y))]);
hold off;

%% Plot p values

plot(X, p_point_OSA, 'o');

%% Plot qq plot of p values

qqplot(p_point_OSA, linspace(0,1,10000));