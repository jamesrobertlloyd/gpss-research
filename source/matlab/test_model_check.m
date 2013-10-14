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
hyp.cov = [0,0,log(std(y) / 10)];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likDelta;
hyp.lik = [];

hyp = minimize(hyp, @gp, -1000, @infDelta, mean_func, cov_func, lik_func, X, y);

K = feval(cov_func{:}, hyp.cov, X);

%% Fit SE + SE

cov_func = {@covSum, {@covSEiso, @covSEiso, @covNoise}};
hyp.cov = [0,0,-1,-1,log(std(y) / 10)];

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