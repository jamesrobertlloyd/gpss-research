%% Generate data from SE

X = linspace(0,1,100)';
cov_func = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [0,0,-5];

K = feval(cov_func{:}, hyp.cov, X);

y = chol(K)' * randn(size(X));

% %% Plot
% 
% plot(X, y, 'o');
% 
% %% Fit SE
% 
% cov_func = {@covSum, {@covSEiso, @covNoise}};
% hyp.cov = [-2,0,log(std(y) / 10)];
% 
% cov_func_1 = @covSEiso;
% cov_func_2 = @covNoise;
% 
% mean_func = @meanZero;
% hyp.mean = [];
% 
% lik_func = @likDelta;
% hyp.lik = [];
% 
% hyp = minimize(hyp, @gp, -1000, @infDelta, mean_func, cov_func, lik_func, X, y);
% 
% K = feval(cov_func{:}, hyp.cov, X);

% %% Plot extrapolation
% 
% X_extrap = linspace(1,5,500)';
% K_star = feval(cov_func{:}, hyp.cov, X, X_extrap);
% K_starstar = feval(cov_func{:}, hyp.cov, X_extrap);
% 
% mu = K_star' * (K \ y);
% sigma = K_starstar - K_star' * (K \ K_star);
% var = diag(sigma);
% 
% plot(X,y,'o');
% hold on;
% plot(X_extrap, mu, 'b-');
% plot(X_extrap, mu+2*sqrt(var), 'b--');
% plot(X_extrap, mu-2*sqrt(var), 'b--');
% hold off;

%% Plot increase in uncertainty

X_extrap = linspace(1,5,500)';
K_star = feval(cov_func{:}, hyp.cov, X, X_extrap);
K_starstar = feval(cov_func{:}, hyp.cov, X_extrap);

mu = K_star' * (K \ y);
sigma = K_starstar - K_star' * (K \ K_star);
var = diag(sigma);

plot(X_extrap, sqrt(var), 'b-');