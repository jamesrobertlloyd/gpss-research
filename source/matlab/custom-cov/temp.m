%% IMT1 fit

load '02-solar.mat'

%X = X - mean(X);
%X = X / std(X);
y = y - min(y);
y = y / std(y);

x = X;

cov_func = {@covSum, {@covIMT1, @covIMT1, @covConst}};
hyp.cov = [0, mean(x), -1, 5, mean(x), -4, -1.4];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

hyp = minimize(hyp, @gp, -10, @infExact, mean_func, cov_func, lik_func, x, y);

xrange = linspace(min(x)-100, max(x)+1000, 10000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;