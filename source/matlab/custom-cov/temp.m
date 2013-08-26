x = linspace(-15, 15, 500)';
y = cos(1*pi*x) .* (1- (max(0,x)>0)) + cos(3*pi*x) .* ((max(0,x)>0)) + 0.1*randn(size(x)) + max(0,x);

cov_func = {@covChangePointLin, {{@covProd, {@covPeriodic, @covSEiso}}, {@covProd, {@covPeriodic, @covSEiso}}}};
hyp.cov = [-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

mean_func = @meanZero;
hyp.mean = [];

lik_func = @likGauss;
hyp.lik = log(std(y-mean(y)) / 10);

repeats = 1;
total_iters = 50;
for i = 1:repeats
  hyp = minimize(hyp, @gp, -floor(total_iters/repeats), @infExact, mean_func, cov_func, lik_func, x, y);
end

xrange = linspace(min(x)-5, max(x)+5, 1000)';

fit = gp(hyp, @infExact, mean_func, cov_func, lik_func, x, y, xrange);

plot(x, y, 'o');
hold on;
plot(xrange, fit);
hold off;