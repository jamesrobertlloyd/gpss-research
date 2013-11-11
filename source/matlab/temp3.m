%% Generate data from SE + SE + SE

X = [linspace(0,10,1000)'; linspace(20,30,1000)'];
X_test = linspace(-0,30,10000)';
cov_func = {@covSum, {@covSEiso,@covSEiso,@covSEiso,@covNoise}};
hyp.cov = [0,0,-0.5,-0.5,-1,-1,-2];

K = feval(cov_func{:}, hyp.cov, X);

y = chol(K)' * randn(size(X));

%% More data

big_X = [];
big_y = [];
for i = 1:10
    big_X = [big_X; X + 0.01*randn(size(X))];
    big_y = [big_y; y + 0.01*randn(size(y))];
end

%% Plot

plot(big_X, big_y, 'o');

%% Fit with a kitchen sink

sinks = 2000;
Z = 1*randn(size(big_X,2),sinks);
%Z = 0.1*trnd(1,size(big_X,2),sinks);
SNR = 0.1;
phi = 1 * cos(big_X * Z) / sqrt(sinks);
phi_test = 1 * cos(X_test * Z) / sqrt(sinks);
p = (phi' * phi + SNR^(-2)*eye(sinks)) \ (phi' * big_y);
big_y_fit = phi * p;
big_y_fit_test = phi_test * p;
resid = big_y - big_y_fit;

%% Plot the fit

plot(big_X, big_y, 'o');
hold on;
plot(X_test, big_y_fit_test, 'ro');
hold off;

%% Something else entirely

plot(double(actuals), 'g');
hold on;
plot(double(predictions), 'r');
hold off;