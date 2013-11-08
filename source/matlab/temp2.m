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

%% Intialise

fit_1 = zeros(size(big_y));
fit_2 = zeros(size(big_y));
fit_3 = zeros(size(big_y));

%% Backfit

for iter = 1:10

    %% Long lengthscale

    sinks = 1000;
    Z = 0.1*randn(size(big_X,2),sinks);
    phi = cos(big_X * Z) / sqrt(sinks);
    phi_test = cos(X_test * Z) / sqrt(sinks);
    SNR = 0.1;
    resid = big_y - fit_2 - fit_3;
    p = (phi' * phi + SNR^(-2)*eye(sinks)) \ (phi' * big_y);
    fit_1 = phi * p;
    fit_1_test = phi_test * p;

    %% Medium lengthscale

    sinks = 1000;
    Z = 1.0*randn(size(big_X,2),sinks);
    phi = cos(big_X * Z) / sqrt(sinks);
    phi_test = cos(X_test * Z) / sqrt(sinks);
    SNR = 0.1;
    resid = big_y - fit_1 - fit_3;
    p = (phi' * phi + SNR^(-2)*eye(sinks)) \ (phi' * big_y);
    fit_2 = phi * p;
    fit_2_test = phi_test * p;

    %% Short lengthscale

    sinks = 1000;
    Z = 2.5*randn(size(big_X,2),sinks);
    phi = cos(big_X * Z) / sqrt(sinks);
    phi_test = cos(X_test * Z) / sqrt(sinks);
    SNR = 0.1;
    resid = big_y - fit_1 - fit_2;
    p = (phi' * phi + SNR^(-2)*eye(sinks)) \ (phi' * big_y);
    fit_3 = phi * p;
    fit_3_test = phi_test * p;

    %% Plot total fit

    plot(X, y, 'o');
    hold on;
    plot(X_test, fit_1_test + fit_2_test + fit_3_test, 'ro');
    hold off;
    
    drawnow;
end