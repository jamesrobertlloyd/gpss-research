%% Generate data from SE + SE + SE

X = [linspace(0,10,1000)'; linspace(20,30,1000)'];
X_test = linspace(-0,30,10000)';
cov_func = {@covSum, {@covSEiso,@covSEiso,@covSEiso,@covNoise}};
hyp.cov = [1,0,0,-0.5,-1,-1,-4];
%cov_func = {@covSum, {@covSEiso,@covNoise}};
%hyp.cov = [-1,0,-2];

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

%% Init

total_fit_test = zeros(size(X_test));
inv_lengthscale = 0.01;
sinks = 250;
resid = big_y;

%%

for iter = 1:70
    %% Fit with a kitchen sink

    Z = inv_lengthscale*randn(size(big_X,2),sinks);
    %Z = inv_lengthscale*trnd(1,size(big_X,2),sinks);
    phi = cos(big_X * Z) / sqrt(sinks);
    phi_test = cos(X_test * Z) / sqrt(sinks);
    SNR = 0.05;
    p = (phi' * phi + SNR^(-2)*eye(sinks)) \ (phi' * resid);
    %p = (phi' * phi + SNR^(-2)*eye(sinks)) \ (phi' * big_y);
    fit = phi * p;
    fit_test = phi_test * p;
    total_fit_test = total_fit_test + fit_test;
    %total_fit_test =  fit_test;
    resid = resid - fit;

    %% Move on

    inv_lengthscale = inv_lengthscale * 1.1;
    sinks = sinks + 25;

    %% Plot

    plot(big_X, big_y, 'o');
    hold on;
    plot(X_test, total_fit_test, 'ro');
    hold off;
    drawnow;
end