%% Generate data from SE + SE + SE

X = [linspace(0,10,1000)'; linspace(20,30,1000)'];
X_test = linspace(-0,30,10000)';
%cov_func = {@covSum, {@covSEiso,@covSEiso,@covSEiso,@covNoise}};
%hyp.cov = [0,0,-0.5,-0.5,-1,-1,-2];
cov_func = {@covSum, {@covSEiso,@covNoise}};
hyp.cov = [0,0,-1];

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

inv_lengthscale = 0.01;
sinks = 50;
phi = [];
phi_test = [];

%%

for iter = 1:50
    %% Fit with a kitchen sink

    Z = inv_lengthscale*randn(size(big_X,2),sinks);
    phi = [phi, cos(big_X * Z)];
    phi_test = [phi_test, cos(X_test * Z)];
    SNR = 0.1;
    p = (phi' * phi + SNR^(-2)*eye(size(phi,2))) \ (phi' * big_y);
    fit = phi * p;
    fit_test = phi_test * p;

    %% Move on

    inv_lengthscale = inv_lengthscale * 1.1;

    %% Plot

    plot(big_X, big_y, 'o');
    hold on;
    plot(X_test, fit_test, 'ro');
    hold off;
    drawnow;
end