%% PRNG

seed=2;   % fixing the seed of the random generators
randn('state',seed); %#ok<RAND>
rand('state',seed); %#ok<RAND>

%% Setup x axis

xrange = linspace(0, 1, 500)';
x_data = linspace(0.1, 0.9, 5)';

x_data = [x_data; x_data; x_data];
x_data = x_data + 0.05 * randn(size(x_data));
x_data = x_data(randperm(numel(x_data)));
x_data = x_data(15:-1:1);

%% Plot some points

y = x_data + 0.1 * randn(size(x_data));

figure(1);

mean_var_plot(x_data, y, xrange, [], [], true);
xlim([0,1]);
ylim([0,1]);

save2pdf([ 'lin_reg/' 'all_data' '.pdf'], gcf, 600, true);
pause(0.01);
drawnow;

%% Least squares

figure(2);

m_hat = ((x_data'*x_data) \ x_data') * y;

mean_var_plot(x_data, y, xrange, xrange*m_hat, zeros(size(xrange)));
xlim([0,1]);
ylim([0,1]);

save2pdf([ 'lin_reg/' 'least_squares' '.pdf'], gcf, 600, true);
pause(0.01);
drawnow;

%% Bayesian linear prior
    
cov_all = {@covSum, {@covLINscaleshift, @covNoise}};
cov_fn = {@covLINscaleshift};
hyp.cov = [0,0,log(0.01)];

K = feval(cov_all{:}, hyp.cov, xrange);
prior_var = diag(K);

figure(3);

mean_var_plot(x_data, y, xrange, zeros(size(xrange)), 2*sqrt(prior_var), false, true);
xlim([0,1]);
ylim([-2,2]);

save2pdf([ 'lin_reg/' 'prior' '.pdf'], gcf, 600, true);
pause(0.01);
drawnow;

hyp.cov = [0,0,log(0.2)];

for i = 1:numel(x_data)
    x_data_subset = x_data(1:i);
    y_subset = y(1:i);
    
    K = feval(cov_all{:}, hyp.cov, x_data_subset);
    K_star = feval(cov_fn{:}, hyp.cov, x_data_subset, xrange);
    K_starstar = feval(cov_fn{:}, hyp.cov, xrange);
    
    mu = K_star' / K * y_subset;
    post_var = diag(K_starstar - K_star' / K * K_star);
    
    figure(3+i);
    
    mean_var_plot(x_data_subset, y_subset, xrange, mu, 2*sqrt(post_var));
    xlim([0,1]);
    ylim([0,1]);

    save2pdf([ 'lin_reg/' 'bayes_' int2str(i) '.pdf'], gcf, 600, true);
    pause(0.01);
    drawnow;
end

%% Close all

close all;