%% PRNG

seed=3;   % fixing the seed of the random generators
randn('state',seed); %#ok<RAND>
rand('state',seed); %#ok<RAND>

%% Misc setup

fig_count = 1;

%% Setup x axis

xrange = linspace(0, 1, 500)';
x_data = linspace(0.1, 0.9, 5)';

x_data = [x_data; x_data; x_data];
x_data = x_data + 0.05 * randn(size(x_data));
x_data = x_data(randperm(numel(x_data)));
x_data = x_data(15:-1:1);

%% Plot some points

y = x_data + 0.1 * randn(size(x_data));

figure(fig_count);
fig_count = fig_count + 1;

mean_var_plot(x_data, y, xrange, [], [], true);
xlim([0,1]);
ylim([0,1]);

pause(0.5);
drawnow;
save2pdf([ 'lin_reg/' 'all_data' '.pdf'], gcf, 600, true);

%% Least squares

figure(fig_count);
fig_count = fig_count + 1;

m_hat = ((x_data'*x_data) \ x_data') * y;

mean_var_plot(x_data, y, xrange, xrange*m_hat, zeros(size(xrange)));
xlim([0,1]);
ylim([0,1]);

pause(0.5);
drawnow;
save2pdf([ 'lin_reg/' 'least_squares' '.pdf'], gcf, 600, true);

%% Bayesian linear prior
    
cov_all = {@covSum, {@covLINscaleshift, @covNoise}};
cov_fn = {@covLINscaleshift};
hyp.cov = [0,0,log(0.2)];

K = feval(cov_fn{:}, hyp.cov, xrange);
%prior_var = diag(K);

figure(fig_count);
fig_count = fig_count + 1;

%mean_var_plot(x_data, y, xrange, zeros(size(xrange)), 2*sqrt(prior_var), false, true);
samples_density_plot(x_data, y, xrange, zeros(size(xrange)), K, false, true);
xlim([0,1]);
ylim([-2,2]);

pause(0.5);
drawnow;
save2pdf([ 'lin_reg/' 'prior' '.pdf'], gcf, 600, true);

hyp.cov = [0,0,log(0.2)];

for i = 1:numel(x_data)
    x_data_subset = x_data(1:i);
    y_subset = y(1:i);
    
    K = feval(cov_all{:}, hyp.cov, x_data_subset);
    K_star = feval(cov_fn{:}, hyp.cov, x_data_subset, xrange);
    K_starstar = feval(cov_fn{:}, hyp.cov, xrange);
    
    mu = K_star' / K * y_subset;
    %post_var = diag(K_starstar - K_star' / K * K_star);
    post_K = K_starstar - K_star' / K * K_star;
    
    figure(fig_count);
    fig_count = fig_count + 1;
    
    samples_density_plot(x_data_subset, y_subset, xrange, mu, post_K);
    xlim([0,1]);
    ylim([0,1]);

    pause(0.5);
    drawnow;
    save2pdf([ 'lin_reg/' 'bayes_' int2str(i) '.pdf'], gcf, 600, true);
end

%% Sq exp prior
    
cov_all = {@covSum, {@covSEiso, @covNoise}};
cov_fn = {@covSEiso};
hyp.cov = [-1,0,log(0.1)];

K = feval(cov_fn{:}, hyp.cov, xrange);
%prior_var = diag(K);

figure(fig_count);
fig_count = fig_count + 1;

%mean_var_plot(x_data, y, xrange, zeros(size(xrange)), 2*sqrt(prior_var), false, true);
samples_density_plot(x_data, y, xrange, zeros(size(xrange)), K, false, true);
xlim([0,1]);
ylim([-2,2]);

pause(0.5);
drawnow;
save2pdf([ 'lin_reg/' 'sq_exp_prior' '.pdf'], gcf, 600, true);

hyp.cov = [0,0,log(0.1)];

for i = 1:numel(x_data)
    x_data_subset = x_data(1:i);
    y_subset = y(1:i);
    
    K = feval(cov_all{:}, hyp.cov, x_data_subset);
    K_star = feval(cov_fn{:}, hyp.cov, x_data_subset, xrange);
    K_starstar = feval(cov_fn{:}, hyp.cov, xrange);
    
    mu = K_star' / K * y_subset;
    %post_var = diag(K_starstar - K_star' / K * K_star);
    post_K = K_starstar - K_star' / K * K_star;
    
    figure(fig_count);
    fig_count = fig_count + 1;
    
    samples_density_plot(x_data_subset, y_subset, xrange, mu, post_K);
    xlim([0,1]);
    ylim([0,1]);

    pause(0.5);
    drawnow;
    save2pdf([ 'lin_reg/' 'sq_exp_' int2str(i) '.pdf'], gcf, 600, true);
end

%% Sample some quadratic data

y = 0.15 + 4 * (x_data - 0.5) .^ 2 + 0.1 * randn(size(x_data));

figure(fig_count);
fig_count = fig_count + 1;

mean_var_plot(x_data, y, xrange, [], [], true);
xlim([0,1]);
ylim([0,1]);

pause(0.5);
drawnow;
save2pdf([ 'quad/' 'all_data' '.pdf'], gcf, 600, true);

%% Bayesian linear prior
    
cov_all = {@covSum, {@covLINscaleshift, @covNoise}};
cov_fn = {@covLINscaleshift};
hyp.cov = [-1,0,log(0.01)];

K = feval(cov_fn{:}, hyp.cov, xrange);
%prior_var = diag(K);

figure(fig_count);
fig_count = fig_count + 1;

%mean_var_plot(x_data, y, xrange, zeros(size(xrange)), 2*sqrt(prior_var), false, true);
samples_density_plot(x_data, y, xrange, zeros(size(xrange)), K, false, true);
xlim([0,1]);
ylim([-2,2]);

pause(0.5);
drawnow;
save2pdf([ 'quad/' 'prior' '.pdf'], gcf, 600, true);

hyp.cov = [0,0,log(0.2)];

for i = 1:numel(x_data)
    x_data_subset = x_data(1:i);
    y_subset = y(1:i);
    
    K = feval(cov_all{:}, hyp.cov, x_data_subset);
    K_star = feval(cov_fn{:}, hyp.cov, x_data_subset, xrange);
    K_starstar = feval(cov_fn{:}, hyp.cov, xrange);
    
    mu = K_star' / K * y_subset;
    %post_var = diag(K_starstar - K_star' / K * K_star);
    post_K = K_starstar - K_star' / K * K_star;
    
    figure(fig_count);
    fig_count = fig_count + 1;
    
    samples_density_plot(x_data_subset, y_subset, xrange, mu, post_K);
    xlim([0,1]);
    ylim([0,1]);

    pause(0.5);
    drawnow;
    save2pdf([ 'quad/' 'bayes_' int2str(i) '.pdf'], gcf, 600, true);
end

%% Sq exp prior
    
cov_all = {@covSum, {@covSEiso, @covNoise}};
cov_fn = {@covSEiso};
hyp.cov = [0,0,log(0.1)];

K = feval(cov_fn{:}, hyp.cov, xrange);
%prior_var = diag(K);

figure(fig_count);
fig_count = fig_count + 1;

%mean_var_plot(x_data, y, xrange, zeros(size(xrange)), 2*sqrt(prior_var), false, true);
samples_density_plot(x_data, y, xrange, zeros(size(xrange)), K, false, true);
xlim([0,1]);
ylim([-2,2]);

pause(0.5);
drawnow;
save2pdf([ 'quad/' 'sq_exp_prior' '.pdf'], gcf, 600, true);

hyp.cov = [0,0,log(0.1)];

for i = 1:numel(x_data)
    x_data_subset = x_data(1:i);
    y_subset = y(1:i);
    
    K = feval(cov_all{:}, hyp.cov, x_data_subset);
    K_star = feval(cov_fn{:}, hyp.cov, x_data_subset, xrange);
    K_starstar = feval(cov_fn{:}, hyp.cov, xrange);
    
    mu = K_star' / K * y_subset;
    %post_var = diag(K_starstar - K_star' / K * K_star);
    post_K = K_starstar - K_star' / K * K_star;
    
    figure(fig_count);
    fig_count = fig_count + 1;
    
    samples_density_plot(x_data_subset, y_subset, xrange, mu, post_K);
    xlim([0,1]);
    ylim([0,1]);

    pause(0.5);
    drawnow;
    save2pdf([ 'quad/' 'sq_exp_' int2str(i) '.pdf'], gcf, 600, true);
end

%% Close all

close all;