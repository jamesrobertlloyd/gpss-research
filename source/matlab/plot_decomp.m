function plot_decomp(X, y, mean_family, mean_params, complete_covfunc, complete_hypers, decomp_list, ...
                     decomp_hypers, lik_family_, lik_params, figname, latex_names, ...
                     full_name, X_mean, X_scale, y_mean, y_scale, max_depth)
                 
%%%%%%%%%%%
% WARNING %
%%%%%%%%%%%
% - Ignores mean
% - Ignores lik - assumes provides a noise parameter
if length(lik_params) == 0
    log_noise = -inf;
else
    log_noise = lik_params(0);
end
%%%%%%%%%%%
% WARNING %
%%%%%%%%%%%

% TODO: Assert that the sum of all kernels is the same as the complete kernel.

if nargin < 18; max_depth = numel(decomp_list); end
% if nargin < 15; max_depth = 4; end

% Convert to double in case python saved as integers
X = double(X);
y = double(y);

%%%% TODO - function should accept a mean function
%y = y - mean(y);

left_extend = 0.1;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';
xrange_no_extrap = linspace(min(X), max(X), num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X) + eye(length(y)).*noise_var;
%complete_sigma = non_singular(complete_sigma);
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange);

% First, plot the data
complete_mean = complete_sigmastar' / complete_sigma * y;
complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
posterior_sigma = complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar;
%posterior_sigma = non_singular(posterior_sigma);
    
figure(1); clf; hold on;
mean_var_plot( X*X_scale+X_mean, y*y_scale+y_mean, ...
               xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, ...
               2.*sqrt(complete_var)*y_scale, false, true); % Only plot the data


title('Raw data');
filename = sprintf('%s_raw_data.fig', figname);
saveas( gcf, filename );

% Now plot the posterior
figure(2); clf; hold on;
mean_var_plot( X*X_scale+X_mean, y*y_scale+y_mean, ...
               xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, ...
               2.*sqrt(complete_var)*y_scale, false, false);

% Remove outer brackets and extra latex markup from name.
if iscell(full_name); full_name = full_name{1}; end
full_name = strrep(full_name, '\left', '');
full_name = strrep(full_name, '\right', '');
%title(full_name);
title('Full model posterior with extrapolations');
filename = sprintf('%s_all.fig', figname);
saveas( gcf, filename );

% Now plot samples from the posterior
figure(3); clf; hold on;
sample_plot( X*X_scale+X_mean, xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, ...
               posterior_sigma);

% Remove outer brackets and extra latex markup from name.
if iscell(full_name); full_name = full_name{1}; end
full_name = strrep(full_name, '\left', '');
full_name = strrep(full_name, '\right', '');
%title(full_name);
title('Random samples from the full model posterior');
filename = sprintf('%s_all_sample.fig', figname);
saveas( gcf, filename );

% Then plot the same thing, but just the end.
% complete_mean = complete_sigmastar' / complete_sigma * y;
% complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
%     
% figure(100); clf; hold on;
% mean_var_plot(X*X_scale+X_mean, y*y_scale+y_mean, xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, 2.*sqrt(complete_var)*y_scale, true, false);
% title(full_name);
% filename = sprintf('%s_all_small.fig', figname);
% saveas( gcf, filename );

% Plot residuals.
% figure(1000); clf; hold on;
% data_complete_mean = feval(complete_covfunc{:}, complete_hypers, X, X)' / complete_sigma * y;
% std_ratio = std((y-data_complete_mean)) / sqrt(noise_var);
% mean_var_plot(X*X_scale+X_mean, (y-data_complete_mean)*y_scale, ...
%               xrange*X_scale+X_mean, zeros(size(xrange)), ...
%               2.*sqrt(noise_var).*ones(size(xrange)).*y_scale, false, true);
% title(['Residuals']);
% filename = sprintf('%s_resid.fig', figname);
% saveas( gcf, filename );

% Determine the order to diaplay the components by computing cross validated MAEs

MAEs = NaN(numel(decomp_list), 1);

folds = 10;

X_train = cell(folds,1);
y_train = cell(folds,1);
X_valid = cell(folds,1);
y_valid = cell(folds,1);

%%%% TODO - Check me for overlap

for fold = 1:folds
    range = max(1,floor(length(X)*(fold-1)/folds)):floor(length(X)*(fold)/folds);
    X_valid{fold} = X(range);
    y_valid{fold} = y(range);
    range = [1:min(length(X),floor(length(X)*(fold-1)/folds)-1),...
            max(1,floor(length(X)*(fold)/folds)+1):length(X)];
    X_train{fold} = X(range);
    y_train{fold} = y(range);
end

idx = [];

cum_kernel = cell(0);
cum_hyp = [];

% Precompute some kernels

K_list = cell(numel(decomp_list), 1);
for i = 1:numel(decomp_list)
    cur_cov = decomp_list{i};
    cur_hyp = decomp_hypers{i};
    K_list{i} = feval(cur_cov{:}, cur_hyp, X, X);
end

% Determine if some components are very similar

% component_corr = zeros(numel(decomp_list));
% for i = 1:numel(decomp_list)
%     for j = (i+1):numel(decomp_list)
%         component_corr(i,j) = -mean(diag(K_list{i}*(complete_sigma\K_list{j}))./sqrt(abs(diag(K_list{i} - K_list{i}*(complete_sigma\K_list{i})).*diag(K_list{j} - K_list{j}*(complete_sigma\K_list{j})))));
%     end
% end
% 
% i = 1;
% while i <= numel(decomp_list)
%     j = (i+1);
%     while j <= numel(decomp_list)
%         if component_corr(i, j) < -0.8
%             % Components v. sim. - remove
%             new_idx = [1:(j-1),(j+1):numel(decomp_list)];
%             component_corr = component_corr(new_idx, new_idx);
%             decomp_list{i} = {@covSum, {decomp_list{i}, decomp_list{j}}};
%             decomp_list = decomp_list(new_idx);
%             decomp_hypers{i} = [decomp_hypers{i}, decomp_hypers{j}];
%             decomp_hypers = decomp_hypers(new_idx);
%         else
%             j = j + 1;
%         end
%     end
%     i = i + 1;
% end

MAEs = zeros(numel(decomp_list), 1);
MAE_reductions = zeros(numel(decomp_list), 1);
MAV_data = mean(abs(y));
previous_MAE = MAV_data;

for i = 1:min(numel(decomp_list), max_depth)
    best_MAE = Inf;
    for j = 1:numel(decomp_list)
        if ~sum(j == idx)
            kernels = cum_kernel;
            kernels{i} = decomp_list{j};
            hyps = cum_hyp;
            hyps = [hyps, decomp_hypers{j}];
            hyp.mean = [];
            hyp.cov = hyps;
            cur_cov = {@covSum, kernels};
            e = NaN(length(X_train), 1);
            for fold = 1:length(X_train)
              K = feval(complete_covfunc{:}, complete_hypers, X_train{fold}) + ...
                  noise_var*eye(length(y_train{fold}));
              Ks = feval(cur_cov{:}, hyp.cov, X_train{fold}, X_valid{fold});

              ymu = Ks' * (K \ y_train{fold});

              e(fold) = mean(abs(y_valid{fold} - ymu));
            end
            
            my_MAE = mean(e);
            if my_MAE < best_MAE
                best_j  = j;
                best_MAE = my_MAE;
            end
        end
    end
    MAEs(i) = best_MAE;
    MAE_reductions(i) = (1 - best_MAE / previous_MAE)*100;
    previous_MAE = best_MAE;
    idx = [idx, best_j];
    cum_kernel{i} = decomp_list{best_j};
    cum_hyp = [cum_hyp, decomp_hypers{best_j}];
end

% Plot each component without data

SNRs = zeros(numel(decomp_list),1);
vars = zeros(numel(decomp_list),1);
monotonic = zeros(numel(decomp_list),1);
gradients = zeros(numel(decomp_list),1);
qq_d_max_p = zeros(numel(decomp_list),1);
qq_d_min_p = zeros(numel(decomp_list),1);
acf_min_p = zeros(numel(decomp_list),1);
acf_min_loc_p = zeros(numel(decomp_list),1);
pxx_max_p = zeros(numel(decomp_list),1);
pxx_max_loc_p = zeros(numel(decomp_list),1);

for j = 1:min(numel(decomp_list), max_depth)
    i = idx(j);
    cur_cov = decomp_list{i};
    cur_hyp = decomp_hypers{i};
    
    % Compute mean and variance for this kernel.
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange_no_extrap);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange_no_extrap);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
    data_mean = decomp_sigma' / complete_sigma * y;
    diffs = data_mean(2:end) - data_mean(1:(end-1));
    data_covar = decomp_sigma - decomp_sigma' / complete_sigma * decomp_sigma;
    data_var = diag(data_covar);
    SNRs(j) = 10 * log10(sum(data_mean.^2)/sum(data_var));
    vars(j) = (1 - var(y - data_mean) / var(y)) * 100;
    if all(diffs>0)
        monotonic(j) = 1;
    elseif all(diffs<0)
        monotonic(j) = -1;
    else
        monotonic(j) = 0;
    end
    gradients(j) = (data_mean(end) - data_mean(1)) / (X(end) - X(1));
    
    % Compute the remaining signal after removing the mean prediction from all
    % other parts of the kernel.
    removed_mean = y - (complete_sigma - decomp_sigma)' / complete_sigma * y;
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X*X_scale+X_mean, removed_mean*y_scale, ...
                   xrange_no_extrap*X_scale+X_mean, ...
                   decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale, false, false, true); % Don't plot data
    
    %set(gca, 'Children', [h_bars, h_mean, h_dots] );
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    %title(latex_names{i});
    title(sprintf('Posterior of component %d', j));
    fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d.fig', figname, j);
    saveas( gcf, filename );
    
    % Plot sample qq
    
    samples = 1000;
    
    many_prior_samples = chol(non_singular(decomp_sigma))' * ...
                    randn(length(y), 100);% ./ ...
                    %repmat(sqrt(diag(decomp_sigma)), 1, samples);
    q_values = linspace(0,1,length(y)+2)';
    q_values = q_values(2:(end-1));
    prior_quantiles = quantile(many_prior_samples(:), q_values);
    prior_samples = chol(non_singular(decomp_sigma))' * ...
                    randn(length(y), samples);% ./ ...
                    %repmat(sqrt(diag(decomp_sigma)), 1, samples);
    prior_qq = zeros(length(y), samples);
    %repeated_samples = repmat(many_prior_samples(:)', length(y), 1);
    %repeated_samples = repeated_samples + ... % To break ties
    %                   0.001*max(max(many_prior_samples))*randn(size(repeated_samples));
    prior_qq_d_max = zeros(samples, 1);
    prior_qq_d_min = zeros(samples, 1);
    for iter = 1:samples
        %a = normcdf(sort(prior_samples(:,iter)));
        a = sort(prior_samples(:,iter));
        %a = sum(repmat(sort(prior_samples(:,iter)), 1, length(repeated_samples)) > repeated_samples, 2) ./ length(repeated_samples);
        prior_qq(:,iter) = a;
        differences = a - prior_quantiles;
        prior_qq_d_max(iter) = differences(differences == max(differences));
        prior_qq_d_min(iter) = differences(differences == min(differences));
    end

    post_samples = (repmat(data_mean, 1, samples) + chol(non_singular(data_covar))' * ...
                    randn(length(y), samples));% ./ ...
                   %repmat(sqrt(diag(data_covar)), 1, samples);
    post_qq = zeros(length(y), samples);
    post_qq_d_max = zeros(samples, 1);
    post_qq_d_min = zeros(samples, 1);
    for iter = 1:samples
        %a = normcdf(sort(post_samples(:,iter)));
        a = sort(post_samples(:,iter));
        %a = sum(repmat(sort(post_samples(:,iter)), 1, length(repeated_samples)) > repeated_samples, 2) ./ length(repeated_samples);
        post_qq(:,iter) = a;
        differences = a - prior_quantiles;
        post_qq_d_max(iter) = differences(differences == max(differences));
        post_qq_d_min(iter) = differences(differences == min(differences));
    end
    
    % Is the value of the distance extreme?
    qq_d_max_p(j) = sum(prior_qq_d_max > post_qq_d_max + 0.0001*max(post_qq_d_max)*randn(size(post_qq_d_max))) / length(post_qq_d_max);
    qq_d_min_p(j) = sum(prior_qq_d_min < post_qq_d_min + 0.0001*max(post_qq_d_min)*randn(size(post_qq_d_min))) / length(post_qq_d_min);
    
    figure(i + 1); clf; hold on;
    two_band_plot(prior_quantiles, ...%linspace(0, 1, length(y))', ...
                  mean(prior_qq, 2), ...
                  quantile(prior_qq, 0.95, 2), ...
                  quantile(prior_qq, 0.05, 2), ...
                  mean(post_qq, 2), ...
                  quantile(post_qq, 0.95, 2), ...
                  quantile(post_qq, 0.05, 2));
    title(sprintf('QQ uncertainty plot for component %d', j));
    filename = sprintf('%s_qq_bands_%d.fig', figname, j);
    saveas( gcf, filename );              
    %hold off;
    
    % Compute kernel at grid to compute acf and periodogram
    
    x_grid = linspace(min(X), max(X), numel(X))';
    grid_distance = x_grid(2) - x_grid(1);
    
    complete_sigma_grid = feval(complete_covfunc{:}, complete_hypers, x_grid);
    decomp_sigma_grid = feval(cur_cov{:}, cur_hyp, x_grid);
    
    data_mean_grid = decomp_sigma_grid' / complete_sigma_grid * y;
    data_covar_grid = decomp_sigma_grid - decomp_sigma_grid' / complete_sigma_grid * decomp_sigma_grid;
    
    % Plot sample acf
    
    samples = 1000;
    
    prior_samples = chol(non_singular(decomp_sigma_grid))' * randn(length(y), samples);
    prior_acf = autocorr(prior_samples(:,1), length(y) -1);
    prior_acf = zeros(length(prior_acf), samples);
    prior_acf_min_loc = zeros(samples,1);
    prior_acf_min = zeros(samples,1);
    for iter = 1:samples
        prior_acf(:,iter) = autocorr(prior_samples(:,iter), length(y) -1);
        prior_acf_min_loc(iter) = find(prior_acf(:,iter) == min(prior_acf(:,iter)));
        prior_acf_min(iter) = min(prior_acf(:,iter));
    end

    post_samples = (repmat(data_mean_grid, 1, samples) + chol(non_singular(data_covar_grid))' * ...
                    randn(length(y), samples));
    post_acf = autocorr(post_samples(:,1), length(y) -1);
    post_acf = zeros(length(post_acf), samples);
    post_acf_min_loc = zeros(samples,1);
    post_acf_min = zeros(samples,1);
    for iter = 1:samples
        post_acf(:,iter) = autocorr(post_samples(:,iter), length(y) -1);
        post_acf_min_loc(iter) = find(post_acf(:,iter) == min(post_acf(:,iter)));
        post_acf_min(iter) = min(post_acf(:,iter));
    end
    
    % Are the values extreme?
    acf_min_p(j) = sum(prior_acf_min < post_acf_min + 0.0001*max(post_acf_min)*randn(size(post_acf_min))) / length(post_acf_min);
    acf_min_loc_p(j) = sum(prior_acf_min_loc < post_acf_min_loc + 0.0001*max(post_acf_min_loc)*randn(size(post_acf_min_loc))) / length(post_acf_min_loc);

    figure(i + 1); clf; hold on;
    two_band_plot((1:size(post_acf, 1))' * grid_distance, ...
                  mean(prior_acf, 2), ...
                  quantile(prior_acf, 0.95, 2), ...
                  quantile(prior_acf, 0.05, 2), ...
                  mean(post_acf, 2), ...
                  quantile(post_acf, 0.95, 2), ...
                  quantile(post_acf, 0.05, 2));
    title(sprintf('ACF uncertainty plot for component %d', j));
    ylabel('Correlation coefficient');
    xlabel('Lag');
    filename = sprintf('%s_acf_bands_%d.fig', figname, j);
    saveas( gcf, filename );
    
    % Plot sample periodogram
    
    samples = 1000;
    
    prior_samples = chol(non_singular(decomp_sigma_grid))' * randn(length(y), samples);
    prior_pxx = 10 * log10(periodogram(prior_samples(:,1)));
    prior_pxx = zeros(length(prior_pxx), samples);
    prior_pxx_max_loc = zeros(samples,1);
    prior_pxx_max = zeros(samples,1);
    for iter = 1:samples
        prior_pxx(:,iter) = 10 * log10(periodogram(prior_samples(:,iter)));
        prior_pxx_max_loc(iter) = find(prior_pxx(:,iter) == max(prior_pxx(:,iter)));
        prior_pxx_max(iter) = max(prior_pxx(:,iter));
    end

    post_samples = (repmat(data_mean_grid, 1, samples) + chol(non_singular(data_covar_grid))' * ...
                    randn(length(y), samples));
    post_pxx = 10 * log10(periodogram(post_samples(:,1)));
    post_pxx = zeros(length(post_pxx), samples);
    post_pxx_max_loc = zeros(samples,1);
    post_pxx_max = zeros(samples,1);
    for iter = 1:samples
        post_pxx(:,iter) = 10 * log10(periodogram(post_samples(:,iter)));
        post_pxx_max_loc(iter) = find(post_pxx(:,iter) == max(post_pxx(:,iter)));
        post_pxx_max(iter) = max(post_pxx(:,iter));
    end
    
    % Are the values extreme?
    pxx_max_p(j) = sum(prior_pxx_max < post_pxx_max + 0.0001*max(post_pxx_max)*randn(size(post_pxx_max))) / length(post_pxx_max);
    pxx_max_loc_p(j) = sum(prior_pxx_max_loc < post_pxx_max_loc + 0.0001*max(post_pxx_max_loc)*randn(size(post_pxx_max_loc))) / length(post_pxx_max_loc);

    figure(i + 1); clf; hold on;
    two_band_plot(linspace(0,1,size(prior_pxx,1))', ...
                  mean(prior_pxx, 2), ...
                  quantile(prior_pxx, 0.95, 2), ...
                  quantile(prior_pxx, 0.05, 2), ...
                  mean(post_pxx, 2), ...
                  quantile(post_pxx, 0.95, 2), ...
                  quantile(post_pxx, 0.05, 2));
    title(sprintf('Periodogram uncertainty plot for component %d', j));
    ylabel('Power / frequency');
    xlabel('Normalised frequency');
    filename = sprintf('%s_pxx_bands_%d.fig', figname, j);
    saveas( gcf, filename );    
    
    % Compute mean and variance for this kernel.
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_sigma_posterior = decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star;
    decomp_var = diag(decomp_sigma_posterior);
    
    data_mean = decomp_sigma' / complete_sigma * y;
    diffs = data_mean(2:end) - data_mean(1:(end-1));
    data_var = diag(decomp_sigma - decomp_sigma' / complete_sigma * decomp_sigma);
    SNRs(j) = 10 * log10(sum(data_mean.^2)/sum(data_var));
    vars(j) = (1 - var(y - data_mean) / var(y)) * 100;
    if all(diffs>0)
        monotonic(j) = 1;
    elseif all(diffs<0)
        monotonic(j) = -1;
    else
        monotonic(j) = 0;
    end
    gradients(j) = (data_mean(end) - data_mean(1)) / (X(end) - X(1));
    
    % Compute the remaining signal after removing the mean prediction from all
    % other parts of the kernel.
    removed_mean = y - (complete_sigma - decomp_sigma)' / complete_sigma * y;
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X*X_scale+X_mean, removed_mean*y_scale, ...
                   xrange*X_scale+X_mean, ...
                   decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale, false, false, true); % Don't plot data
    
    %set(gca, 'Children', [h_bars, h_mean, h_dots] );
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    %title(latex_names{i});
    title(sprintf('Posterior of component %d', j));
    %fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d_extrap.fig', figname, j);
    saveas( gcf, filename );
    
    figure(i + 1); clf; hold on;
    sample_plot( X*X_scale+X_mean, xrange*X_scale+X_mean, decomp_mean*y_scale, decomp_sigma_posterior )
    
    %set(gca, 'Children', [h_bars, h_mean, h_dots] );
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    %title(latex_names{i});
    title(sprintf('Random samples from the posterior of component %d', j));
    %fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d_sample.fig', figname, j);
    saveas( gcf, filename );
end

% Plot cumulative components with data

cum_kernel = cell(0);
cum_hyp = [];

var(y);
resid = y;

cum_SNRs = zeros(numel(decomp_list),1);
cum_vars = zeros(numel(decomp_list),1);
cum_resid_vars = zeros(numel(decomp_list),1);

anti_cum_kernel = cell(0);
anti_cum_hyp = [];
for j = 1:min(numel(decomp_list), max_depth)
    i = idx(j);
    anti_cum_kernel{j} = decomp_list{i};
    anti_cum_hyp = [anti_cum_hyp, decomp_hypers{i}];
end

for j = 1:min(numel(decomp_list), max_depth)
    i = idx(j);
    cum_kernel{j} = decomp_list{i};
    cum_hyp = [cum_hyp, decomp_hypers{i}];
    cur_cov = {@covSum, cum_kernel};
    cur_hyp = cum_hyp;
    
    % Compute mean and variance for this kernel.
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange_no_extrap);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange_no_extrap);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
    var(y-decomp_sigma' / complete_sigma * y);    
    
    data_mean = decomp_sigma' / complete_sigma * y;
    data_var = diag(decomp_sigma - decomp_sigma' / complete_sigma * decomp_sigma);
    cum_SNRs(j) = 10 * log10(sum(data_mean.^2)/sum(data_var));
    cum_vars(j) = (1 - var(y - data_mean) / var(y)) * 100;
    cum_resid_vars(j) = (1 - var(y - data_mean) / var(resid)) * 100;
    resid = y - data_mean;
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X*X_scale+X_mean, y*y_scale, ...
                   xrange_no_extrap*X_scale+X_mean, ...
                   decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale, false, false);
    
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    %title(['The above + ' latex_names{i}]);
    title(sprintf('Sum of components up to component %d', j));
    %fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d_cum.fig', figname, j);
    saveas( gcf, filename );
    
    % Compute mean and variance for this kernel.
    
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
    var(y-decomp_sigma' / complete_sigma * y);    
    
    data_mean = decomp_sigma' / complete_sigma * y;
    data_var = diag(decomp_sigma - decomp_sigma' / complete_sigma * decomp_sigma);
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X*X_scale+X_mean, y*y_scale, ...
                   xrange*X_scale+X_mean, ...
                   decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale, false, false);
    
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    %title(['The above + ' latex_names{i}]);
    title(sprintf('Sum of components up to component %d', j));
    %fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d_cum_extrap.fig', figname, j);
    saveas( gcf, filename );
    
    posterior_sigma = decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star;
    figure(i + 1); clf; hold on;
    sample_plot( X*X_scale+X_mean, xrange*X_scale+X_mean, ...
                 decomp_mean*y_scale, posterior_sigma);
    
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    %title(['The above + ' latex_names{i}]);
    title(sprintf('Random samples from the cumulative posterior', j));
    %fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d_cum_sample.fig', figname, j);
    saveas( gcf, filename );
    
    % Now plot posterior of residuals
    
    if j < numel(decomp_list)
        anti_cum_kernel = anti_cum_kernel(2:end);
        anti_cum_hyp = anti_cum_hyp((length(decomp_hypers{i})+1):end);
        cur_cov = {@covSum, anti_cum_kernel};
        cur_hyp = anti_cum_hyp;

        % Compute mean and variance for this kernel.
        decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange_no_extrap);
        decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange_no_extrap);
        decomp_mean = decomp_sigma_star' / complete_sigma * y;
        decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);

        figure(i + 1); clf; hold on;
        mean_var_plot( X*X_scale+X_mean, resid*y_scale, ...
                       xrange_no_extrap*X_scale+X_mean, ...
                       decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale, false, false, false);
                   
        title(sprintf('Residuals after component %d', j));
        filename = sprintf('%s_%d_anti_cum.fig', figname, j);
        saveas( gcf, filename );
    end
end

% Plot LOO posterior predictive, residuals and QQ

p_point_LOO = nan(size(X));
mean_LOO = nan(size(X));
var_LOO = nan(size(X));

K = complete_sigma;

for i = 1:length(X)
    not_i = [1:(i-1),(i+1):length(X)];
    
    K_ii = K(not_i,not_i);
    K_i = K(i,not_i);
    y_i = y(not_i);
    
    mean_LOO(i) = K_i * (K_ii \ y_i);
    var_LOO(i) = K(i,i) - K_i * (K_ii \ K_i');
    standard = (y(i) - mean_LOO(i)) ./ sqrt(var_LOO(i));
    
    p_point_LOO(i) = normcdf(standard);
end

figure(333); clf; hold on;
mean_var_plot( X*X_scale+X_mean, y*y_scale, ...
               X*X_scale+X_mean, ...
               mean_LOO*y_scale, 2.*sqrt(var_LOO)*y_scale);
           
title(sprintf('LOO posterior predictive', j));
filename = sprintf('%s_loo_pp.fig', figname);
saveas( gcf, filename );

figure(444); clf; hold on;
mean_var_plot( X*X_scale+X_mean, p_point_LOO, ...
               X*X_scale+X_mean, ...
               mean_LOO*y_scale, 2.*sqrt(var_LOO)*y_scale, ...
               false, true);
           
title(sprintf('LOO residuals', j));
filename = sprintf('%s_loo_resid.fig', figname);
saveas( gcf, filename );

figure(555); clf; hold on;
qq_uniform_plot(p_point_LOO);
           
title(sprintf('LOO residuals QQ-plot', j));
filename = sprintf('%s_loo_qq.fig', figname);
saveas( gcf, filename );

% Plot LCO posterior predictives, residuals and QQ

chunk_size = 0.1;

p_point_LOO = nan(size(X));
mean_LOO = nan(size(X));
var_LOO = nan(size(X));

K = complete_sigma;

for i = 1:length(X)
    not_close = abs(X-X(i)) > ((max(X) - min(X)) * chunk_size * 0.5);
    
    K_ii = K(not_close,not_close);
    K_i = K(i,not_close);
    y_i = y(not_close);
    
    mean_LOO(i) = K_i * (K_ii \ y_i);
    var_LOO(i) = K(i,i) - K_i * (K_ii \ K_i');
    standard = (y(i) - mean_LOO(i)) ./ sqrt(var_LOO(i));
    
    p_point_LOO(i) = normcdf(standard);
end

figure(777); clf; hold on;
mean_var_plot( X*X_scale+X_mean, y*y_scale, ...
               X*X_scale+X_mean, ...
               mean_LOO*y_scale, 2.*sqrt(var_LOO)*y_scale);
           
title(sprintf('LCO posterior predictive', j));
filename = sprintf('%s_lco_pp.fig', figname);
saveas( gcf, filename );

figure(888); clf; hold on;
mean_var_plot( X*X_scale+X_mean, p_point_LOO, ...
               X*X_scale+X_mean, ...
               mean_LOO*y_scale, 2.*sqrt(var_LOO)*y_scale, ...
               false, true);
           
title(sprintf('LCO residuals', j));
filename = sprintf('%s_lco_resid.fig', figname);
saveas( gcf, filename );

figure(999); clf; hold on;
qq_uniform_plot(p_point_LOO);
           
title(sprintf('LCO residuals QQ-plot', j));
filename = sprintf('%s_lco_qq.fig', figname);
saveas( gcf, filename );

% Plot z score residuals

L = chol(K);
z = (L') \ y;
p = normcdf(z);

figure(123); clf; hold on;
mean_var_plot( X*X_scale+X_mean, p, ...
               X*X_scale+X_mean, ...
               mean_LOO*y_scale, 2.*sqrt(var_LOO)*y_scale, ...
               false, true);
           
title(sprintf('z score residuals', j));
filename = sprintf('%s_z_resid.fig', figname);
saveas( gcf, filename );

figure(234); clf; hold on;
qq_uniform_plot(p);
           
title(sprintf('z score residuals QQ-plot', j));
filename = sprintf('%s_z_qq.fig', figname);
saveas( gcf, filename );

% Save data to file

save(sprintf('%s_decomp_data.mat', figname), 'idx', 'SNRs', 'vars', ...
     'cum_SNRs', 'cum_vars', 'cum_resid_vars', 'MAEs', 'MAV_data', ...
     'MAE_reductions', 'monotonic', 'gradients', ...
     'qq_d_min_p', 'qq_d_max_p', ...
     'acf_min_p', 'acf_min_loc_p', ...
     'pxx_max_p', 'pxx_max_loc_p');
 
% Convert everything to pdf

dirname = fileparts(figname);
files = dir([dirname, '/*.fig']);
for f_ix = 1:numel(files)
    curfile = [dirname, '/', files(f_ix).name];
    h = open(curfile);
    outfile = [dirname, '/', files(f_ix).name];
    pdfname = strrep(outfile, '.fig', '')
    save2pdf( pdfname, gcf, 600, true );
    %export_fig(pdfname, '-pdf');
    close all
end

end


function mean_var_plot( xdata, ydata, xrange, forecast_mu, forecast_scale, small_plot, data_only, no_data )

    if nargin < 6; small_plot = false; end
    if nargin < 7; data_only = false; end
    if nargin < 8; no_data = false; end

    % Figure settings.
    lw = 1.2;
    opacity = 1;
    light_blue = [227 237 255]./255;
    
    if ~data_only
        % Plot confidence bears.
        jbfill( xrange', ...
            forecast_mu' + forecast_scale', ...
            forecast_mu' - forecast_scale', ...
            light_blue, 'none', 1, opacity); hold on;   
    end
    
    
    set(gca,'Layer','top');  % Stop axes from being overridden.
        
    % Plot data.
    %plot( xdata, ydata, 'ko', 'MarkerSize', 2.1, 'MarkerFaceColor', facecol, 'MarkerEdgeColor', facecol ); hold on;    
    %h_dots = line( xdata, ydata, 'Marker', '.', 'MarkerSize', 2, 'MarkerEdgeColor',  [0 0 0], 'MarkerFaceColor', [0 0 0], 'Linestyle', 'none' ); hold on;    
    if ~no_data
        plot( xdata, ydata, 'k.');
    end
 
    if ~data_only
        % Plot mean function.
        plot(xrange, forecast_mu, 'Color', colorbrew(2), 'LineWidth', lw); hold on;
    end
        

    
    %set(gca, 'Children', [h_dots, h_bars, h_mean ] );
    %e1 = (max(xrange) - min(xrange))/300;
    %for i = 1:length(xdata)
    %   line( [xdata(i) - e1, xdata(i) + e1], [ydata(i) + e1, ydata(i) + e1], 'Color', [0 0 0 ], 'LineWidth', 2 );
    %end
    %set_fig_units_cm( 12,6 );   
    %ag_plot_little_circles_no_alpha(xdata, ydata, 0.02, [0 0 0])
    
    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'TickDir', 'out');
    
    xlim([min(xrange), max(xrange)]);
    if small_plot
        totalrange = (max(xrange) - min(xrange));
        xlim([min(xrange) + totalrange*0.7, max(xrange) - totalrange*0.05]);
    end    
    
    % Plot a vertical bar to indicate the start of extrapolation.
    if ~all(forecast_mu == 0) && ~(max(xdata) == max(xrange))  % Don't put extrapolation line on residuals plot.
        y_lim = get(gca,'ylim');
        line( [max(xdata), max(xdata)], y_lim, 'Linestyle', '--', 'Color', [0.3 0.3 0.3 ]);
    end 
    
    % Plot a vertical bar to indicate the start of extrapolation.
    if ~all(forecast_mu == 0) && ~(min(xdata) == min(xrange))  % Don't put extrapolation line on residuals plot.
        y_lim = get(gca,'ylim');
        line( [min(xdata), min(xdata)], y_lim, 'Linestyle', '--', 'Color', [0.3 0.3 0.3 ]);
    end 
    
    %set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
    %set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
    %set(gca, 'TickDir', 'out')
    
    set_fig_units_cm( 16,8 );
    
    if small_plot
        set_fig_units_cm( 6, 6 );
    end
end

function sample_plot( xdata, xrange, forecast_mu, forecast_sigma )

    % Figure settings.
    lw = 1.2;
    opacity = 1;
    light_blue = [227 237 255]./255;
    
    set(gca,'Layer','top');  % Stop axes from being overridden.
    
    K = forecast_sigma;
    K = non_singular(K);
    L = chol(K);
 
    sample = forecast_mu + L' * randn(size(forecast_mu));
    plot(xrange, sample, 'Color', colorbrew(2), 'LineWidth', lw);
    sample = forecast_mu + L' * randn(size(forecast_mu));
    plot(xrange, sample, 'Color', colorbrew(3), 'LineWidth', lw);
    sample = forecast_mu + L' * randn(size(forecast_mu));
    plot(xrange, sample, 'Color', colorbrew(4), 'LineWidth', lw);
    xlim([min(xrange), max(xrange)]);
    
    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'TickDir', 'out');
    
    % Plot a vertical bar to indicate the start of extrapolation.
    if ~all(forecast_mu == 0) && ~(max(xdata) == max(xrange))  % Don't put extrapolation line on residuals plot.
        y_lim = get(gca,'ylim');
        line( [max(xdata), max(xdata)], y_lim, 'Linestyle', '--', 'Color', [0.3 0.3 0.3 ]);
    end 
    
    % Plot a vertical bar to indicate the start of extrapolation.
    if ~all(forecast_mu == 0) && ~(min(xdata) == min(xrange))  % Don't put extrapolation line on residuals plot.
        y_lim = get(gca,'ylim');
        line( [min(xdata), min(xdata)], y_lim, 'Linestyle', '--', 'Color', [0.3 0.3 0.3 ]);
    end 
    
    set_fig_units_cm( 16,8 );
end

function qq_uniform_plot( sample )
    
    lw = 1.2;

    sample = sort(sample);
    U = linspace(0,1,length(sample)+2);
    U = U(2:(end-1));
    
    plot(U, U, 'Color', colorbrew(2), 'LineWidth', lw);
    plot(U, sample, 'k.');
    
    xlim([0,1]);
    ylim([0,1]);
    
    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'TickDir', 'out');
    
    xlabel('Uniform quantiles');
    ylabel('Sample quantiles');
    
    set_fig_units_cm( 16,8 );
end


