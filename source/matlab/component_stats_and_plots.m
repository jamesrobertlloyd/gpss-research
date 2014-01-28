function component_stats_and_plots(X, y, mean_family, mean_params, ...
                     complete_covfunc, ...
                     complete_hypers, decomp_list, ...
                     decomp_hypers, envelope_list, ...
                     envelope_hypers, lik_family_, lik_params, figname, ...
                     idx)
% Plots decomposition and extrapolations 
% And saves some statistics
                 
%%%%%%%%%%%
% WARNING %
%%%%%%%%%%%
% - Ignores mean
% - Ignores lik - assumes provides a noise parameter

if isempty(lik_params)
    log_noise = -inf;
else
    log_noise = lik_params(0);
end
noise_var = exp(2*log_noise);

% Convert to double in case python saved as integers
X = double(X);
y = double(y);

%%%% TODO - function should accept a mean function
%y = y - mean(y);

%%%% TODO - turn into parameters
left_extend = 0.0;%0.1  % What proportion to extend beyond the data range.
right_extend = 0.1;%0.4

%%%% TODO - parameter
env_thresh = 0.99; % Threshold above which a component is considered active

%%%% TODO - parameter
num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';
xrange_no_extrap = linspace(min(X), max(X), num_interpolation_points)';

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
mean_var_plot( X, y, ...
               xrange, complete_mean, ...
               2.*sqrt(complete_var), false, true); % Only plot the data
title('Raw data');
filename = sprintf('%s_raw_data.fig', figname);
saveas( gcf, filename );

% Now plot the posterior
figure(2); clf; hold on;
mean_var_plot( X, y, ...
               xrange, complete_mean, ...
               2.*sqrt(complete_var), false, false);
title('Full model posterior with extrapolations');
filename = sprintf('%s_all.fig', figname);
saveas( gcf, filename );

% Now plot samples from the posterior
figure(3); clf; hold on;
sample_plot( X, xrange, complete_mean, ...
               posterior_sigma);
title('Random samples from the full model posterior');
filename = sprintf('%s_all_sample.fig', figname);
saveas( gcf, filename );

% Prepare to save some data

SNRs = zeros(numel(decomp_list),1);
vars = zeros(numel(decomp_list),1);
monotonic = zeros(numel(decomp_list),1);
gradients = zeros(numel(decomp_list),1);

% Plot each component without data

for j = 1:numel(decomp_list)
    i = idx(j);
    cur_cov = decomp_list{i};
    cur_hyp = decomp_hypers{i};
    env_cov = envelope_list{i};
    env_hyp = envelope_hypers{i};
    
    % Compute mean and variance for this kernel.
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange_no_extrap);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange_no_extrap);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
    envelope = diag(feval(env_cov{:}, env_hyp, X));
    
    data_mean = decomp_sigma' / complete_sigma * y;
    diffs = data_mean(2:end) - data_mean(1:(end-1));
    data_covar = decomp_sigma - decomp_sigma' / complete_sigma * decomp_sigma;
    data_var = diag(data_covar);
    SNRs(j) = 10 * log10(sum(data_mean.^2)/sum(data_var));
    vars(j) = (1 - var(y - data_mean) / var(y)) * 100;
    diffs_thresh = diffs((envelope(1:(end-1))>env_thresh) & (envelope(2:end)>env_thresh));
    if isempty(diffs_thresh)
        monotonic(j) = 0;
    else
        if all(diffs_thresh>0)
            monotonic(j) = 1;
        elseif all(diffs_thresh<0)
            monotonic(j) = -1;
        else
            monotonic(j) = 0;
        end
    end
    thresholded_data_mean = data_mean(envelope>env_thresh);
    X_thresholded = X(envelope>env_thresh);
    if isempty(thresholded_data_mean)
        gradients(j) = 0;
    else
        gradients(j) = (thresholded_data_mean(end) - thresholded_data_mean(1)) ...
                       / (X_thresholded(end) - X_thresholded(1));
    end
    
    % Compute the remaining signal after removing the mean prediction from all
    % other parts of the kernel.
    removed_mean = y - (complete_sigma - decomp_sigma)' / complete_sigma * y;
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X, removed_mean, ...
                   xrange_no_extrap, ...
                   decomp_mean, 2.*sqrt(decomp_var), false, false, true); % Don't plot data
    title(sprintf('Posterior of component %d', j));
    filename = sprintf('%s_%d.fig', figname, j);
    saveas( gcf, filename );   
    
    % Compute mean and variance for extrapolation
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_sigma_posterior = decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star;
    decomp_var = diag(decomp_sigma_posterior);
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X, removed_mean, ...
                   xrange, ...
                   decomp_mean, 2.*sqrt(decomp_var), false, false, true); % Don't plot data
    title(sprintf('Posterior of component %d', j));
    filename = sprintf('%s_%d_extrap.fig', figname, j);
    saveas( gcf, filename );
    
    figure(i + 1); clf; hold on;
    sample_plot( X, xrange, decomp_mean, decomp_sigma_posterior )

    title(sprintf('Random samples from the posterior of component %d', j));
    filename = sprintf('%s_%d_sample.fig', figname, j);
    saveas( gcf, filename );
end

% Plot cumulative components with data

cum_kernel = cell(0);
cum_hyp = [];

var(y);
resid = y;

% Prepare to save some data

cum_SNRs = zeros(numel(decomp_list),1);
cum_vars = zeros(numel(decomp_list),1);
cum_resid_vars = zeros(numel(decomp_list),1);

anti_cum_kernel = cell(0);
anti_cum_hyp = [];
for j = 1:numel(decomp_list)
    i = idx(j);
    anti_cum_kernel{j} = decomp_list{i};
    anti_cum_hyp = [anti_cum_hyp, decomp_hypers{i}]; %#ok<AGROW>
end

for j = 1:numel(decomp_list)
    i = idx(j);
    cum_kernel{j} = decomp_list{i};
    cum_hyp = [cum_hyp, decomp_hypers{i}]; %#ok<AGROW>
    cur_cov = {@covSum, cum_kernel};
    cur_hyp = cum_hyp;
    
    % Compute mean and variance for this kernel.
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange_no_extrap);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange_no_extrap);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);  
    
    data_mean = decomp_sigma' / complete_sigma * y;
    data_var = diag(decomp_sigma - decomp_sigma' / complete_sigma * decomp_sigma);
    cum_SNRs(j) = 10 * log10(sum(data_mean.^2)/sum(data_var));
    cum_vars(j) = (1 - var(y - data_mean) / var(y)) * 100;
    cum_resid_vars(j) = (1 - var(y - data_mean) / var(resid)) * 100;
    resid = y - data_mean;
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X, y, ...
                   xrange_no_extrap, ...
                   decomp_mean, 2.*sqrt(decomp_var), false, false);
    title(sprintf('Sum of components up to component %d', j));
    filename = sprintf('%s_%d_cum.fig', figname, j);
    saveas( gcf, filename );
    
    % Compute mean and variance for extrapolation
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);   
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X, y, ...
                   xrange, ...
                   decomp_mean, 2.*sqrt(decomp_var), false, false);
    title(sprintf('Sum of components up to component %d', j));
    filename = sprintf('%s_%d_cum_extrap.fig', figname, j);
    saveas( gcf, filename );
    
    posterior_sigma = decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star;
    figure(i + 1); clf; hold on;
    sample_plot( X, xrange, ...
                 decomp_mean, posterior_sigma);
    title(sprintf('Random samples from the cumulative posterior', j));
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
        mean_var_plot( X, resid, ...
                       xrange_no_extrap, ...
                       decomp_mean, 2.*sqrt(decomp_var), false, false, false);
                   
        title(sprintf('Residuals after component %d', j));
        filename = sprintf('%s_%d_anti_cum.fig', figname, j);
        saveas( gcf, filename );
    end
end

% Save data to file

save(sprintf('%s_component_data.mat', figname), 'SNRs', 'vars', ...
     'cum_SNRs', 'cum_vars', 'cum_resid_vars', ...
     'monotonic', 'gradients');
 
% Convert everything to pdf

% dirname = fileparts(figname);
% files = dir([dirname, '/*.fig']);
% for f_ix = 1:numel(files)
%     curfile = [dirname, '/', files(f_ix).name];
%     open(curfile);
%     outfile = [dirname, '/', files(f_ix).name];
%     pdfname = strrep(outfile, '.fig', '');
%     save2pdf( pdfname, gcf, 600, true );
%     %export_fig(pdfname, '-pdf');
%     close all
% end

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