function plot_decomp(X, y, complete_covfunc, complete_hypers, decomp_list, ...
                     decomp_hypers, log_noise, figname, latex_names, ...
                     full_name, X_mean, X_scale, y_mean, y_scale)

% TODO: Assert that the sum of all kernels is the same as the complete kernel.

% Convert to double in case python saved as integers
X = double(X);
y = double(y);

%%%% TODO - this should be an option
%y = y - mean(y);

left_extend = 0.1;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);

% First, plot the original, combined kernel
complete_mean = complete_sigmastar' / complete_sigma * y;
complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
    
figure(1); clf; hold on;
mean_var_plot( X*X_scale+X_mean, y*y_scale+y_mean, ...
               xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, ...
               2.*sqrt(complete_var)*y_scale, false, true); % Only plot the data

% Remove outer brackets and extra latex markup from name.
if iscell(full_name); full_name = full_name{1}; end
full_name = strrep(full_name, '\left', '');
full_name = strrep(full_name, '\right', '');
title(full_name);
filename = sprintf('%s_all.fig', figname);
saveas( gcf, filename );

% Then plot the same thing, but just the end.
complete_mean = complete_sigmastar' / complete_sigma * y;
complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
    
figure(100); clf; hold on;
mean_var_plot(X*X_scale+X_mean, y*y_scale+y_mean, xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, 2.*sqrt(complete_var)*y_scale, true, false);
title(full_name);
filename = sprintf('%s_all_small.fig', figname);
saveas( gcf, filename );

% Plot residuals.
figure(1000); clf; hold on;
data_complete_mean = feval(complete_covfunc{:}, complete_hypers, X, X)' / complete_sigma * y;
std_ratio = std((y-data_complete_mean)) / sqrt(noise_var);
mean_var_plot(X*X_scale+X_mean, (y-data_complete_mean)*y_scale, ...
              xrange*X_scale+X_mean, zeros(size(xrange)), ...
              2.*sqrt(noise_var).*ones(size(xrange)).*y_scale, false, true);
title(['Residuals']);
filename = sprintf('%s_resid.fig', figname);
saveas( gcf, filename );

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

for i = 1:numel(decomp_list)
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
    idx = [idx, best_j];
    cum_kernel{i} = decomp_list{best_j};
    cum_hyp = [cum_hyp, decomp_hypers{best_j}];
end

% Plot each component without data

for j = 1:numel(decomp_list)
    i = idx(j);
    cur_cov = decomp_list{i};
    cur_hyp = decomp_hypers{i};
    
    % Compute mean and variance for this kernel.
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange, xrange);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
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
    title(latex_names{i});
    fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d.fig', figname, j);
    saveas( gcf, filename );
end

% Plot cumulative components with data

cum_kernel = cell(0);
cum_hyp = [];

for j = 1:numel(decomp_list)
    i = idx(j);
    cum_kernel{j} = decomp_list{i};
    cum_hyp = [cum_hyp, decomp_hypers{i}];
    cur_cov = {@covSum, cum_kernel};
    cur_hyp = cum_hyp;
    
    % Compute mean and variance for this kernel.
    decomp_sigma = feval(cur_cov{:}, cur_hyp, X, X);
    decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange, xrange);
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X*X_scale+X_mean, y*y_scale, ...
                   xrange*X_scale+X_mean, ...
                   decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale, false, false);
    
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    title(['The above + ' latex_names{i}]);
    fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d_cum.fig', figname, j);
    saveas( gcf, filename );
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
    if ~all(forecast_mu == 0)  % Don't put extrapolation line on residuals plot.
        y_lim = get(gca,'ylim');
        line( [xdata(end), xdata(end)], y_lim, 'Linestyle', '--', 'Color', [0.3 0.3 0.3 ]);
    end 
    
    %set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
    %set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
    %set(gca, 'TickDir', 'out')
    
    set_fig_units_cm( 16,8 );
    
    if small_plot
        set_fig_units_cm( 6, 6 );
    end
end


