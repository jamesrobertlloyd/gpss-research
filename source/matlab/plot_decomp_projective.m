function plot_decomp_projective(X, y, complete_covfunc, complete_hypers, decomp_list, ...
                     decomp_hypers, log_noise, figname, latex_names, ...
                     full_name, X_mean, X_scale, y_mean, y_scale)

% TODO: Assert that the sum of all kernels is the same as the complete kernel.

% Convert to double in case python saved as integers
X = double(X);
y = double(y);

%%%% New code - remove the middle to look at interpolation

% left_cut = min(X) + (max(X) - min(X)) * 0.4;
% right_cut = min(X) + (max(X) - min(X)) * 0.6;
% i = (X < left_cut | X > right_cut);
% X = X(i);
% y = y(i);

%%%% FIXME - this is an assumption that may no longer be valid
%y = y - mean(y);

left_extend = 0.1;  % What proportion to extend beyond the data range.
right_extend = 0.4;

num_interpolation_points = 2000;

x_left = min(X) - (max(X) - min(X))*left_extend;
x_right = max(X) + (max(X) - min(X))*right_extend;
xrange = linspace(x_left, x_right, num_interpolation_points)';
%xrange2 = linspace(left_cut, right_cut, num_interpolation_points)';

noise_var = exp(2*log_noise);
complete_sigma = feval(complete_covfunc{:}, complete_hypers, X, X) + eye(length(y)).*noise_var;
%complete_sigma = complete_sigma + 1e-5*max(max(complete_sigma))*eye(size(complete_sigma));
complete_sigmastar = feval(complete_covfunc{:}, complete_hypers, X, xrange);
%complete_sigmastar2 = feval(complete_covfunc{:}, complete_hypers, X, xrange2);
complete_sigmastarstart = feval(complete_covfunc{:}, complete_hypers, xrange, xrange);
%complete_sigmastarstart2 = feval(complete_covfunc{:}, complete_hypers, xrange2, xrange2);

% First, plot the original, combined kernel
complete_mean = complete_sigmastar' / complete_sigma * y;
%complete_mean2 = complete_sigmastar2' / complete_sigma * y;
complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
%complete_var2 = diag(complete_sigmastarstart2 - complete_sigmastar2' / complete_sigma * complete_sigmastar2);
    
figure(1); clf; hold on;
%mean_var_plot2( X*X_scale+X_mean, y*y_scale+y_mean, ...
%               xrange*X_scale+X_mean, xrange2*X_scale+X_mean, complete_mean*y_scale+y_mean, ...
%               2.*sqrt(complete_var)*y_scale, complete_mean2*y_scale+y_mean, ...
%               2.*sqrt(complete_var2)*y_scale);
mean_var_plot( X*X_scale+X_mean, y*y_scale+y_mean, ...
               xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, ...
               2.*sqrt(complete_var)*y_scale, false, true);

% Remove outer brackets and extra latex markup from name.
if iscell(full_name); full_name = full_name{1}; end
full_name = strrep(full_name, '\left', '');
full_name = strrep(full_name, '\right', '');
%full_name = strtrim(full_name);
%if full_name(1) == '('; full_name(1) = ''; end
%if full_name(end) == ')'; full_name(end) = ''; end
title(full_name);
filename = sprintf('%s_all.fig', figname);
saveas( gcf, filename );
%filename = sprintf('%s_all.pdf', figname);
%save2pdf( filename, gcf, 400, true )

% Then plot the same thing, but just the end.
complete_mean = complete_sigmastar' / complete_sigma * y;
complete_var = diag(complete_sigmastarstart - complete_sigmastar' / complete_sigma * complete_sigmastar);
    
figure(100); clf; hold on;
mean_var_plot(X*X_scale+X_mean, y*y_scale+y_mean, xrange*X_scale+X_mean, complete_mean*y_scale+y_mean, 2.*sqrt(complete_var)*y_scale, true, false);
title(full_name);
filename = sprintf('%s_all_small.fig', figname);
saveas( gcf, filename );
%filename = sprintf('%s_all.pdf', figname);
%save2pdf( filename, gcf, 400, true )

% Plot kernel

% figure(666); clf; hold on;
% imagesc(complete_sigmastarstart);
% xlim([1, length(xrange)]);
% ylim([1, length(xrange)]);
% title(full_name);
% filename = sprintf('%s_kernel.fig', figname);
% saveas( gcf, filename );

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

%scaled_vars = NaN(numel(decomp_list), 1);
MAEs = NaN(numel(decomp_list), 1);

% for i = 1:numel(decomp_list)
%     cur_cov = decomp_list{i};
%     cur_hyp = decomp_hypers{i};
%     
%     % Compute mean and variance for this kernel.
%     decomp_sigma = feval(cur_cov{:}, cur_hyp, X, X);
%     decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, X);
%     decomp_mean = decomp_sigma_star' / complete_sigma * y;
% 
%     scaled_vars(i) = var(decomp_mean) / var(y);
% end

folds = 10;

X_train = cell(folds,1);
y_train = cell(folds,1);
X_valid = cell(folds,1);
y_valid = cell(folds,1);

% Chekc me for overlap

for fold = 1:folds
    range = max(1,floor(length(X)*(fold-1)/folds)):floor(length(X)*(fold)/folds);
    X_valid{fold} = X(range);
    y_valid{fold} = y(range);
    range = [1:min(length(X),floor(length(X)*(fold-1)/folds)-1),...
            max(1,floor(length(X)*(fold)/folds)+1):length(X)];
    X_train{fold} = X(range);
    y_train{fold} = y(range);
end

% idx = [];
% 
% decomp_sigma_star_list = cell(numel(decomp_list),1);
% decomp_sigma_starstar_list = cell(numel(decomp_list),1);
% 
% for i = 1:numel(decomp_list)
%     cur_cov = decomp_list{i};
%     cur_hyp = decomp_hypers{i};
%     
%     % Compute mean and variance for this kernel.
%     decomp_sigma_star_list{i} = feval(cur_cov{:}, cur_hyp, X, xrange);
%     decomp_sigma_starstar_list{i} = feval(cur_cov{:}, cur_hyp, xrange, xrange);
% end
% 
% temp_decomp_sigma_star_list = decomp_sigma_star_list;
% temp_decomp_sigma_starstar_list = decomp_sigma_starstar_list;
% 
% for i = 1:numel(decomp_list)
%     best_var = Inf;
%     for j = 1:numel(decomp_list)
%         if ~sum(j == idx)
%             % Project all other remaining components
%             for k = 1:numel(decomp_list)
%                 if (~sum(k == idx)) && (k ~= j)
%                     alpha = solve_sdp(decomp_sigma_starstar_list{k}, decomp_sigma_starstar_list{j});
%                     temp_decomp_sigma_star_list{j} = decomp_sigma_star_list{j} + alpha*decomp_sigma_star_list{j};
%                     temp_decomp_sigma_starstar_list{j} = decomp_sigma_starstar_list{j} + alpha*decomp_sigma_starstar_list{j};
%                 end
%             end
% 
%             decomp_var = diag(temp_decomp_sigma_starstar_list{j} - temp_decomp_sigma_star_list{j}' / complete_sigma * temp_decomp_sigma_star_list{j});
%             
%             my_var = mean(decomp_var);
%             if my_var < best_var
%                 best_j  = j;
%                 best_var = my_var;
%             end
%         end
%     end
%     idx = [idx, best_j];
%     % Project
%     for k = 1:numel(decomp_list)
%         if (~sum(k == idx)) && (k ~= best_j)
%             alpha = solve_sdp(decomp_sigma_starstar_list{k}, decomp_sigma_starstar_list{best_j});
%             decomp_sigma_starstar_list{k} = decomp_sigma_starstar_list{k} - alpha*decomp_sigma_starstar_list{best_j};
%             decomp_sigma_star_list{k} = decomp_sigma_star_list{k} - alpha*decomp_sigma_star_list{best_j};
%             decomp_sigma_starstar_list{best_j} = decomp_sigma_starstar_list{best_j} + alpha*decomp_sigma_starstar_list{best_j};
%             decomp_sigma_star_list{best_j} = decomp_sigma_star_list{best_j} + alpha*decomp_sigma_star_list{best_j};
%         end
%     end
% end

idx = [];

K_list = cell(length(X_train),1);
Ks_list = cell(length(X_train),numel(decomp_list));
Ks_sum_list = cell(length(X_train),1);

for fold = 1:length(X_train)
  K_list{fold} = feval(complete_covfunc{:}, complete_hypers, X_train{fold}) + ...
      noise_var*eye(length(y_train{fold}));
    for i = 1:numel(decomp_list)
        cur_cov = decomp_list{i};
        cur_hyp = decomp_hypers{i};
        Ks_list{fold,i} = feval(cur_cov{:}, cur_hyp, X_train{fold}, X_valid{fold});
    end
    Ks_sum_list{fold} = zeros(length(X_train{fold}), length(X_valid{fold}));
end

temp_Ks_list = Ks_list;

decomp_sigma_starstar_list = cell(numel(decomp_list),1);

for i = 1:numel(decomp_list)
    cur_cov = decomp_list{i};
    cur_hyp = decomp_hypers{i};
    
    % Compute mean and variance for this kernel.
    decomp_sigma_starstar_list{i} = feval(cur_cov{:}, cur_hyp, xrange, xrange);
end

for i = 1:numel(decomp_list)
    best_MAE = Inf;
    for j = 1:numel(decomp_list)
        if ~sum(j == idx)
            % Project all other remaining components
            for k = 1:numel(decomp_list)
                if (~sum(k == idx)) && (k ~= j)
                    alpha = solve_sdp(decomp_sigma_starstar_list{k}, decomp_sigma_starstar_list{j});
                    for fold = 1:length(X_train)
                        temp_Ks_list{fold, j} = Ks_list{fold, j} + alpha*Ks_list{fold, j};
                    end
                end
            end
            e = NaN(length(X_train), 1);
            for fold = 1:length(X_train)
              K = K_list{fold};
              Ks = Ks_sum_list{fold} + temp_Ks_list{fold,j};

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
    % Project
    for k = 1:numel(decomp_list)
        if (~sum(k == idx)) && (k ~= best_j)
            alpha = solve_sdp(decomp_sigma_starstar_list{k}, decomp_sigma_starstar_list{best_j});
            for fold = 1:length(X_train)
                Ks_list{fold, k} = Ks_list{fold, k} - alpha*Ks_list{fold, best_j};
                Ks_list{fold, best_j} = Ks_list{fold, best_j} + alpha*Ks_list{fold, best_j};
            end
            decomp_sigma_starstar_list{k} = decomp_sigma_starstar_list{k} - alpha*decomp_sigma_starstar_list{best_j};
            decomp_sigma_starstar_list{best_j} = decomp_sigma_starstar_list{best_j} + alpha*decomp_sigma_starstar_list{best_j};
        end
    end
    for fold = 1:length(X_train)
        Ks_sum_list{fold} = Ks_sum_list{fold} + Ks_list{fold, best_j};
    end
end

% idx = [];
% 
% cum_kernel = cell(0);
% cum_hyp = [];
% %used_components = [];
% 
% 
% cum_kernel = cell(0);
% cum_hyp = [];
% 
% for i = 1:numel(decomp_list)
%     best_MAE = Inf;
%     for j = 1:numel(decomp_list)
%         if ~sum(j == idx)
%             kernels = cum_kernel;
%             kernels{i} = decomp_list{j};
%             hyps = cum_hyp;
%             hyps = [hyps, decomp_hypers{j}];
%             % This is broken - noise is wrong
%             hyp.mean = [];
%             hyp.cov = hyps;
%             cur_cov = {@covSum, kernels};
%             %decomp_sigma = feval(cur_cov{:}, hyps, X, X);
%             %ymu = decomp_sigma' / complete_sigma * y;
%             %hyp.lik = log(std(ymu - y));
%             %hyp = minimize(hyp, @gp, -100, @infExact, @meanZero, cur_cov, @likGauss, X, y);
%             e = NaN(length(X_train), 1);
%             for fold = 1:length(X_train)
%               K = feval(complete_covfunc{:}, complete_hypers, X_train{fold}) + ...
%                   noise_var*eye(length(y_train{fold}));
%               %K = K + 1e-5*max(max(K))*eye(size(K));
%               Ks = feval(cur_cov{:}, hyp.cov, X_train{fold}, X_valid{fold});
% 
%               ymu = Ks' * (K \ y_train{fold});
% 
%               e(fold) = mean(abs(y_valid{fold} - ymu));
%             end
%             
%             my_MAE = mean(e);
%             %my_MAE = gp(hyp, @infExact, @meanZero, cur_cov, @likGauss, X, y);
%             %my_MAE = mean(abs(ymu-y));
%             if my_MAE < best_MAE
%                 best_j  = j;
%                 best_MAE = my_MAE;
%             end
%         end
%     end
%     idx = [idx, best_j];
%     cum_kernel{i} = decomp_list{best_j};
%     cum_hyp = [cum_hyp, decomp_hypers{best_j}];
% end

%[dummy, idx] = sort(scaled_vars, 'descend');
%[dummy, idx] = sort(MAEs, 'descend');

decomp_sigma_list = cell(numel(decomp_list),1);
decomp_sigma_star_list = cell(numel(decomp_list),1);
decomp_sigma_starstar_list = cell(numel(decomp_list),1);

for i = 1:numel(decomp_list)
    cur_cov = decomp_list{i};
    cur_hyp = decomp_hypers{i};
    
    % Compute mean and variance for this kernel.
    decomp_sigma_list{i} = feval(cur_cov{:}, cur_hyp, X, X);
    decomp_sigma_star_list{i} = feval(cur_cov{:}, cur_hyp, X, xrange);
    decomp_sigma_starstar_list{i} = feval(cur_cov{:}, cur_hyp, xrange, xrange);
end

for k = 1:numel(decomp_list)
    i = idx(k);
    for j = idx((k+1):end)
        alpha = solve_sdp(decomp_sigma_starstar_list{j}, decomp_sigma_starstar_list{i});
        decomp_sigma_list{j} = decomp_sigma_list{j} - alpha*decomp_sigma_list{i};
        decomp_sigma_star_list{j} = decomp_sigma_star_list{j} - alpha*decomp_sigma_star_list{i};
        decomp_sigma_starstar_list{j} = decomp_sigma_starstar_list{j} - alpha*decomp_sigma_starstar_list{i};
        decomp_sigma_list{i} = decomp_sigma_list{i} + alpha*decomp_sigma_list{i};
        decomp_sigma_star_list{i} = decomp_sigma_star_list{i} + alpha*decomp_sigma_star_list{i};
        decomp_sigma_starstar_list{i} = decomp_sigma_starstar_list{i} + alpha*decomp_sigma_starstar_list{i};
    end
end

for j = 1:numel(decomp_list)
    i = idx(j);
    cur_cov = decomp_list{i};
    cur_hyp = decomp_hypers{i};
    
    % Compute mean and variance for this kernel.
    %decomp_sigma = feval(cur_cov{:}, cur_hyp, X, X);
    %decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    %decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange, xrange);
    decomp_sigma = decomp_sigma_list{i};
    decomp_sigma_star = decomp_sigma_star_list{i};
    decomp_sigma_starstar = decomp_sigma_starstar_list{i};
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
    % Compute the remaining signal after removing the mean prediction from all
    % other parts of the kernel.
    removed_mean = y - (complete_sigma - decomp_sigma)' / complete_sigma * y;
    %removed_mean = y;
    
    figure(i + 1); clf; hold on;
    mean_var_plot_no_data( X*X_scale+X_mean, removed_mean*y_scale, ...
                   xrange*X_scale+X_mean, ...
                   decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale, false, false);
    
    %set(gca, 'Children', [h_bars, h_mean, h_dots] );
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    title(latex_names{i});
    fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d.fig', figname, j);
    saveas( gcf, filename );
    %filename = sprintf('%s_%d.pdf', figname, i);
    %save2pdf( filename, gcf, 400, true );
    
    % Plot the kernel
    
%     figure(666); clf; hold on;
%     imagesc(decomp_sigma_starstar);
%     xlim([1, length(xrange)]);
%     ylim([1, length(xrange)]);
%     title(latex_names{i});
%     filename = sprintf('%s_%d_kernel.fig', figname, j);
%     saveas( gcf, filename );
end

cum_kernel = cell(0);
cum_hyp = [];

decomp_sigma = zeros(length(X), length(X));
decomp_sigma_star = zeros(length(X), length(xrange));
decomp_sigma_starstar = zeros(length(xrange), length(xrange));

for j = 1:numel(decomp_list)
    i = idx(j);
    cum_kernel{j} = decomp_list{i};
    cum_hyp = [cum_hyp, decomp_hypers{i}];
    cur_cov = {@covSum, cum_kernel};
    cur_hyp = cum_hyp;
    
    % Compute mean and variance for this kernel.
    %decomp_sigma = feval(cur_cov{:}, cur_hyp, X, X);
    %decomp_sigma_star = feval(cur_cov{:}, cur_hyp, X, xrange);
    %decomp_sigma_starstar = feval(cur_cov{:}, cur_hyp, xrange, xrange);
    decomp_sigma = decomp_sigma + decomp_sigma_list{i};
    decomp_sigma_star = decomp_sigma_star + decomp_sigma_star_list{i};
    decomp_sigma_starstar = decomp_sigma_starstar + decomp_sigma_starstar_list{i};
    decomp_mean = decomp_sigma_star' / complete_sigma * y;
    decomp_var = diag(decomp_sigma_starstar - decomp_sigma_star' / complete_sigma * decomp_sigma_star);
    
    % Compute the remaining signal after removing the mean prediction from all
    % other parts of the kernel.
    %removed_mean = y - (complete_sigma - decomp_sigma)' / complete_sigma * y;
    removed_mean = y;
    
    figure(i + 1); clf; hold on;
    mean_var_plot( X*X_scale+X_mean, removed_mean*y_scale, ...
                   xrange*X_scale+X_mean, ...
                   decomp_mean*y_scale, 2.*sqrt(decomp_var)*y_scale, false, false);
    
    %set(gca, 'Children', [h_bars, h_mean, h_dots] );
    latex_names{i} = strrep(latex_names{i}, '\left', '');
    latex_names{i} = strrep(latex_names{i}, '\right', '');
    title(['The above + ' latex_names{i}]);
    fprintf([latex_names{i}, '\n']);
    filename = sprintf('%s_%d_cum.fig', figname, j);
    saveas( gcf, filename );
    %filename = sprintf('%s_%d.pdf', figname, i);
    %save2pdf( filename, gcf, 400, true );
    
    % Plot the kernel
    
%     figure(666); clf; hold on;
%     imagesc(decomp_sigma_starstar);
%     xlim([1, length(xrange)]);
%     ylim([1, length(xrange)]);
%     title(['The above + ' latex_names{i}]);
%     filename = sprintf('%s_%d_cum_kernel.fig', figname, j);
%     saveas( gcf, filename );
end
end


function mean_var_plot( xdata, ydata, xrange, forecast_mu, forecast_scale, small_plot, data_only )

    if nargin < 6; small_plot = false; end
    if nargin < 7; small_plot = false; end

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
    plot( xdata, ydata, 'k.');
 
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
function mean_var_plot_no_data( xdata, ydata, xrange, forecast_mu, forecast_scale, small_plot, data_only )

    if nargin < 6; small_plot = false; end
    if nargin < 7; small_plot = false; end

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
    %plot( xdata, ydata, 'k.');
 
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

function mean_var_plot2( xdata, ydata, xrange, xrange2, forecast_mu, forecast_scale, forecast_mu2, forecast_scale2, small_plot, data_only )

    if nargin < 6; small_plot = false; end
    if nargin < 7; data_only = false; end

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

        jbfill( xrange2', ...
            forecast_mu2' + forecast_scale2', ...
            forecast_mu2' - forecast_scale2', ...
            light_blue, 'none', 1, opacity);  
    end
    
    
    set(gca,'Layer','top');  % Stop axes from being overridden.
        
    % Plot data.
    %plot( xdata, ydata, 'ko', 'MarkerSize', 2.1, 'MarkerFaceColor', facecol, 'MarkerEdgeColor', facecol ); hold on;    
    %h_dots = line( xdata, ydata, 'Marker', '.', 'MarkerSize', 2, 'MarkerEdgeColor',  [0 0 0], 'MarkerFaceColor', [0 0 0], 'Linestyle', 'none' ); hold on;    
    plot( xdata, ydata, 'k.');
 
    
    % Plot mean function.
    if ~data_only
        plot(xrange, forecast_mu, 'Color', colorbrew(2), 'LineWidth', lw); hold on;
        plot(xrange2, forecast_mu2, 'Color', colorbrew(2), 'LineWidth', lw);
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


