function samples_density_plot( xdata, ydata, xrange, forecast_mu, forecast_K, data_only, no_data )

    if nargin < 6; data_only = false; end
    if nargin < 7; no_data = false; end

    % Figure settings.
    lw = 1.3;
    %light_blue = [227 237 255]./255;
    
    quantiles = 0:0.01:0.5;
    opacity = 0.2;
    
    if ~data_only
        % Plot confidence bears.
        variance = diag(forecast_K);
        f = forecast_mu;
        for s = quantiles
            % CHECK ME
            edges = [f+norminv(s, 0, 1).*sqrt(variance); ...
            flipdim(f-norminv(s, 0, 1).*sqrt(variance),1)]; 
            hc1 = fill([xrange; flipdim(xrange,1)], edges, color_spectrum(2*s), 'EdgeColor', 'none'); hold on;
        end  
        hold on;   
        % Plot samples
        for n_sample = 1:10
            %cur_ixs = sort(randperm(numel(xrange), 2));
            %cur_range = cur_ixs(1):cur_ixs(2);
            cur_range = 1:numel(xrange);
            %sample = mvnrnd( f, non_singular(forecast_K), 1);
            sample = f + chol(non_singular(forecast_K))' * randn(size(xrange));
            hs = plot( xrange(cur_range), sample(cur_range), '-', 'Color', colorbrew(n_sample), 'Linewidth', lw); hold on;
        end
    end
    
    set(gca,'Layer','top');  % Stop axes from being overridden.
    
    if ~no_data
        plot( xdata, ydata, 'k.', 'MarkerSize', 20);
    end
 
    %if ~data_only
    %    % Plot mean function.
    %    plot(xrange, forecast_mu, 'Color', colorbrew(2), 'LineWidth', lw); hold on;
    %end
    
    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'TickDir', 'out');
    
    xlim([min(xrange), max(xrange)]); 
    
    set_fig_units_cm( 16,8 );
end

function col = color_spectrum(p)
    no_col = [1 1 1];
    full_col = [ 1 0 0 ];
    col = (1 - p)*no_col + p*full_col;
end