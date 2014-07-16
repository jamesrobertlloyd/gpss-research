function mean_var_plot( xdata, ydata, xrange, forecast_mu, forecast_scale, data_only, no_data )

    if nargin < 6; data_only = false; end
    if nargin < 7; no_data = false; end

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
    
    if ~no_data
        plot( xdata, ydata, 'k.', 'MarkerSize', 20);
    end
 
    if ~data_only
        % Plot mean function.
        plot(xrange, forecast_mu, 'Color', colorbrew(2), 'LineWidth', lw); hold on;
    end
    
    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'TickDir', 'out');
    
    xlim([min(xrange), max(xrange)]); 
    
    set_fig_units_cm( 16,8 );
end