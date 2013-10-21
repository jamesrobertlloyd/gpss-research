function two_band_plot( x, mu_1, high_1, low_1, mu_2, high_2, low_2 )

    % Figure settings.
    lw = 1.2;
    opacity = 1.0;
    light_blue = [227 237 255]./255;
    light_red = [255 227 237]./255;
    
    % Plot confidence bears.
    jbfill( x', ...
        high_1', ...
        low_1', ...
        light_blue, 'none', 1, opacity); hold on;
    
    set(gca,'Layer','top');  % Stop axes from being overridden.
    
    plot(x, mu_1, 'Color', colorbrew(2), 'LineWidth', lw); hold on;
    plot(x, mu_2, 'Color', colorbrew(3), 'LineWidth', lw); hold on;
    plot(x, high_2, '--', 'Color', colorbrew(3), 'LineWidth', lw); hold on;
    plot(x, low_2, '--', 'Color', colorbrew(3), 'LineWidth', lw); hold on;
    
    % Make plot prettier.
    set(gcf, 'color', 'white');
    set(gca, 'TickDir', 'out');
    
    xlim([min(x), max(x)]);
    
    set_fig_units_cm( 16,8 );
end