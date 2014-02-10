function samples_plot( xrange, samples, col )
    % Figure settings.
    lw = 2;
    fontsize = 10;
 
    for i = 1:size(samples, 2);
        plot(xrange, samples(:,i), 'Color', colorbrew(col), 'LineWidth', lw); hold on;
    end
    
    % Make plot prettier.  
    xlim([min(xrange), max(xrange)]);
    set( gca, 'XTick', [ 0 ] );
    %set( gca, 'yTick', [ 0 ] );
    set( gca, 'yTick', [] );
    set( gca, 'XTickLabel', '' );
    set( gca, 'yTickLabel', '' );
    %xlabel( '$x$', 'Fontsize', fontsize );
    %ylabel( '$f(x)$', 'Fontsize', fontsize );
    set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
    set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
    %set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off');
    set(gcf, 'color', 'white');
    set(gca, 'YGrid', 'off');
    
    set_fig_units_cm( 5,3 );
end