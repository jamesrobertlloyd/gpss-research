function kernel_plot( xrange, vals, color_ix )
    % Figure settings.
    lw = 2;
    fontsize = 10;
 
    plot(xrange, vals, 'Color', colorbrew(color_ix), 'LineWidth', lw); hold on;
       
    if all( vals >= 0 ); lowlim = 0; else lowlim = min(vals) * 1.05; end
    % Make plot prettier.  
    xlim([min(xrange), max(xrange)]);
    ylim([min(lowlim,0), max(vals) * 1.05]);
    set( gca, 'XTick', [ 0 ] );
    set( gca, 'yTick', [ 0 ] );
    set( gca, 'XTickLabel', '' );
    %set( gca, 'yTickLabel', '' );
 %   xlabel( '$x - x''$', 'Fontsize', fontsize );
    %xlabel(' ');
    %ylabel( '$k(x, 0)$', 'Fontsize', fontsize );
    set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
    set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
    %set(gca, 'TickDir', 'out')
    set(gca, 'Box', 'off');
    set(gcf, 'color', 'white');
    set(gca, 'YGrid', 'off');
    
    set_fig_units_cm( 4,3 );
end

