function gp_nice_plot
%
% A demo showing what happens if you condition on GPs.
%

addpath(genpath( '../utils' ));
close all;

seed=0;   % fixing the seed of the random generators
randn('state',seed);
rand('state',seed);

% Specify GP model.
sigma = .02;
length_scale = 405;
output_variance = 0.14;

% Make up some data
x = [ 25 80 40 50 ]';
y = [ 10 35 15 25 ]' ./100;

x(2) = [];
y(2) = [];

max_N = length(x);

% Condition on one datapoint at a time.
N = max_N

% Fill in gram matrix
K = NaN(N,N);
for j = 1:N
    for k = 1:N
        K(j,k) = kernel( x(j), x(k), length_scale, output_variance );
    end
end

% Compute inverse covariance
Kn = K + sigma^2 .* diag(ones(N, 1));

% Compute covariance with test points.
xstar = -10:1:110;
Nstar = length(xstar);
Kstar = NaN(Nstar, N);
kfull = NaN(Nstar, Nstar);
for j = 1:Nstar
    for k = 1:N
        Kstar(j,k) = kernel( xstar(j), x(k), length_scale, output_variance);
    end
    kstarstar = kernel( xstar(j), xstar(j), length_scale, output_variance);
    for k = 1:Nstar
        kfull(j,k) = kernel( xstar(j), xstar(k), length_scale, output_variance);
    end
end

% Compute posterior mean and variance.
f = (Kstar / Kn) * y(1:N);
variance = kstarstar - diag((Kstar / Kn) * Kstar');

full_variance = kfull - (Kstar / Kn) * Kstar';

% Plot posterior mean and variance.
figure;


quantiles = 0:0.05:0.5;
opacity = 0.2;


for s = quantiles
    edges = [f+norminv(s, 0, 1).*sqrt(variance); ...
     flipdim(f-norminv(s, 0, 1).*sqrt(variance),1)]; 
    hc1 = fill([xstar'; flipdim(xstar',1)], edges, color_spectrum(2*s), 'EdgeColor', 'none'); hold on;
end    

%set(gca,'Color',[0 0 0]);
%set(gcf,'Color',[0 0 0]);

%set(gcf,'BackgroundColor',[0,0,0]);
%h1 = plot( xstar, f, 'b-', 'Linewidth', 2); hold on;
%h2 = plot( x(1:N), y(1:N), 'kd', 'Linewidth', 2); hold on;


% Add axes, legend, make the plot look nice, and save it.
set_pagewidth(1);
dpi = 600;
upper = 1;
ylim( [-0.5 0.8]);
xlim( [xstar(1), xstar(end)]);

set(gca,'Layer','top')   % Show the axes again

%legend_handle = legend( [hc1 h2 ], { 'GP Posterior Uncertainty', 'Data'}, 'Location', 'SouthEast');
%set_thesis_fonts( gca, legend_handle );
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
%xlabel( '$x$' );
%ylabel( '$f(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 16);


if 1
for n_sample = 1:20
    cur_ixs = sort(randperm(Nstar, 2));
    cur_range = cur_ixs(1):cur_ixs(2);
    sample = mvnrnd( f, full_variance, 1);
    hs = plot( xstar(cur_range), sample(cur_range), '-', 'Color', colorbrew(n_sample), 'Linewidth', 1); hold on;
    %save2pdf(sprintf('simple_sample_figures/1d_posterior_samples_%d.pdf', n_samples), gcf, dpi, true );
end
end

end

function col = color_spectrum(p)
    no_col = [1 1 1];
    full_col = [ 1 0 0 ];
    col = (1 - p)*no_col + p*full_col;
end

function d = kernel(x, y, length_scale, output_variance)
    d = output_variance * exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ length_scale ) ...
      + 3.*output_variance * exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ length_scale / 10 );
end

%{

xstar = 1:1:100;

X = [ 10 40 70 ]';
y = [ 4 10 8 ]';
dy = [ 0 10 0 ]';
all_obs = [ y; dy ];

scale = 100;
kernel = @(x,xp)(mvnpdf(x, xp, diag(ones(size(x,1), 1) .* scale )));
dxkernel = @(x,xp)(-(1/scale).*(x - xp) .* mvnpdf(x, xp, diag(ones(size(x,1), 1) .* scale )));
ddkernel = @(x,xp)((1/scale).*( (1/scale).*(x - xp).*(x - xp) - 1) .* mvnpdf(x, xp, diag(ones(size(x,1), 1) .* scale )));

N = numel(y);
for i = 1:N;
    for j = 1:N
        K(i,j) = kernel( X(i, : ), X(j, :));
        Kdx(i,j) = dxkernel( X(i, : ), X(j, :));
        Kdd(i,j) = ddkernel( X(i, : ), X(j, :));
    end
end

full_K = [ K Kdx; Kdx' Kdd];

for i = 1:numel(xstar)
    fstar(i) = kernel( xstar(i), X )' * inv(K) * y;
    fstar_full(i) = [kernel( xstar(i), X )' dxkernel( xstar(i), X)' ] * inv(full_K) * all_obs;
end

plot( xstar, fstar, 'k-'); hold on;
plot( xstar, fstar_full, 'b-'); hold on;
delta = 5;
for d = 1:numel(dy);
    line( [X(d) - delta, X(d) + delta], [y(d) - dy(d) * delta, y(d) + dy(d) * delta], 'Color', [1 0 0] ); hold on;
end



%}
