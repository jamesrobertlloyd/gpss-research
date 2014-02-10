%% PRNG

seed=1;   % fixing the seed of the random generators
randn('state',seed);
rand('state',seed);

%% Setup x axis

xrange = linspace(0, 1, 500)';

%% Setup kernels

kernels = cell(1,1);
hyps = cell(1,1);

kernels{1} = {@covChangePointMultiD, {1, {@covSEiso}, {@covSEiso}}};
hyps{1} = [0.5, 2, -1, 0, -3, 0];

kernels{2} = {@covChangePointMultiD, {1, @covSEiso, @covPeriodicNoDC}};
hyps{2} = [0.5, 2, -3, 0, 2, -2.5, 0];

kernels{3} = {@covChangePointMultiD, {1, @covLINone, @covSEiso}};
hyps{3} = [0.5, 2, -2, 0, 0];

kernels{4} = {@covChangePointMultiD, {1, @covPeriodicNoDC, @covPeriodicNoDC}};
hyps{4} = [0.5, 0, 2, -2, 1, 2, -3, 0];

%% Plot

for i = 1:numel(kernels)
    kernel = kernels{i};
    hyp = hyps{i};
    K = feval(kernel{:}, hyp, xrange);
    y = chol(non_singular(K))' * randn(size(xrange));
    figure(i);
    samples_plot(xrange, y, i);
    save2pdf([ 'cp_examples/draw_' int2str(i) '.pdf'], gcf, 600, true);
    pause(0.01);
    drawnow; 
end