%% PRNG

seed=1;   % fixing the seed of the random generators
randn('state',seed);
rand('state',seed);

%% Setup x axis

xrange = linspace(0, 1, 500)';

%% Setup kernels

kernels = cell(1,1);
hyps = cell(1,1);

kernels{1} = {@covPeriodicNoDC};
hyps{1} = [0, -2.5, 0];

kernels{2} = {@covProd, {@covSEiso, @covPeriodicNoDC}};
hyps{2} = [-0.5, 0, 0, -2.5, 0];

kernels{3} = {@covProd, {@covSEiso, @covPeriodicNoDC, @covLINscaleshift}};
hyps{3} = [-0.5, 0, 0, -2.5, 0, 0, 0];

kernels{4} = {@covProd, {@covSEiso, @covPeriodicNoDC, @covLINscaleshift}};
hyps{4} = [-0.5, 0, 0, -2.5, 0, 0, 0];

%% Plot

for i = 1:numel(kernels)
    for j = 1:4
        seed=j;   % fixing the seed of the random generators
        randn('state',seed);
        rand('state',seed);
        kernel = kernels{i};
        hyp = hyps{i};
        K = feval(kernel{:}, hyp, xrange);
        y = chol(non_singular(K))' * randn(size(xrange));
        if i == 4
            y = y ./ (1 + exp(50*(xrange - 0.5)));
        end
        figure((i-1)*4+j);
        samples_plot(xrange, y, j);
        save2pdf([ 'trans_samples/draw_' int2str(i) int2str(j) '.pdf'], gcf, 600, true);
        pause(0.01);
        drawnow;
    end
end

%% Close all

close all;