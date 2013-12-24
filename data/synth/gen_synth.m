%% Change window

x = linspace(-10, 10, 31)';
[X,Y] = meshgrid(x,x);
x = [X(:), Y(:)];

cov_func = {@covChangeWindowMultiD, 2, {{@covMask, {[1, 0], @covSEiso}}, {@covMask, {[1, 1], @covSEiso}}}};
hyp.cov = [0, 0, 2, 0, 0, 0, 0];

K = feval(cov_func{:}, hyp.cov, x);
K = K + 1e-9*max(max(K))*eye(size(K));

y = chol(K)' * randn(size(x, 1),1);

surf(reshape(y, 31, 31));

X = x;

save('cw1.mat', 'X', 'y');