%%

X = linspace(-25, 25, 1000)';
K = feval(@covSEiso, [0,0], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-5, 5, 1000)';
K = feval(@covRQiso, [0,0,-10], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-5, 5, 1000)';
K = feval(@covRQiso, [0,0,-8], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-5, 5, 1000)';
K = feval(@covMaterniso, 1, [0,0], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-25, 25, 1000)';
K = feval(@covMaterniso, 3, [-0.3,0], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-5, 5, 1000)';
K = feval(@covMaterniso, 5, [0,0], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-5, 5, 1000)';
K = feval(@covPPiso, 0, [0,0], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-5, 5, 1000)';
K = feval(@covPPiso, 1, [0,0], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-5, 5, 1000)';
K = feval(@covPPiso, 2, [0,0], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);

%%

X = linspace(-5, 5, 1000)';
K = feval(@covPPiso, 3, [0,0], X);
K = K + 1e-6*max(max(K))*eye(size(K));
y = chol(K)' * randn(length(X), 1);
plot(X, y);