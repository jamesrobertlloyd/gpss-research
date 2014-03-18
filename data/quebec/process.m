%% Load data

data = csvread('quebec-births-fixed.csv', 1, 0);
X_full = data(:,1:4);
y_full = data(:,5);

%% Plot
plot(X_full(:,1), y_full, 'o');

% Set random seed.
randn('state',0);
rand('state',0);

%% Subsample
for n = [1000,2000,5000]
    rp = randperm(length(y_full));
    idx = rp(1:n);

    X = X_full(idx,:);
    y = y_full(idx,:);

    save(['quebec-', int2str(n), '.mat'], 'X', 'y');
end

X = X_full;
y = y_full;

save('quebec-all.mat', 'X', 'y');
