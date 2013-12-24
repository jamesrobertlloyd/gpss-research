%% Load data

data = csvread('quebec-births-standard.csv', 1, 0);
X_full = data(:,1:4);
y_full = data(:,5);

%% Plot

plot(X_full(:,1), y_full, 'o');

%% Subsample

n = 5000;

rp = randperm(length(y_full));
idx = rp(1:n);

X = X_full(idx,:);
y = y_full(idx,:);

save(['quebec-', int2str(n), '.mat'], 'X', 'y');

plot(X(:,1), y, 'o');