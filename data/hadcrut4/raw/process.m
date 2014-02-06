%% Load data

data = csvread('HadCRUT-4-2-0-0-monthly-ns-avg-median-only.csv', 1, 0);
X_full = data(:,1);
y_full = data(:,2);

%% Plot

plot(X_full(:,1), y_full, 'o');

%% Save

X = X_full;
y = y_full;

save(['HadCRUT-4-2-0-0-monthly-ns-avg-median-only', '.mat'], 'X', 'y');

%% Subsample

% n = 5000;
% 
% rp = randperm(length(y_full));
% idx = rp(1:n);
% 
% X = X_full(idx,:);
% y = y_full(idx,:);
% 
% save(['quebec-', int2str(n), '.mat'], 'X', 'y');
% 
% plot(X(:,1), y, 'o');