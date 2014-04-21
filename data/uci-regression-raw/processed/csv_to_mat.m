%% Concrete

concrete = csvread('concrete.csv');
X = concrete(:,1:(end-1));
y = concrete(:,end);
save('../../uci-regression/concrete.mat','X','y');

%% Housing

housing = csvread('housing.csv');
X = housing(:,1:(end-1));
y = housing(:,end);
save('../../uci-regression/housing.mat','X','y');