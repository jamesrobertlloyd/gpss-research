files = dir('./*_predictions.csv');
for file = files'
    data = csvread(file.name, 1, 0);
    actuals = data(:,1);
    predictions = data(:,2);
    save([file.name(1:(end-3)) 'mat'], 'actuals', 'predictions');
end