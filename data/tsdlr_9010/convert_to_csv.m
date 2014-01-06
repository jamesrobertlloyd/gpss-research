files = dir('./*mat');
for file = files'
    load(file.name);
    X = double(X);
    y = double(y);
    Xtest = double(Xtest);
    ytest = double(ytest);
    dlmwrite([file.name(1:(end-4)) '-train.csv'], [X,y], 'precision', '%f');
    dlmwrite([file.name(1:(end-4)) '-test.csv'], [Xtest,ytest], 'precision', '%f');
end