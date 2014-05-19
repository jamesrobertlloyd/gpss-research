# Setup

err.sum <- 0
folds <- 10
# data.name <- 'breast'
# data.name <- 'heart'
data.name <- 'ionosphere'
# data.name <- 'liver'
# data.name <- 'pima'
# data.name <- 'sonar'

for (fold in 1:folds)
{
  # Read data files
  
  train.name <- paste('../../../data/add-class/r_', data.name, '/r_', data.name, '_fold_', fold, '_of_', folds, '-train.csv', sep='')
  test.name  <- paste('../../../data/add-class/r_', data.name, '/r_', data.name, '_fold_', fold, '_of_', folds, '-test.csv', sep='')
  
  data.train <- read.table(train.name, header = FALSE, sep = ',')
  data.test <- read.table(test.name, header = FALSE, sep = ',')
  
  features.train <- data.train[,1:(dim(data.train)[2]-1)]
  features.test  <- data.test[,1:(dim(data.train)[2]-1)]
  targets.train  <- data.train[,dim(data.train)[2]]
  targets.test   <- data.test[,dim(data.train)[2]]
  
  targets.train[targets.train==-1] <- 0
  targets.test[targets.test==-1] <- 0
  
  # Random forest it
  
  library(randomForest)
  
  # Go random forest!
  
  set.seed(1234)
  trees <- 5000
  do.trace <- 5000
  
  rf <- randomForest(features.train, as.factor(targets.train), xtest=features.test, ytest=as.factor(targets.test), do.trace=do.trace, ntree=trees, importance=TRUE, keep.forest=FALSE)
  predictions <- rf$test$votes[,2]
  
  # Score performance
  
  err.rate <- tail(rf$test$err.rate, n=1)[1]
  err.sum <- err.sum + err.rate
}

err <- err.sum / folds

print(err)