data <- read.csv('extrap.csv', header=TRUE, sep=',')
X <- data[,c(1,2,4,3,5,6,7)]
X2 <- stack(X)
xcoord <- rep(0, length(X2$ind))
xcoord[X2$ind=="GPSS"]<- 1
xcoord[X2$ind=="GPSS.add"]<- 2
xcoord[X2$ind=="SP"]<- 4
xcoord[X2$ind=="TCI"]<- 3
xcoord[X2$ind=="SE"]<- 5
xcoord[X2$ind=="EL"]<- 6
xcoord[X2$ind=="CP"]<- 7

boxplot(log(X), ylim=c(0,2.5))
par(new=T)
plot(xcoord, log(X2$values), xlim=c(0.5, 7.5), ylim=c(0,2.5), axes=F, ylab="Log Standardised RMSE", xlab="")