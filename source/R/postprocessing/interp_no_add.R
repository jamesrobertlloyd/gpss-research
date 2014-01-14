data <- read.csv('interp_no_add.csv', header=TRUE, sep=',')
X <- data[,c(1,2,3,4,5,6)]
X2 <- stack(X)
xcoord <- rep(0, length(X2$ind))
xcoord[X2$ind=="GPSS"]<- 1
xcoord[X2$ind=="SP"]<- 3
xcoord[X2$ind=="TCI"]<- 2
xcoord[X2$ind=="SE"]<- 4
xcoord[X2$ind=="CP"]<- 5
xcoord[X2$ind=="EL"]<- 6

boxplot(X, ylim=c(1,3))
par(new=T)
plot(xcoord, (X2$values), xlim=c(0.5,6.5), ylim=c(1,3), axes=F, ylab="Standardised RMSE", xlab="")