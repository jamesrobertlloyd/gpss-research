data <- read.csv('interp.csv', header=TRUE, sep=',')
X <- data[,c(1,2,3,4,5,6,7,8,9)]
X2 <- stack(X)
xcoord <- rep(0, length(X2$ind))
xcoord[X2$ind=="ABCD.acc"]<- 1
xcoord[X2$ind=="ABCD.int"]<- 2
xcoord[X2$ind=="SP"]<- 3
xcoord[X2$ind=="TCI"]<- 4
xcoord[X2$ind=="MKL"]<- 5
xcoord[X2$ind=="EL"]<- 6
xcoord[X2$ind=="CP"]<- 7
xcoord[X2$ind=="SE"]<- 8
xcoord[X2$ind=="LN"]<- 9

boxplot(X, ylim=c(1,3), names=rep('',times=9))
mtext("ABCD\naccuracy", side=1, at=1, line=2)
mtext("ABCD\ninterpretability", side=1, at=2, line=2)
mtext("Spectral\nkernels", side=1, at=3, line=2)
mtext("Trend, cyclical\nirregular", side=1, at=4, line=2)
mtext("Bayesian\nMKL", side=1, at=5, line=2)
mtext("Eureqa", side=1, at=6, line=1.5)
mtext("Changepoints", side=1, at=7, line=1.5)
mtext("Squared\nExponential", side=1, at=8, line=2)
mtext("Linear\nregression", side=1, at=9, line=2)
par(new=T)
plot(xcoord, X2$values, xlim=c(0.5, 9.5), ylim=c(1,3), axes=F, ylab="Standardised RMSE", xlab="")