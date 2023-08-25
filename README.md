Mesh-clustered Gaussian Process (mcGP) emulator for partial differential
equation systems Problems (Reproducibility)
================
Chih-Li Sung
January 22, 2023

This instruction aims to reproduce the results in the paper “*mcGP:
mesh-clustered Gaussian Process emulator for partial differential
equation systems*” by C.-L. Sung, W. Wang, L. Ding, and X. Wang.

The following results are reproduced in this file

-   Section 5.1: Figures 4, 5, and 7
-   Section 5.2: Figures 8 and 10, and Table 2
-   Section 5.3: Figure 12 and Table 3

Note that here some of the figures are not demonstrated here because
they are plotted by MATLAB with specific tools, such as Figure 15, where
Partial Differential Equation Toolbox needs to be installed on MATLAB.
This file also demonstrate the results that can be reproduced via the
free software `R`.

##### Step 0.1: load functions and packages

``` r
library(R.matlab)
library(scoringRules)
library(gridExtra)
library(ggplot2)
library(plgp)
library(CholWishart)
library(parallel)
library(doParallel)
library(foreach)
source("mcGP.R")                # mcGP
source("GP.R")                  # iGP and uGP for comparison 
source("pcaGP.R")               # PCA GP for comparison 
source("matern.kernel.R")       # matern kernel
```

##### Step 0.2: setting

``` r
eps <- sqrt(.Machine$double.eps) #small nugget for numeric stability
```

## Section 5.1: Reproducing Figure 4

The FEM data for the Poisson’s equation is saved as `.mat` files so we
will need `R.matlab` package to read the data. We first demonstrate our
approach using the FEM data with mesh size 0.2.

``` r
# read input, output data, and the mesh data
matdata <- readMat("matlab/poisson_train/meshsize200/poi1.mat")
X <- matrix(0,ncol=1,nrow=5)
Y <- matrix(0,ncol=5,nrow=length(matdata$u))

for(i in 1L:5){
  matdata <- readMat(paste0("matlab/poisson_train/meshsize200/poi", i,".mat"))
  Y[,i] <- matdata$u         # output data (numeric solutions)
  X[i,1] <- matdata$a[1,1]   # input data
}
S <- t(matdata$nodes)        # mesh locations

X.test <- as.matrix(-0.25,ncol=1)

# perform mcGP 
mcGP.fit <- mcGP(X, Y, S, parallel = TRUE) 
```

    ##   |                                                                              |                                                                      |   0%

``` r
mcGP.pred <- predict.mcGP(mcGP.fit, Y=Y, xnew=X.test, sig2.fg=TRUE)

# plot variational distribution qZ 
Z <- apply(mcGP.fit$q_Z,1,which.max)
select.idx <- which(apply(mcGP.fit$q_Z, 2, max) > 0.1)

gg.out <- vector("list", length(select.idx))
j <- 0
for(i in select.idx){
  j <- j + 1
  qZ.df <- data.frame(cbind(S,qZ=mcGP.fit$q_Z[,i]))
  colnames(qZ.df)[1:2] <- c("V2", "V3")
  gg.out[[j]] <- ggplot(qZ.df, aes(x=V2, y=V3, alpha=qZ)) + 
    geom_point(color="#69b3a2",size=2.5)+xlab(paste0("(",letters[j],") k=",i))+ylab("")+
    scale_alpha_continuous(breaks = c(0.2,0.4,0.6,0.8)) +
    theme(legend.position = "none",
          panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          panel.background = element_blank(), axis.line = element_blank(), 
          axis.ticks = element_blank(), axis.text = element_blank(),
          axis.title=element_text(size=20))
  
}
grid.arrange(gg.out[[1]], gg.out[[2]], gg.out[[3]], gg.out[[4]], ncol = 4)
```

<img src="README_files/figure-gfm/unnamed-chunk-3-1.png" style="display: block; margin: auto;" />

## Section 5.1: Reproducing Figure 5

Now we produce Figure 5.

``` r
# plot hyperparameters in each group
tau2 <- data.frame(tau2=mcGP.fit$tau2,k=as.factor(1:10))
g1 <- ggplot(tau2, aes(x=k, y=tau2, color=k, shape=k)) + 
  geom_point(size=3)+theme_bw()+
  scale_shape_manual(values=seq(0,10))+ylab(expression(tau^2)) + theme(legend.position = "none")
theta.hat <- data.frame(theta=mcGP.fit$theta,k=as.factor(1:10))
g2 <- ggplot(theta.hat, aes(x=k, y=theta, color=k, shape=k)) + 
  geom_point(size=3)+theme_bw()+
  scale_shape_manual(values=seq(0,10))+ theme(legend.position = "none")+
  ylab(expression(theta))
grid.arrange(g1, g2, ncol = 2)
```

<img src="README_files/figure-gfm/unnamed-chunk-4-1.png" style="display: block; margin: auto;" />

##### Section 5.1: Reproducing Figure 7

Compare with other methods with different mesh sizes. It will take a
little bit time.

``` r
# set up
meshsize.vt <- c(0.4,0.2,0.1,0.05,0.025) # try five different mesh sizes
node.size <- rep(0, length(meshsize.vt))

FitTime <- PredTime <- RMSE <- Score <- matrix(0, ncol=5, nrow=length(meshsize.vt))
colnames(FitTime) <- colnames(PredTime) <- colnames(RMSE) <- colnames(Score) <- c("mcGP", "uGP", "iGP", "pcaGP", "FEM")

for(ii in 1:length(meshsize.vt)){
  meshsize <- meshsize.vt[ii]
  matdata <- readMat(paste0("matlab/poisson_train/meshsize", meshsize*1000 ,"/poi1.mat"))
  
  # read training data
  Y <- matrix(0,ncol=5,nrow=length(matdata$u))
  X <- matrix(0,ncol=1,nrow=5)
  for(i in 1L:5){
    matdata <- readMat(paste0("matlab/poisson_train/meshsize", meshsize*1000 ,"/poi", i,".mat"))
    Y[,i] <- matdata$u
    X[i,1] <- matdata$a[1,1]
  }
  S <- t(matdata$nodes)
  node.size[ii] <- nrow(S)
  
  # read test data
  Y.test <- matrix(0,ncol=201,nrow=length(matdata$u))
  X.test <- matrix(0,ncol=1,nrow=201)
  FEM.time <- rep(0, 201)
  for(i in 1:201){
    matdata <- readMat(paste0("matlab/poisson_test/meshsize", meshsize*1000 ,"/poi_test", i,".mat"))
    Y.test[,i] <- matdata$u
    X.test[i,1] <- matdata$a[1,1]
    FEM.time[i] <- matdata$cpu.time
  }
  
  PredTime[ii,"FEM"] <- mean(FEM.time)
  
  # perform mcGP 
  mcGP.fit <- mcGP(X, Y, S, parallel = TRUE) 
  mcGP.pred <- predict.mcGP(mcGP.fit, Y=Y, xnew=X.test, sig2.fg=TRUE)
  FitTime[ii,"mcGP"] <- mcGP.fit$time.elapsed[3]
  PredTime[ii,"mcGP"] <- mcGP.pred$time.elapsed[3]/nrow(X.test)
  
  # perform uGP
  uGP.fit <- uGP(X, Y, S)
  uGP.pred <- predict.uGP(uGP.fit, Y=Y, xnew=X.test, sig2.fg=TRUE)
  FitTime[ii,"uGP"] <- uGP.fit$time.elapsed[3]
  PredTime[ii,"uGP"] <- uGP.pred$time.elapsed[3]/nrow(X.test)
  
  # perform iGP 
  iGP.fit <- iGP(X, Y, S, parallel = TRUE) 
  iGP.pred <- predict.iGP(iGP.fit, Y=Y, xnew=X.test, sig2.fg=TRUE)
  FitTime[ii,"iGP"]<- iGP.fit$time.elapsed[3]
  PredTime[ii,"iGP"] <- iGP.pred$time.elapsed[3]/nrow(X.test)
  
  # perform pcaGP
  pcaGP.fit <- pcaGP(X, Y) 
  pcaGP.pred <- predict.pcaGP(pcaGP.fit, Y=Y, xnew=X.test, sig2.fg=TRUE)
  FitTime[ii,"pcaGP"]<- pcaGP.fit$time.elapsed[3]
  PredTime[ii,"pcaGP"] <- pcaGP.pred$time.elapsed[3]/nrow(X.test) 
 
  for(i in 1:nrow(X.test)){
    Score[ii,"mcGP"] <- Score[ii,"mcGP"] + mean(crps(y=Y.test[,i], family = "mixnorm", 
                                                     m = mcGP.pred$mean.array[,i,], 
                                                     s = sqrt(mcGP.pred$sig2.array[,i,]), w=mcGP.fit$q_Z))
    Score[ii,"uGP"] <- Score[ii,"uGP"] + mean(crps(y = Y.test[,i], family = "normal", 
                                                   mean = uGP.pred$mean[,i], 
                                                   sd = sqrt(uGP.pred$sig2[,i])))
    Score[ii,"iGP"] <- Score[ii,"iGP"] + mean(crps(y = Y.test[,i], family = "normal", 
                                                   mean = iGP.pred$mean[,i], 
                                                   sd = sqrt(iGP.pred$sig2[,i])))
    Score[ii,"pcaGP"] <- Score[ii,"pcaGP"] + mean(crps(y = Y.test[,i], family = "normal", 
                                                   mean = pcaGP.pred$mean[,i], 
                                                   sd = pmax(eps, sqrt(pcaGP.pred$sig2[,i]))))
    RMSE[ii,"mcGP"] <- RMSE[ii,"mcGP"] + mean((Y.test[,i] - mcGP.pred$mean[,i])^2)
    RMSE[ii,"uGP"] <- RMSE[ii,"uGP"] + mean((Y.test[,i] - uGP.pred$mean[,i])^2)
    RMSE[ii,"iGP"] <- RMSE[ii,"iGP"] + mean((Y.test[,i] - iGP.pred$mean[,i])^2)
    RMSE[ii,"pcaGP"] <- RMSE[ii,"pcaGP"] + mean((Y.test[,i] - pcaGP.pred$mean[,i])^2)
  }
  
  Score[ii,] <- Score[ii,]/nrow(X.test) 
  RMSE[ii,] <- sqrt(RMSE[ii,]/nrow(X.test))
}
```

    ##   |                                                                              |                                                                      |   0%  |                                                                              |                                                                      |   0%  |                                                                              |                                                                      |   0%  |                                                                              |                                                                      |   0%  |                                                                              |                                                                      |   0%

``` r
RMSE.df <- stack(data.frame(RMSE[,1:4]))
RMSE.df <- cbind(rep(node.size, 4), RMSE.df)

colnames(RMSE.df) <- c("meshsize", "RMSE", "method")
g1 <- ggplot(RMSE.df, aes(x=log(meshsize), y=log(RMSE), shape=method, color=method)) +
  scale_color_manual(values=c("#F8766D", "#00BA38", "#619CFF", "#C77CFF"))+
  geom_point(size=3)+geom_line()+theme_bw()+ theme(legend.position="none") + xlab("log(N)")
g1 <- g1+scale_shape_manual(values=c(0:2,4))

Score.df <- stack(data.frame(Score[,1:4]))
Score.df <- cbind(rep(node.size, 4), Score.df)
colnames(Score.df) <- c("meshsize", "CRPS", "method")
g2 <- ggplot(Score.df, aes(x=log(meshsize), y=log(CRPS), shape=method, color=method)) +
  scale_color_manual(values=c("#F8766D", "#00BA38", "#619CFF", "#C77CFF"))+
  geom_point(size=3)+geom_line()+theme_bw()+ theme(legend.position="none") + xlab("log(N)")
g2 <- g2+scale_shape_manual(values=c(0:2,4))

FitTime.df <- stack(data.frame(FitTime[,1:4]))
FitTime.df <- cbind(rep(node.size, 4), FitTime.df)
colnames(FitTime.df) <- c("meshsize", "time", "method")
g3 <- ggplot(FitTime.df, aes(x=meshsize/1000, y=time, shape=method, color=method)) +
  scale_color_manual(values=c("#F8766D", "#00BA38", "#619CFF", "#C77CFF"))+
  geom_point(size=3)+geom_line()+theme_bw()+ theme(legend.position="none") + xlab("N (X 1,000)")
g3 <- g3+scale_shape_manual(values=c(0:2,4)) + ylab("fitting time (sec.)")

PredTime.df <- stack(data.frame(PredTime))
PredTime.df <- cbind(rep(node.size, 5), PredTime.df)
colnames(PredTime.df) <- c("meshsize", "time", "method")
g4 <- ggplot(PredTime.df, aes(x=meshsize/1000, y=time, shape=method, color=method)) +
  geom_point(size=3)+geom_line(aes(linetype=method))+theme_bw()+ theme(legend.position="none") + xlab("N (X 1,000)")
g4 <- g4+scale_linetype_manual(values=c("solid", "solid", "solid", "solid", "dashed"))+
  scale_color_manual(values=c("#F8766D", "#00BA38", "#619CFF", "#C77CFF","#000000"))+
  scale_shape_manual(values=c(0:2,4,19)) + ylab("prediction time per run (sec.)")

grid.arrange(g1, g2, g3, g4, ncol = 4)
```

<img src="README_files/figure-gfm/unnamed-chunk-5-1.png" style="display: block; margin: auto;" />
