##### pcaGP #####
pcaGP <- function(X, Y, prop=0.99,
                  GP.settings=list(nu=2.5, g=eps, theta.init=0.1, theta.lower=eps, theta.upper=100),
                  parallel=FALSE, n.cores=detectCores(), trace=FALSE)
{
  tick <- proc.time()
  pca.out <- prcomp(t(Y), scale = FALSE, center = TRUE)
  n.comp <- which(summary(pca.out)$importance[3,] > prop)[1]
  
  iGP.fit <- iGP(X, t(pca.out$x[,1:n.comp]), matrix(1:n.comp,nrow=n.comp,ncol=1),
                      GP.settings=GP.settings, parallel=parallel, n.cores=n.cores, trace=trace) 
  tock <- proc.time()
  
  return(list(iGP.fit=iGP.fit, pca.out=pca.out, n.comp=n.comp, time.elapsed=tock-tick,
              GP.settings=GP.settings))
  
}

predict.pcaGP <- function(fit, Y, xnew=NULL, sig2.fg=TRUE){
  
  iGP.fit <- fit$iGP.fit
  n.comp <- fit$n.comp
  pca.out <- fit$pca.out
  
  tick <- proc.time()
  iGP.pred <- predict.iGP(iGP.fit, t(pca.out$x[,1:n.comp]), xnew, sig2.fg=sig2.fg)
  pred.recon <- pca.out$rotation[,1:n.comp] %*% iGP.pred$mean
  pred.recon <- t(scale(t(pred.recon), scale = FALSE, center = -1 * pca.out$center))
  s2.recon <- pca.out$rotation[,1:n.comp]^2 %*% iGP.pred$sig2
  tock <- proc.time()
  
  if(sig2.fg){
    return(list(mean=pred.recon, sig2=s2.recon, time.elapsed=tock-tick))
  }else{
    return(list(mean=pred.recon, time.elapsed=tock-tick))
  }
}
  
