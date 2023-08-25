##### iGP and uGP #####
iGP <- function(X, Y, S,
                GP.settings=list(nu=2.5, g=eps, theta.init=0.1, theta.lower=eps, theta.upper=100),
                parallel=FALSE, n.cores=detectCores(), trace=FALSE){
  
  d <- ncol(X)
  p <- ncol(S)
  n <- nrow(X)
  N <- nrow(S)
  
  ##### GP setting #####
  
  # smoothing parameter for matern kernel
  if(is.null(GP.settings$nu)){
    nu <- 4.5 
  }else{
    nu <- GP.settings$nu
  }
  
  # nugget
  if(is.null(GP.settings$g)){
    g <- eps
  }else{
    g <- GP.settings$g
  }
  
  # theta.init
  if(is.null(GP.settings$theta.init)){
    theta.init <- 0.1
  }else{
    theta.init <- GP.settings$theta.init
  }
  if(length(theta.init)==1) theta.init <- rep(theta.init,d)
  
  # theta.lower
  if(is.null(GP.settings$theta.lower)){
    theta.lower <- eps
  }else{
    theta.lower <- GP.settings$theta.lower
  }
  if(length(theta.lower)==1) theta.lower <- rep(theta.lower,d)
  
  # theta.upper
  if(is.null(GP.settings$theta.upper)){
    theta.upper <- 3
  }else{
    theta.upper <- GP.settings$theta.upper
  }
  if(length(theta.upper)==1) theta.upper <- rep(theta.upper,d)
  
  ##### initialization #####
  if(parallel) {
    cl <- makeCluster(n.cores)
    registerDoParallel(cl)
  }
  
  theta <- matrix(theta.init, ncol=d, nrow=N)
  tau2 <- rep(0, N)
  
  tick <- proc.time()
  ##### (MLE) estimate theta ####
  if(parallel){
    foreach.out <- foreach(k = 1:N, .packages = "plgp", .combine = "rbind", .export = c("matern.kernel","eps")) %dopar% {
      neg.logl <- function(para){
        Phi <- matern.kernel(X, para, nu=nu) + g*diag(1,n)
        ldetPhi <- determinant(Phi, logarithm=TRUE)$modulus
        neg.ll <- (n/2)*log(t(Y[k,]) %*% solve(Phi) %*% Y[k,]+eps) + (1/2)*ldetPhi
        return(neg.ll)
      }
      theta.out <- optim(theta.init, neg.logl,
                         method="L-BFGS-B", lower=theta.lower, upper=theta.upper)
      
      theta.return <- theta.out$par 
      
      Phi <- matern.kernel(X,theta.return,nu=nu) + g*diag(1,n)
      tau2.return <- (t(Y[k,]) %*% solve(Phi) %*% Y[k,])/n
      if(tau2.return<eps) tau2.return <- eps
      
      return(c(theta.return, tau2.return))
    }
    theta <- foreach.out[,1:d,drop=FALSE]
    tau2 <- foreach.out[,d+1]
  }else{
    for(k in 1:N){
      neg.logl <- function(para){
        Phi <- matern.kernel(X, para, nu=nu) + g*diag(1,n)
        ldetPhi <- determinant(Phi, logarithm=TRUE)$modulus
        neg.ll <- (n/2)*log(t(Y[k,]) %*% solve(Phi) %*% Y[k,]+eps) + (1/2)*ldetPhi
        return(neg.ll)
      }      
      theta.out <- optim(theta[k,], neg.logl,
                         method="L-BFGS-B", lower=theta.lower, upper=theta.upper)
      
      theta[k,] <- theta.out$par 
      
      Phi <- matern.kernel(X,theta[k,],nu=nu) + g*diag(1,n)
      tau2[k] <- (t(Y[k,]) %*% solve(Phi) %*% Y[k,])/n
      if(tau2[k]<eps) tau2[k] <- eps
    }
  }
  
  tock <- proc.time()
  if(parallel) stopCluster(cl)
  
  return(list(theta=theta,tau2=tau2,nu=nu,g=g,X=X,  
              d=d,p=p,n=n,N=N, time.elapsed=tock-tick,
              GP.settings=list(nu=nu, g=g, theta.init=theta.init, theta.lower=theta.lower, theta.upper=theta.upper)))
}


predict.iGP <- function(fit, Y, xnew=NULL, sig2.fg=TRUE){
  
  X <- fit$X
  theta <- fit$theta
  tau2 <- fit$tau2
  nu <- fit$nu
  g <- fit$g
  d <- fit$d
  p <- fit$p
  n <- fit$n
  N <- fit$N
  
  if(is.null(xnew)){
    xnew <- X
  }
  
  n.new <- nrow(xnew)
  
  y.pred <- matrix(0, nrow=N, ncol=n.new)
  if(sig2.fg) y.sig2 <- matrix(0, nrow=N, ncol=n.new)
  
  tick <- proc.time()
  for(k in 1:N){
    Phi <- matern.kernel(X,theta[k,],nu=nu) + g*diag(1,n)
    Phi.inv <- solve(Phi)
    Phi_Xx <- matern.kernel(X,theta[k,],nu,xnew)
    y.pred[k,] <- drop(t(Phi_Xx)%*%Phi.inv %*% Y[k,])
    
    if(sig2.fg) y.sig2[k,] <- tau2[k]*(1+g-pmin(1, diag(t(Phi_Xx)%*%Phi.inv%*%Phi_Xx)))
  }
  tock <- proc.time()
  
  if(sig2.fg){
    return(list(mean=y.pred, sig2=y.sig2, time.elapsed=tock-tick))
  }else{
    return(list(mean=y.pred, time.elapsed=tock-tick))
  }
}


uGP <- function(X, Y, S,
                     GP.settings=list(nu=2.5, g=eps, theta.init=0.1, theta.lower=eps, theta.upper=3),
                     trace=FALSE){
  
  d <- ncol(X)
  p <- ncol(S)
  n <- nrow(X)
  N <- nrow(S)
  
  ##### GP setting #####
  
  # smoothing parameter for matern kernel
  if(is.null(GP.settings$nu)){
    nu <- 4.5 
  }else{
    nu <- GP.settings$nu
  }
  
  # nugget
  if(is.null(GP.settings$g)){
    g <- eps
  }else{
    g <- GP.settings$g
  }
  
  # theta.init
  if(is.null(GP.settings$theta.init)){
    theta.init <- 0.1
  }else{
    theta.init <- GP.settings$theta.init
  }
  if(length(theta.init)==1) theta.init <- rep(theta.init,d)
  
  # theta.lower
  if(is.null(GP.settings$theta.lower)){
    theta.lower <- eps
  }else{
    theta.lower <- GP.settings$theta.lower
  }
  if(length(theta.lower)==1) theta.lower <- rep(theta.lower,d)
  
  # theta.upper
  if(is.null(GP.settings$theta.upper)){
    theta.upper <- 3
  }else{
    theta.upper <- GP.settings$theta.upper
  }
  if(length(theta.upper)==1) theta.upper <- rep(theta.upper,d)
  
  tau2 <- rep(0, N)
  
  tick <- proc.time()
  
  ##### (M) update theta ####
  neg.logl <- function(para){
    Phi <- matern.kernel(X, para, nu=nu) + g*diag(1,n)
    Phi.chol <- chol(Phi)
    logPhi.det <- determinant(Phi.chol,logarithm=TRUE)$modulus*2
    neg.ll <- 0
    for(k in 1:N) neg.ll <- neg.ll + 1/2*(logPhi.det+n*log(t(Y[k,]) %*% solve(Phi) %*% Y[k,]+eps))
    return(neg.ll)
  }
  
  theta.out <- optim(theta.init, neg.logl,
                     method="L-BFGS-B", lower=theta.lower, upper=theta.upper)
  
  theta <- theta.out$par 

  Phi <- matern.kernel(X,theta,nu=nu) + g*diag(1,n)
  Phi.inv <- solve(Phi)
  for(k in 1:N) tau2[k] <- (t(Y[k,]) %*% solve(Phi) %*% Y[k,])/n
  tau2 <- pmax(eps, tau2)
   
  tock <- proc.time()
  
  return(list(theta=theta,tau2=tau2,nu=nu,g=g,X=X,  
              d=d,p=p,n=n,N=N, time.elapsed=tock-tick,
              GP.settings=list(nu=nu, g=g, theta.init=theta.init, theta.lower=theta.lower, theta.upper=theta.upper)))
}


predict.uGP <- function(fit, Y, xnew=NULL, sig2.fg=TRUE){
  
  X <- fit$X
  theta <- fit$theta
  tau2 <- fit$tau2
  nu <- fit$nu
  g <- fit$g
  d <- fit$d
  p <- fit$p
  n <- fit$n
  N <- fit$N
  
  if(is.null(xnew)){
    xnew <- X
  }
  
  n.new <- nrow(xnew)
  
  y.pred <- matrix(0, nrow=N, ncol=n.new)
  if(sig2.fg) y.sig2 <- matrix(0, nrow=N, ncol=n.new)
  
  tick <- proc.time()
  Phi <- matern.kernel(X,theta,nu=nu) + g*diag(1,n)
  Phi.inv <- solve(Phi)
  Phi_Xx <- matern.kernel(X,theta,nu,xnew)
  DD <- t(Phi_Xx)%*%Phi.inv
  y.pred <- t(DD%*%t(Y))
  
  if(sig2.fg) {
    y.sig2 <- matrix(1+g-pmin(1, diag(DD%*%Phi_Xx)), nrow=N, ncol=n.new,byrow=TRUE)
    for(k in 1:N) y.sig2[k,] <- tau2[k]*y.sig2[k,]
  }
  tock <- proc.time()
  
  if(sig2.fg){
    return(list(mean=y.pred, sig2=y.sig2, time.elapsed=tock-tick))
  }else{
    return(list(mean=y.pred, time.elapsed=tock-tick))
  }
}
