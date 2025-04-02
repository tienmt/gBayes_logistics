library(glmnet); library(horseshoenlm); library(ncvreg); library(tictoc)
Iters = 25000
burnin = 5000
# random data generation
n = 50    # samples
p = 100   # predictors
s0 = 5  # sparsity

rho <- 0.
Sigma <- outer(1:p, 1:p, function(i, j)rho^abs(i - j))
LL = chol(Sigma) 
X =  matrix( rnorm(p*n), ncol = p) %*% LL; tX = t(X)
btrue = rep( 0, p); 
btrue[1:floor(s0/2)] = .5 ; btrue[ ( floor(s0/2) +1):s0 ] = -.5

# binomial model
mu = X %*%btrue   # ; m2x = 1/(1+mu^2) ;  pnorm(mu) 
Ytrue = rbinom(n = n, size = 1, prob = 1/(1+ exp(-mu^2)) )   # 
#Ytrue = sign( mu ) ;
yy = Ytrue; yy[yy<0] <-0
Y = Ytrue ; Y[Y==0] <- -1

### glmnet
tic()
cvfit.glmet <- cv.glmnet(X, Ytrue, family = "binomial", type.measure = "class")
beta_glmnet <- as.vector(coef(cvfit.glmet, s = "lambda.min"))[-1]
c(mean( (beta_glmnet - btrue)^2 ) ,mean(sign(X%*%beta_glmnet) != Y) )
toc()
# SCAD
tic()
cvfit <- cv.ncvreg(X, Ytrue, family = "binomial", penalty = 'SCAD', nfolds = 5)
beta_scad <- coef(cvfit, s = "lambda.min" )[-1]
c(mean( (beta_scad - btrue)^2 ) ,mean(sign(X%*%beta_scad) != Y) )
toc()
tic()
# horse shoe
horsH = logiths(z = yy, X, method.tau = "halfCauchy", burn = 500, nmc = 5000)$BetaHat
c(mean( (horsH - btrue)^2 ) ,mean(sign(X%*%horsH) != Y) )
toc()

###############################################################
### MALA logistic
alpha = .98
B_mala = matrix( 0 ,nrow = p) ;a = 0  ; tau = .1
h = 1/(p)^3.8  # 2.3
M = beta_glmnet
tic()
for(s in 1:Iters){
  exp_YXm = exp(-Y*(X%*%M) )*alpha
  tam = M + h*tX%*%(Y*exp_YXm/( 1 + exp_YXm) ) - h*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h)*rnorm(p)
  exp_YXtam = exp(-Y*(X%*%tam) )*alpha
  pro.tam = - sum(log(1+exp_YXtam )) -sum(2*log(tau^2 + tam^2))
  pro.M = - sum(log(1+exp_YXm) ) -sum(2*log(tau^2 + M^2))
  tran.m = -sum((M-tam -h*tX%*%(Y*exp_YXtam/( 1 + exp_YXtam) ) -h*sum(2*log(tau^2 + tam^2)) )^2)/(4*h)
  tran.tam = -sum((tam-M - h*tX%*%(Y*exp_YXm/( 1 + exp_YXm) ) -h*sum(2*log(tau^2 + M^2)) )^2)/(4*h)
  pro.trans = pro.tam+tran.m-pro.M-tran.tam
  if(log(runif(1)) <= pro.trans){
    M = tam;  a = a+1  } ;     if (s%%10000==0){ print(s)  }
  if (s>burnin)B_mala = B_mala + M/(Iters-burnin)
} ;a/Iters
toc()
### LMC with logistic loss
B_lmc = matrix( 0 ,nrow = p) ; h_lmc = h/4
M = beta_glmnet
tic()
for(s in 1:Iters){
  exp_YXm = exp(-Y*(X%*%M) )*alpha
  M = M + h_lmc*tX%*%(Y*exp_YXm/(1 + exp_YXm) ) -h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
  if (s>burnin)B_lmc = B_lmc + M/(Iters-burnin)
}
toc()
c(mean( (beta_glmnet - btrue)^2 ) , mean(sign(X%*%beta_glmnet) != Y) )
c(mean( (B_mala - btrue)^2 ) , mean(sign(X%*%B_mala) != Y))
c(mean( (B_lmc - btrue)^2 ) , mean(sign(X%*%B_lmc) != Y)  )
c(mean( (horsH - btrue)^2 ) ,mean(sign(X%*%horsH) != Y) )
c(mean( (beta_scad - btrue)^2 ) ,mean(sign(X%*%beta_scad) != Y) )
a/Iters
c(mean(abs(beta_glmnet - btrue) ), mean(abs(B_mala - btrue) ),  mean(abs(B_lmc - btrue) ),  mean(abs(horsH - btrue) ) ,  mean(abs(beta_scad- btrue) )  )
c( mean( (X%*%beta_glmnet - X%*%btrue)^2 ) , mean( (X%*%B_mala - X%*%btrue)^2 ) , mean( (X%*%B_lmc - X%*%btrue)^2 ) , 
   mean( (X%*%horsH - X%*%btrue)^2 )  ,mean( (X%*%beta_scad- X%*%btrue)^2 )  ) /  mean( (X%*%btrue)^2 ) 

