library(glmnet)
Iters = 25000
burnin = 5000
# random data generation
n = 80  # samples
p = 100   # predictors
s0 = 50    # sparsity


rho <- 0.
Sigma <- outer(1:p, 1:p, function(i, j)rho^abs(i - j))
LL = chol(Sigma) 
X =  matrix( rnorm(p*n), ncol = p) %*% LL
tX = t(X)
#btrue = matrix( 0, nrow = p); btrue[1:s0] = rnorm(s0,sd = 10)
btrue = matrix( rnorm(p,sd = 1), nrow = p)
btrue[1:s0] = rnorm(s0,sd = 3)

# binomial model
mu = X%*%btrue  # + rnorm(n)
Ytrue = rbinom(n = n, size = 1, prob = 1/(1+ exp(-mu)) )
Ytrue[Ytrue==0] <- -1
Y = Ytrue
#Z = sample(c(-1,1),size = n,replace = T,prob =c(0.1, 0.9)) ;Y = Ytrue*Z

### glmnet
cvfit.glmet <- cv.glmnet(X,Y, family = "binomial", type.measure = "class",intercept = FALSE)
beta_glmnet <- as.vector(coef(cvfit.glmet, s = "lambda.min"))[-1]
c(mean( (beta_glmnet - btrue)^2 ) ,mean(sign(X%*%beta_glmnet) != Ytrue) )



### MALA logistic
alpha = .8
B_mala = matrix( 0 ,nrow = p) ;a = 0  ; tau = 1
h = 1/(p)^2.4  # 2.3
M = beta_glmnet
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
    M = tam;  a = a+1  } ;     if (s%%5000==0){ print(s)  }
  if (s>burnin)B_mala = B_mala + M/(Iters-burnin)
}
a/Iters

### LMC with logistic loss
B_lmc = matrix( 0 ,nrow = p) ; h_lmc = h/4
M = beta_glmnet
for(s in 1:Iters){
  exp_YXm = exp(-Y*(X%*%M) )
  M = M + h_lmc*tX%*%(Y*exp_YXm/(1 + exp_YXm) ) -h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
  if (s>burnin)B_lmc = B_lmc + M/(Iters-burnin)
}
c(mean( (beta_glmnet - btrue)^2 ) , mean(sign(X%*%beta_glmnet) != Ytrue) )
c(mean( (B_mala - btrue)^2 ) , mean(sign(X%*%B_mala) != Ytrue))
c(mean( (B_lmc - btrue)^2 ) , mean(sign(X%*%B_lmc) != Ytrue)  )
mean(abs(beta_glmnet - btrue) ) ; mean(abs(B_mala - btrue) ) ;  mean(abs(B_lmc - btrue) ) 
