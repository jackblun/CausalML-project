#' ---	
#' title: "Neural Nets  and Other Stuff"	
#' author: "VC, May 2016"	
#' date: 'Thanks to M. Demirer for R-code examples'	
#' output:	
#'   slidy_presentation: default	
#'   ioslides_presentation: null	
#'   keep_tex: yes	
#'   mathjax: local	
#'   self_contained: no	
#'   beamer_presentation: default	
#' ---	
#' 	
#' 	
#' 	
#' 	
#' ## Neural Networks | Introduction	
#' * Inspired by the mode of operation of the brain, imitation of the human brain	
#' * Idea: Extract linear combinations of the inputs as contructed regressors, and model the target ($Y$) as a nonlinear function of these regressors	
#' * Fields:  Artificial Intelligence, Econometrics (Hal White),	
#' * What it is: a fancy nonlinear regression	
#' 	
#' 	
#' 	
#' ## Neural Networks	
#' * Large class of models / learning methods	
#' * We focus on single hidden layer  network trained via backpropagation	
#' * Can be seen as nonlinear regression models, neatly representable by network diagrams	
#' * Diagram: playground.tensorflow.org, representing "deep learning" network	
#' 	
#' ## Neural Networks	
#' * $X$ = a vector of inputs	
#' * $(Y_k)_{k=1}^K$ = a vector of outcome variables, being predicted	
#' *	$Z_m= \sigma(\alpha_{0m} + \alpha_m'X)), m=1,\ldots, M$  	
#' *	$T_k = \beta_{0k} + \beta_k'z, k=1,\ldots, K$  	
#' *	$f_k(x) = g_k(T), k=1,\ldots, K$	
#'  	
#' ## Neural Networks	
#' * Activation function: 	
#' 	
#'  -- $\sigma(v)=\frac{1}{1+e^{-v}}$ (sigmoid) (cf blackboard) 	
#' 	
#'  -- $\sigma(v) = \log (1 + exp(x))$ (soft max or "rectified linear unit")	
#' 	
#' * Regression case: $g_k(T)=T_k$;	
#' * Multinomial logit ("classification") case: $g_k(T)=\frac{e^{T_k}}{\sum_{l=1}^K e^{T_l}}$ (softmax fct.)	
#' * Measure of fit $R(\theta)$: sum-of-squared errors (regression), or quasi-log-likelihood for the logit case	
#' * Estimation: $R(\theta)$ by gradient descent (\textquotedblleft back propagation\textquotedblright); regularization is needed to avoid overfitting	
#' 	
#' ## (Classical) Fitting of Neural Networks	
#' * unknown parameters, called weights, $\theta$:	
#' * $\{ \alpha_{0m}, \alpha_m; m=1,2,\ldots, M \}$ $M(p+1)$ weights	
#' * $\{ \beta_{0m}, \beta_m; k=1,2,\ldots, K \}$ $K(p+1)$ weights	
#' * Criterion function: $R(\theta)= \sum_{k=1}^K \sum_{i=1}^N (y_{ik} - f_k(x_i))^2 = \sum_{i=1}^N R_i$	
#' * Derivatives: $\frac{\partial R_i}{\partial \beta_{km}} = -2(y_i-f_k(x_i))g_k'(\beta_k^Tz_i)z_{mi}$	
#' * Analog Derivatives $\frac{\partial R_i}{\partial \alpha_{ml}}$	
#' 	
#' ## (Classical) Fitting of Neural Networks	
#' A gradient descent update at the (r+1)st iteration is given by  	
#' * $\beta_{km}^{(r+1)} =  \beta_{km}^{(r)} - \gamma_r \sum_{i=1}^N \frac{\partial R_i}{\partial  \beta_{km}^{(r)}}$  	
#' * $\alpha_{ml}^{(r+1)} =  \alpha_{ml}^{(r)} - \gamma_r \sum_{i=1}^N \frac{\partial R_i}{\partial  \alpha_{ml}^{(r)}}$  	
#' * $\gamma_r$ learning rate	
#' 	
#' ## Fitting Neural Networks	
#' * Rewrite Derivatives as 	
#' * $\frac{\partial R_i}{\partial  \beta_{km}} = \delta_{ki}z_{mi}$	
#' * $\frac{\partial R_i}{\partial  \alpha_{ml}} = s_{mi}x_{il}$	
#' * $s_{mi}=\sigma'(\alpha^T_m x_i) \sum_{k=1}^K \beta_{km} \delta_{ki}$ (*)	
#' * $\delta_{ki}$ and $s_{mi}$ "errors"	
#'  	
#' ## Fitting Neural Networks	
#' (Standard, classical) Estimation via back-propagation equations:	
#' 	
#' * An updating step involves two passes	
#' * Forward pass: current weights (parameters) are fixed, calculate $\hat{f}_k(x_i)$  	
#' * Backward pass calculate $\delta_{ki}$, back-propogate via (*), calculate gradients and update.  	
#' 	
#' ## Neural Networks	
#' * Starting values: random values near zero. Intuition: model starts out nearly linear and becomes nonlinear as the weights increase.	
#' * Overfitting: to prevent overfitting early stopping and penalization ("weight decay""; $R(\theta) + \lambda \|\theta\|^2_2$	
#' or  $+ \lambda \| \theta \|_1$).	
#' * Scaling of the inputs: large effects on the quality of the final solution. Default: standardization and normalization of of inputs (packages	
#' may not do it for you).	
#' 	
#' 	
#' * Number of hidden units and layers: better to have too many hidden units than too few.  (Flexibility + Regularization!)  The recent	
#' success of having several hidden layers... "deep learning" ("currently beats every problem in AI")	
#' * Multiple Minima: nonconvex criterion function with many local minima (different starting values, average of predictions of collection of neural nets)	
#' 	
#' 	
#' ## Single Layer Neural Net | R Implementation	
#' 	
#' 	
  nnetF <- function(data, train, form_y){	
  datause = data[train,]	
  dataout <- data[-train,]	
  linout=TRUE	
  clas  = FALSE	
  if(clas==TRUE){ linout=FALSE}	
  maxs <- apply(datause, 2, max) 	
  mins <- apply(datause, 2, min)	
  	
  datause <- as.data.frame(scale(datause, center = mins, scale = maxs - mins))	
  dataout <- as.data.frame(scale(dataout, center = mins, scale = maxs - mins))	
  	
  form           <- as.formula(paste(form_y, "~", "." ))	
  	
  nn             <- nnet(form, data=datause, size=10,  maxit=1000, decay=0.01, linout = linout, trace=FALSE)	
  k              <- which(colnames(dataout)==form_y)	
  	
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=datause); 	
  yhatuse        <- predict(nn, datause)*(maxs[k]-mins[k])+mins[k]	
  resuse         <- fit.p$y*((maxs[k]-mins[k])+mins[k]) - yhatuse	
  xuse           <- fit.p$x	
  	
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=dataout); 	
  yhatout        <- predict(nn, dataout)*(maxs[k]-mins[k])+mins[k]	
  resout         <- fit.p$y*(maxs[k]-mins[k])+mins[k] - yhatout	
  xout           <- fit.p$x	
  	
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=nn, min=mins, max=maxs,k=k));	
}	
	
#' 	
#' 	
#' ## Single Layer Neural Net | R Example, Wage Data	
#' 	
#' 	
library(ISLR)  # Introduction to Statistical Learning with R		
library(nnet)	
set.seed(1)	
data(Wage)		
Wage <- subset(Wage,select=-c(wage))		
data = Wage	
Y  =  model.matrix(~logwage-1, data=Wage)		
X  =  model.matrix(~(factor(year)+ age +  race+ maritl + jobclass + education +health+health_ins-1), data=Wage)             # X matrix for neural and forest	
data      = as.data.frame(cbind(Y,X))	
y.name =  "logwage"	
train    <- sample(1:nrow(data),floor(nrow(data)/2))	
# regress everything else in the data on y.name	
fit  <- nnetF(data, train, y.name)	
#' 	
#' 	
#' ## Performance on Test Sample	
#' 	
#' Evaluate Test MSE as before	
#' 	
#' 	
summary(lm((fit$yhatout-data[-train,y.name])^2~1))$coef[1:2]		
#' 	
#' 	
#' ## Hybrid Learning:  Alternating Machines	
#' 	
#' * Instead of doing ensembles, can do iterated machines on the residuals (joint work with M. Demirer)	
#' 	
#' * Can for example, use rLasso to take out a smooth linear trend and then pick up deviations	
#' by Random Forest.	
#' 	
#' 	
#' * Formally,	
#' $$	
#' x\mapsto \hat f(x) = \hat f_{rlasso}(x) + \hat g_{rforest}(x),	
#' $$	
#' where $\hat g_{Rforest}$ is trained on the data $(x_i, r_i), i \in Train$ and	
#' $r_i = Y_i - \hat f_{rlasso}(x_i)$. In principle, can iterate on this procedure.	
#' 	
#' 	
#' ##  rlasso+rforest hybrid | R Example	
#' 	
#' * rlasso service function	
#' 	
#' 	
rlassoF <- function(datause, dataout, form_y, post, logit=FALSE){	
  form           <- as.formula(paste(form_y, "~", "." ))	
  if(logit==FALSE){	
    lasso         <- rlasso(form, data=datause, post = post, intercept=TRUE)	
  }	
  if(logit==TRUE){	
    lasso         <- rlassologit(form, data=datause, post = post, intercept=TRUE)	
  }	
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause); 	
  yhatuse         <- predict(lasso, newdata=fit.p$x, type = "response")	
  resuse          <- fit.p$y - predict(lasso, newdata=fit.p$x, type = "response")	
  xuse            <- fit.p$x	
  	
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=dataout); 	
  yhatout         <- predict(lasso, newdata=fit.p$x, type = "response")  	
  resout          <- fit.p$y - predict(lasso, newdata=fit.p$x, type = "response")	
  xout            <- fit.p$x	
    return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=lasso, yout=fit.p$y));	
}	
#' 	
#' 	
#' ----------	
#' 	
#' * RandomForest service function	
#' 	
#' 	
RF <- function(datause, dataout,  form_x, form_y, x=NA, y=NA, xout=NA, yout=NA, nodesize, mtry= 1, ntree=2000, reg=TRUE, tune=FALSE){	
  yhatout <- NA	
  reuse   <- NA	
  yhatuse <- NA	
  resout  <- NA	
  	
  if(is.null(x)){	
    form           <- as.formula(paste(form_y, "~", "." ))	
    	
    if(tune==FALSE){	
      forest       <- randomForest(form, nodesize=nodesize, mtry=mtry, ntree=ntree,  na.action=na.omit, data=datause)	
    }	
    if(tune==TRUE){	
      fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause); 	
      forest_t        <- tuneRF(x=fit.p$x, y=fit.p$y, mtryStart=floor(sqrt(ncol(fit.p$x))), stepFactor=1.5, improve=0.05, nodesize=5, ntree=ntree, doBest=TRUE, plot=FALSE, trace=FALSE)	
      min             <- forest_t$mtry	
      forest          <- randomForest(form, nodesize=nodesize, mtry=min, ntree=ntree,  na.action=na.omit, data=datause)	
    }	
    	
    fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause); 	
    yhatuse         <- forest$predicted	
    resuse          <- fit.p$y - yhatuse	
    fit.p           <- lm(form,  x = TRUE, y = TRUE, data=dataout);    	
    if(reg==TRUE)  {yhatout         <- predict(forest, dataout, type="response")}	
    if(reg==FALSE) {yhatout         <- predict(forest, dataout, type="prob")[,2]}	
    	
    resout          <- (as.numeric(fit.p$y)-1) - yhatout	
  }	
  	
  if(!is.null(x)){    	
    forest          <- randomForest(x=x, y=y, nodesize=nodesize, ntree=ntree,  na.action=na.omit)	
    yhatuse         <- forest$predicted    	
    resuse          <- y - yhatuse 	
    	
    if(!is.null(xout)){	
      	
      if(reg==TRUE)  {yhatout         <- predict(forest, newdata=xout, type="response")}	
      if(reg==FALSE) {yhatout         <- predict(forest, newdata=xout, type="prob")[,2]}	
      resuse          <- yout - yhatout	
    }  	
  }	
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, model = forest));	
}	
#' 	
#' ----	
#' 	
#' * rlasso-Rforest hybrid	
#' 	
#' 	
RLassoForest <- function(data, dataLasso, train, form_y){	
  datause <- data[train,]	
  dataout <- data[-train,]	
  datauseL <- dataLasso[train,]	
  dataoutL <- dataLasso[-train,]	
  post  = TRUE	
  tune  = TRUE	
  ntree = 1000	
  form           <- as.formula(paste(form_y, "~", "." ))	
  rlasso          <- rlassoF(datauseL, dataoutL, form_y, post=FALSE, logit=FALSE)	
  forest          <- RF(datause=NA, dataout=NA, x=formC(form,datause)$x[,-1], y=rlasso$resuse, xout=formC(form,dataout)$x[,-1], yout=rlasso$resout,  nodesize=5, ntree=ntree, tune=tune)	
  yhatout         <- rlasso$yhatout + forest$yhatout	
  return(list(yhatout = yhatout));	
}	
	
formC <- function(form, data){	
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=data); 	
  return(list(x = fit.p$x, y=fit.p$y));	
}	
	
#' 	
#' 	
#' ## rlasso+rforest | R Example	
#' 	
################ Wage Data ########################	
options(warn=-1)	
library(ISLR)  # Introduction to Statistical Learning with R		
library(hdm)	
library(randomForest)	
set.seed(1)	
data(Wage)		
Wage <- subset(Wage,select=-c(wage))		
data = Wage	
Y  =  model.matrix(~logwage-1, data=Wage)		
X  =  model.matrix(~(factor(year)+ age +  race+ maritl + jobclass + education +health+health_ins-1), data=Wage)             # X matrix for neural and forest	
XL =  model.matrix(~(factor(year)+ age + age^2+ age^3+ race+ sex+ maritl + jobclass + education +health+health_ins-1)^2-1, data=Wage)	 # X matrix for lasso	
data      = as.data.frame(cbind(Y,X))	
dataLasso = as.data.frame(cbind(Y,XL))	
y.name =  "logwage"	
train    <- sample(1:nrow(data),floor(nrow(data)/2))	
# regress everything else in the data on y.name	
# For RlassoForest we input two datasets. dataLasso includes interactions.	
fit.lrf  <- RLassoForest(data,dataLasso, train, y.name)	
#' 	
#' 	
#' ## rlasso+rforest | R Example	
#' 	
#' * Evalute MSE on the Test Sample	
#' 	
summary(lm((fit.lrf$yhatout-data[-train,y.name])^2~1))$coef[1:2]		
#' 	
#' 	
#' * Yay! "Best" MSE "ever" (aka marginally better than others)	
#' 	
#' ## Summary of Prediciton/Function Fitting Methods	
#' 	
#' * We concluded a broad overview of function fitting methods	
#' * Penalized methods for flexible linear (in partameters) models: Lasso etc.	
#' * Tree-based methods for flexible models: bagged, boosted trees and random forests	
#' * Neural nets for flexible non-linear models	
#' * Ensemble methods for combining predictions	
#' * Cross-breeding methods for combining models (e.g. rlasso + rforest, rlasso + ridge=lava) 	
#' 	
#' 	
#' 	
