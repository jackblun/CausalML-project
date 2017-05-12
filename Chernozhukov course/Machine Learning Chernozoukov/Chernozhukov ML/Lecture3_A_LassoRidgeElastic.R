#' ---	
#' title: "Lasso, Ridge, Elastic Net for Prediction"	
#' author: "VC"	
#' date: "April 25, 2016"	
#' output: slidy_presentation	
#' 	
#' ---	
#' 	
#' 	
#' -----------------------	
#' 	
#' 	
#' 	
#' # Good Prediction and Good Approximation of Reg Function Are Equivalent Problems	
#' 	
#' * Consider outcome $Y$ and covariates  $X$.	
#' 	
#' * Suppose we have a prediction rule $x \mapsto f(x)$, using which	
#' we predict $Y$ by $f(X)$.	
#' 	
#' * If $f$ is a good predicion rule then it is also a good approximation to $\mathbb{E}[Y \mid X]$.	
#' $$	
#' \mathbb{E}[Y - f(X)]^2 = I + II = \mathbb{E}[Y - E(Y \mid X)]^2 + \mathbb{E}[E(Y \mid X) - f(X)]^2	
#' $$	
#' 	
#' * The term $I$ is the irreducible error, and it does not depent on $f$.	
#' 	
#' * If we pick the best approximation rule $f$ over a class $\mathcal{F}$, then we minimize $\mathbb{E}[Y - f(X)]^2$.	
#' 	
#' * If we pick the best prediction rule $f$ over a class $\mathcal{F}$, then we minimize $II$.	
#' 	
#' * The two problems are equivalent. We thus focus on trying to solve the prediction problem.	
#' 	
#' 	
#' 	
#' 	
#' #  Overview of Penalized Regression Methods for Prediction	
#' 	
#' Interested in prediction in the linear model:	
#' $$	
#' Y_i = X_i'\beta_0 + \epsilon_i.	
#' $$	
#' The generic predictor will be of the form $\hat Y_i = X_i'\hat \beta$.	
#' 	
#' - The idea of penalized regression is to avoid overfitting in the sample by penalizing the size of the model through various norms	
#' of coefficients.	
#' 	
#' -  The level of penalization ideally should be picked to approximately minimize the Mean Squared Error:	
#' $$	
#' \text{MSE} = \mathbb{E} (Y_i - \hat Y_i)^2.	
#' $$	
#' 	
#' 	
#' 	
#' 	
#' 	
#' -----------------------	
#' 	
#' # Lasso 	
#' 	
#' Normalize $X_i$'s so that the second empirical moment is 1 for each component, namely $\frac{1}{n} \sum X^2_{ij} =1$. Otherwise need to modify penalty function to get equivariance to rescaling	
#' of components.	
#' 	
#' - LASSO solves:	
#' 	
#' $$ \hat{\beta}(\lambda)=\arg \min_{\beta \in \mathbb{R}^p} \sum (Y_i - X_{i}'\beta)^2 + \lambda ||\beta||_1 $$	
#' 	
#' Choice of penalty level $\lambda$:	
#' 	
#' -- cross-validation (not justified theoretically but works well), implemented in $\mathbf{glmnet}$,	
#' 	
#' -- plug-in method (Belloni et al) (justified theoretically and works well in practice), implemented in $\mathbf{hdm}.$  	
#' 	
#' -- produces a sparse model/ does dimension reduction	
#' 	
#' LASSO is very good for approximately sparse models. 	
#' 	
#' 	
#' ------------------	
#' 	
#' # Post-Lasso	
#' 	
#' - POST-LASSO 	
#' 	
#'  -- refit OLS after Lasso (justified theoretically and works well in practice). \	
#' 	
#' Choice of penalty level $\lambda$:	
#' 	
#' -- cross-validation (not justified theoretically but works well);	
#' 	
#' -- plug-in method (Belloni et al) (justified theoretically and works well in practice), implemented in $\mathbf{hdm}.$  	
#' 	
#' 	
#' 	
#' Post-LASSO is very good for approximately sparse models. \	
#' 	
#' 	
#' 	
#' -----------------	
#' 	
#' # RIDGE 	
#' 	
#' $$ \hat{\beta}(\lambda)=\arg \min_{\beta \in \mathbb{R}^p} \sum (Y_i - X_{i}'\beta)^2 + \lambda ||\beta||^2_2 $$	
#' 	
#' Choice of penalty level: 	
#' 	
#' -- cross-validation (not yet justified theoretically but works well) and 	
#' 	
#' -- plug-in method ($\lambda = \sigma/^2/\| \beta_0\|^2/$) (justified theoretically). CV implemented in $\mathbf{glmnet}$. 	
#' 	
#' 	
#' -- Very good for models with many small coefficients.	
#' 	
#' -- Ridge fit is never sparse; does not do variable selection.	
#' 	
#' 	
#' 	
#' ----------------	
#' 	
#' # ELASTIC NET	
#' 	
#' $$ \hat{\beta}(\lambda_1, \lambda_2)=\arg \min_{\beta \in \mathbb{R}^p} \sum (Y_i - X_{i}'\beta)^2 + \lambda_1 ||\beta||^2_2 + \lambda_2 ||\beta||_1  $$	
#' 	
#' Choice of penalty level: 	
#' 	
#' -- cross-validation (not justified theoretically but works well) and plug-in method	
#'  CV implemented in $\mathbf{glmnet}$. 	
#' 	
#' -- Interpolates between ridge and lasso.  	
#' 	
#' -- Elastic net produces a sparse model (unless $\lambda_2 =0$).	
#' 	
#' 	
#' ----------------------	
#' 	
#' # LAVA	
#' 	
#' 	
#' $$ \hat{\beta}(\lambda_1, \lambda_2)= ( \hat{\gamma}(\lambda_1, \lambda_2) +  \hat{\delta}(\lambda_1, \lambda_2)) = \arg \min_{\gamma + \delta \in \mathbb{R}^p} \sum (Y_i - X_{i}'(\gamma + \delta))^2 + \lambda_1 ||\gamma||^2_2 + \lambda_2 ||\delta ||_1  $$	
#' 	
#' -- Choice of penalty level: cross-validation (not justified theoretically but works well). Interpolates	
#' between ridge and lasso.	
#' 	
#' -- Allows $\textbf{sparse + dense}$ coefficients.	
#' $$	
#' \beta_0  = \underbrace{\gamma_0}_{\text{dense}} +\underbrace{\delta_0}_{\text{sparse}}.	
#' $$	
#' 	
#' -- Some coefficients can be large and some can be small for LAVA to work well.	
#' 	
#' -- Lava never produces a sparse model.	
#' 	
#' 	
#' # Set-up a Monte-Carlo Example  	
#' 	
options(warn=-1,message=-1)	
library(glmnet)  # Package to fit ridge/lasso/elastic net models	
library(hdm) # Package to fit lasso "rigorously"	
library(randomForest)	
library(rpart)	
library(rpart.plot)	
set.seed(1)  # Set seed for reproducibility	
n <- 200  # Number of observations	
p <- 300  # Number of predictors included in model	
beta<- c(4/(1:p)^2)	
x <- matrix(rnorm(n*p), nrow=n, ncol=p)	
y <- x%*%beta + rnorm(n)*4	
R2<- var(x%*%beta)/var(y)	
R2	
#' 	
#' 	
#' # Approximately Sparse Model	
#' 	
plot(1:20, beta[1:20], type="bar", xlab="j", ylab="beta_j", main="Coefficients")	
	
	
#' 	
#' 	
#' # Split Data into Training and Test Sets	
#' 	
# Split data into train (1/2) and test (1/2) sets	
train_rows <- sample(1:n, .5*n)	
x.train <- x[train_rows, ]	
x.test <- x[-train_rows, ]	
	
y.train <- y[train_rows]	
y.test <- y[-train_rows]	
#' 	
#' 	
#' # Fit Lasso, Ridge, Elastic Net	
#' 	
#' 	
fit.rlasso<- rlasso(y.train~x.train)	
fit.rlasso2<- rlasso(y.train~x.train, post=FALSE)	
fit.lasso <- cv.glmnet(x.train, y.train, family="gaussian", alpha=1)	
fit.ridge <- cv.glmnet(x.train, y.train, family="gaussian", alpha=0)	
fit.elnet <- cv.glmnet(x.train, y.train, family="gaussian", alpha=.5)	
#' 	
#' -----	
#' 	
#' # Plotting for GLMNET	
#' 	
#' 
dev.off()
plot(fit.lasso)	
#plot(fit.ridge)	
#plot(fit.elnet)	
#' 	
#' 	
#' ------	
#' 	
#' # Test using Out of Sample Performance	
#' 	
#' -  Compute predictions for test sample	
#' 	
#' 	
yhat.rlasso<- predict(fit.rlasso, newdata=x.test)	
yhat.rlasso2<- predict(fit.rlasso2, newdata=x.test)	
yhat.lasso <- predict(fit.lasso, newx=x.test)	
yhat.ridge <- predict(fit.ridge, newx=x.test)	
yhat.elnet <- predict(fit.elnet, newx=x.test)	
#' 	
#' 	
#' -  Record the MSE (Mean Squared Error) for Test Sample.	
#' 	
#' -  Compute SE (Standard Error) for MSE.	
#' 	
#' -  This is done using OLS of $(Y_i - \hat Y_i)^2$ on the intercept in the test sample.	
#' 	
#' 	
MSE.rlasso= summary(lm((y.test-yhat.rlasso)^2~1))$coef[1:2]	
MSE.rlasso2= summary(lm((y.test-yhat.rlasso2)^2~1))$coef[1:2]	
MSE.lasso = summary(lm((y.test-yhat.lasso)^2~1))$coef[1:2]	
MSE.ridge = summary(lm((y.test-yhat.ridge)^2~1))$coef[1:2]	
MSE.elnet = summary(lm((y.test-yhat.elnet)^2~1))$coef[1:2]	
	
#' 	
#' -------	
#' 	
#' # process results into a table	
#' 	
#' 	
#' 	
library(xtable)	
table<- matrix(0, 5, 2)	
table[1,]<- MSE.rlasso	
table[2,]<- MSE.rlasso2	
table[3,]<- MSE.lasso	
table[4,]<- MSE.ridge	
table[5,]<- MSE.elnet	
  colnames(table)<- c("MSE", "S.E. for MSE")	
rownames(table)<- c("rlasso (post-lasso)", "rlasso (lasso)", "glmnet lasso", "ridge", "elnet")	
tab<- xtable(table, digits =2)	
#' 	
#' ----------	
#' 	
#' # Out-of Sample Prediction Results	
#' 	
#' 	
print(tab, type="latex")	
#' 	
#' 	
#' -  Here we see the MSE (Mean Squared Error) for Test Sample.	
#' 	
#' -  We also see the SE (Standard Error) for MSE.	
#' 	
#' -  95% confidence bands for the true MSE is the estimate of MSE $\pm 2 *SE$ 	
#' 	
#' -  Using this table, we can pick the winner.  This method is almost as good as the oracle choice (Wegkamp, Annals of Statistics).	
#' 	
#' 	
#' 	
#' # Conclusions from MC?	
#' 	
#' The model simulated favors lasso, so we should not conclude that lasso is necessarily better than ridge or elastic net.	
#' 	
#' Theoretical penalization implemented in HDM's rlasso seems to beat ad-hoc cross-validated lasso in GLMNET pack.	
#' 	
#' # Try Methods on Wage Data for 2013 for MidAtlantic Region	
#' 	
#' 	
library(ISLR)  # Introduction to Statistical Learning with R	
library(hdm)	
library(randomForest)	
library(glmnet)	
options(warn=-1)	
#' 	
#' 	
#' # Load the Data (Package ISLR)	
#' 	
#' 	
data(Wage)	
Wage <- subset(Wage,select=-c(wage))	
summary(Wage)	
#' 	
#' 	
#' # Build outcome Y and X	
#' 	
#' 	
Y = Wage$logwage	
X =  model.matrix(~(factor(year)+ poly(age,4) +  race+ maritl + jobclass + education +health+health_ins)^2, data=Wage)	
W =  model.matrix(~(factor(year)+ age +  race+ maritl + jobclass + education +health+health_ins),  data=Wage)	
#' 	
#' 	
#' - $Y$ is outcome	
#' - $W$ are raw predictors	
#' - $X$ are contains technical predictors, consisting of basic predictors, interactions, and square in age 	
#' 	
#' # Split Data into Training and Testing Sample	
#' 	
#' 	
inTrain <- sample(length(Y), length(Y)*(1/2), replace=FALSE)	
y.train<- Y[inTrain]	
y.test<- Y[-inTrain]	
x.train<- X[inTrain,]	
x.test<- X[-inTrain,]	
w.train<- W[inTrain,]	
w.test<- W[-inTrain,]	
#' 	
#' 	
#' 	
#' 	
#' #  Train the linear models on the training sample	
#' 	
#' 	
fit.lm1<- lm(y.train~w.train)	
fit.lm2<- lm(y.train~x.train)	
fit.rlasso<- rlasso(y.train~x.train)	
fit.rlasso2<- rlasso(y.train~ x.train, post=FALSE)	
fit.lasso <- cv.glmnet(x.train, y.train, family="gaussian", alpha=1)	
fit.ridge <- cv.glmnet(x.train, y.train, family="gaussian", alpha=0)	
fit.elnet <- cv.glmnet(x.train, y.train, family="gaussian", alpha=.5)	
#' 	
#' 	
#' 	
#' 	
#' # Print out small OLS	
#' 	
#' 	
summary(fit.lm1)	
#' 	
#' 	
#' 	
#' # Print out large OLS	
#' 	
#' 	
#summary(fit.lm2)	
#' 	
#' 	
#' # Print out Post-Lasso	
#' 	
summary(fit.rlasso,all=FALSE)	
#' 	
#' 	
#' 	
#' # Train a pruned tree	
#' 	
#' 	
set.seed(1)	
fit.trees<- rpart(y.train~w.train)	
 bestcp        <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]	
fit.prunedtree          <- prune(fit.trees,cp=bestcp)	
prp(fit.prunedtree)	
#' 	
#' 	
#' # Train a forest	
#' 	
#' 	
RFfit<- tuneRF(w.train, y.train, mtryStart=floor(sqrt(ncol(w.train))),stepFactor=1.5, improve=0.05, nodesize=5, ntree=2000, doBest=TRUE) 	
min             <- RFfit$mtry	
fit.rf2 <-randomForest(w.train, y.train, nodesize=5, mtry=min, ntree=2000)	
#' 	
#' 	
#' 	
#' 	
#' 	
#' 	
#' # Compute Out of Sample Performance	
#' 	
#' - Carry out predictions for the test sample	
#' 	
#' 	
yhat.lm1<- predict(fit.lm1, newdata=as.data.frame(w.test))	
yhat.lm2<- predict(fit.lm2, newdata=as.data.frame(x.test))	
yhat.rlasso<- predict(fit.rlasso, newdata = x.test)	
yhat.rlasso2<- predict(fit.rlasso2, newdata=x.test)	
yhat.lasso <- predict(fit.lasso, newx  =x.test)	
yhat.ridge <- predict(fit.ridge, newx=x.test)	
yhat.elnet <- predict(fit.elnet, newx=x.test)	
yhat.rf2<- predict(fit.rf2, newdata=w.test)	
yhat.pt<- predict(fit.prunedtree, newdata=as.data.frame(w.test))	
#' 	
#' 	
#' -  Record the MSE (Mean Squared Error) for Test Sample.	
#' 	
#' -  Compute SE (Standard Error) for MSE.	
#' 	
#' -  This is done using OLS of $(Y_i - \hat Y_i)^2$ on the intercept.	
#' 	
#' 	
#' 	
MSE.lm1= summary(lm((y.test-yhat.lm1)^2~1))$coef[1:2]	
MSE.lm2= summary(lm((y.test-yhat.lm2)^2~1))$coef[1:2]	
MSE.rlasso= summary(lm((y.test-yhat.rlasso)^2~1))$coef[1:2]	
MSE.rlasso2= summary(lm((y.test-yhat.rlasso2)^2~1))$coef[1:2]	
MSE.lasso = summary(lm((y.test-yhat.lasso)^2~1))$coef[1:2]	
MSE.ridge = summary(lm((y.test-yhat.ridge)^2~1))$coef[1:2]	
MSE.elnet = summary(lm((y.test-yhat.elnet)^2~1))$coef[1:2]	
MSE.rf2 = summary(lm((y.test-yhat.rf2)^2~1))$coef[1:2]	
MSE.pt = summary(lm((y.test-yhat.pt)^2~1))$coef[1:2]	
#' 	
#' 	
#' 	
#' # process results into a table	
#' 	
#' 	
library(xtable)	
table<- matrix(0, 9, 2)	
table[1,]<- MSE.lm1	
table[2,]<- MSE.lm2	
table[3,]<- MSE.rlasso	
table[4,]<- MSE.rlasso2	
table[5,]<- MSE.lasso	
table[6,]<- MSE.ridge	
table[7,]<- MSE.elnet	
table[8,]<- MSE.rf2	
table[9,]<- MSE.pt	
  colnames(table)<- c("MSE", "S.E. for MSE")	
rownames(table)<- c("ols-small", "ols-large", "rlasso (post-lasso)", "rlasso (lasso)", "glmnet lasso", "ridge", "elnet",  "tuned rf", "pruned tree")	
tab<- xtable(table, digits =4)	
#' 	
#' 	
#' 	
#' -  95% confidence bands for the true MSE is the estimate of MSE $\pm 2 *SE$ 	
#' 	
#' 	
#' 	
#' #  Out-of-Sample Prediction Results for Wage Data	
#' 	
#' - print results	
#' 	
print(tab, type="latex")	
#' 	
#' 	
#' 	
plot(1:length(table[,1]),table[,1], ylim =range (c(table[,1]+ 2*table[,2], table[,1]- 2*table[,2] ) ), ylab="MSE", xlab="method")	
lines(1:length(table[,1]),table[,1]+ qnorm(1-.05/9)*table[,2])	
lines(1:length(table[,1]),table[,1]- qnorm(1-.05/9)*table[,2])	
#' 	
#' 	
#' 	
#' # Should We Draw Some Conclusions?	
#' 	
#' - Simple linear model does not do a good job, because they omit important cross-terms, resulting	
#' in too much bias	
#' 	
#' - Large linear model overfits, because they are too noisy, resulting in too much variance	
#' 	
#' - rlasso (post and straight-up versions)  work very well in the simulation and the empirical example	
#' 	
#' ------	
#' 	
#' - glmnet's Lasso is not bad either	
#' 	
#' - Default Reg Trees don't work well in the empirical example; 	
#' 	
#' - Tuned Random Forest works well -- why does it work?	
#' 	
#' - Would you pick an interpretable simple linear model fitted by rlasso or the linear combo of trees fitted by RF?	
#' 	
#' 	
#' 	
#' 	
#' 	
#' 	
#' 	
