#' ---	
#' title: 'Lecture 2.C: Partialling-out and Post-Selection/Regularization Inference for	
#'   IV Models'	
#' author: "VC"	
#' date: "June 20, 2016"	
#' output:	
#'   slidy_presentation:	
#'     keep_md: yes	
#'   ioslides_presentation: default	
#'   beamer_presentation: default	
#' fontsize: 14pt	
#' ---	
#' 	
#' # Two equations linear IV model	
#' The canonical linear structural equation model of Econometrics:	
#' $$  \begin{aligned}	
#'     Y &= \alpha_1 D + \alpha_2' W + U, \quad U \perp (W',Z')',  \\	
#'     D &= \beta_1 Z + \beta_2' W +V, \quad V \perp (W', Z')',	
#'   \end{aligned}	
#' $$	
#' 	
#' Context is given e.g. by Acemoglu, Johnson, Robinson (2001), where	
#' 	
#' * Y is log GDP;	
#' * D is a measure of Protection from Expropriation, a proxy for 	
#' quality of insitutions;	
#' * Z is the log of Settler's mortality;	
#' * W are geographical variables (atitute, latitude squared, continent dummies as well as interactions)	
#' 	
#' # Partialling out	
#'   	
#' * Variables $Y$ and $D$ are endogenous, $W$ are controls, involving transformations	
#' of raw controls, $Z$ is the instrumental variable.  	
#' 	
#' 	
#' * Application of the partialling out operator in population to both sides of each of the equations in \eqref{IVM} gives us a much	
#' simpler system of equations:	
#' 	
#' $$	
#' \begin{array}{l}	
#' \tilde Y = \alpha_1 \tilde D + U, \quad U \perp \tilde Z, \\	
#' \tilde D = \beta_1 \tilde Z + V, \quad V \perp \tilde Z.	
#' \end{array}	
#' $$	
#' 	
#' * The target parameter $\alpha_1$ is identified by IV of $\tilde Y$ on $\tilde D$ using $\tilde Z$	
#' as the instrument.	
#' 	
#' * If $W$'s are high-dimensional, we will rely on Lasso/Post-lasso or other high-quality techniques to partial out $W$'s. 	
#' 	
#' 	
#' # Analyze AJR (2001) | R	
#' 	
library(hdm)	
#' 	
#' *  First we process data:	
#' 	
data(AJR); 	
y = AJR$GDP; 	
d = AJR$Exprop; 	
z = AJR$logMort	
w = model.matrix(~ -1 + (Latitude + Latitude2 + Africa + 	
                   Asia + Namer + Samer)^2, data=AJR)	
dim(w)	
#' 	
#' -----	
#' 	
#' # Analyze AJR (2001) |R	
#' 	
#' * We can first try to use OLS for partialling out. 	
#' 	
#' * We don't expect this to work well since $X$'s 	
#' are high-dimensional. 	
#' 	
#' 	
fmla.y = GDP ~ (Latitude + Latitude2 + Africa + Asia + Namer + Samer)^2	
fmla.d = Exprop ~ (Latitude + Latitude2 + Africa + Asia + Namer + Samer)^2	
fmla.z = logMort ~ (Latitude + Latitude2 + Africa + Asia + Namer + Samer)^2	
rY = lm(fmla.y, data = AJR)$res	
rD = lm(fmla.d, data = AJR)$res	
rZ = lm(fmla.z, data = AJR)$res	
ivfit.lm = tsls(y=rY,d=rD, x=NULL, z=rZ, intercept=FALSE)	
res = cbind(ivfit.lm$coef, ivfit.lm$se)	
colnames(res) = c("est", "se")	
print(res,digits=3)	
#' 	
#' 	
#' * We see that the estimates exhibit large standard errors (they should in fact be larger, see recent work of Newey).  	
#' 	
#' * This is expected because dimension of $X$ is quite large, comparable to the sample size.	
#' 	
#' 	
#' #  Analyze AJR (2001) | R	
#' 	
#' * Then we estimate an IV model with Lasso-based selection/regularization of $X$:	
#' 	
#' 	
options(warn=-1)	
AJR.Xselect = rlassoIV(x=w, d=d, y=y, z=z, select.X=TRUE, select.Z=FALSE) 	
# note that X plays the role of W 	
#' 	
#' 	
#' *  Then we obtain point estimates and standard errors:	
#' 	
summary(AJR.Xselect)	
#' 	
#' 	
#' *  Using these we can then obtain confidence intervals:	
#' 	
confint(AJR.Xselect)	
#' 	
#' 	
#' ----	
#' 	
#' ##  What is this procedure doing?	
#' 	
#' * The above procedure approximates the partialling out in population:	
#' 	
#' * In essence, it approximately partials out $W$ from $Y$, $D$ and $Z$ using Post-Lasso and applies the 2SLS to the residualized quantities. 	
#' 	
#' * Ref: Chernozhukov, Hansen, Spinder (ARE, 2015), which also covers selection/regularizaiont of both $W$ and $Z$, in cases where $Z$ is high-dimensional.	
#' 	
#' # What is this procedure doing?	
#' 	
#' * Partialling out by (pos) Lasso	
#' 	
#' 	
rY = rlasso(fmla.y, data = AJR)$res	
rD = rlasso(fmla.d, data = AJR)$res	
rZ = rlasso(fmla.z, data = AJR)$res	
ivfit.lasso = tsls(y=rY,d=rD, x=NULL, z=rZ, int=FALSE)	
res= cbind(ivfit.lasso$coef, ivfit.lasso$se)	
colnames(res) = c("est", "se")	
print(res, digits=3)	
#' 	
#' 	
#' * This is  what command rlassoIV() is doing. 	
#' 	
#' * In comparison to OLS-based partialling-out, we see precision is improved by doing selection/regularization of $W$'s	
#' 	
#' 	
#' #Summary	
#' 	
#' 	
library(xtable)	
table<- matrix(0, 2, 2)	
table[1,1]<- ivfit.lm$coef	
table[1,2]<- sqrt(ivfit.lm$vcov)	
table[2,1]<- ivfit.lasso$coef	
table[2,2]<- sqrt(ivfit.lasso$vcov)	
colnames(table)<- c("est", "se")	
rownames(table)<- c("IV no selection", "IV double lasso")	
tab<- xtable(table, digits =2)	
#' 	
#' 	
#' 	
#' 	
print(tab, type="html")	
#' 	
#' 	
#' * Traditional IV method done with flexibly geographical controls is very noisy	
#' 	
#' * The "New" IV method provides a less noisy estimate, which is theoreticall justified.	
#' 	
#' 	
#' # Experiments | Let's try regression trees and random forests next... why not?	
#' 	
#' - regression trees:	
#' 	
#' 	
library(rpart)	
library(rpart.plot) 	
  trees  = rpart(y~w)	
  bestcp = trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]	
  ptree  = prune(trees,cp=bestcp)	
 rY  <-  y-predict(ptree,data=x)	
prp(ptree)	
#' 	
#' -------------	
#' 	
#' # Experiments | Tree-based partialling out	
#' 	
#' 	
#' 	
  trees          <- rpart(z~w)	
  bestcp        <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]	
  ptree          <- prune(trees,cp=bestcp)	
rZ<- z - predict(ptree, data=w)	
prp(ptree)	
#' 	
#' -----------	
#' 	
#' # Experiments | Tree-based partialling out	
#' 	
#' 	
 trees          <- rpart(d~w)	
  bestcp        <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]	
  ptree          <- prune(trees,cp=bestcp)	
	
rD<- d - predict(ptree, data=w)	
prp(ptree)	
#' 	
#' 	
#' # Experiments | Tree-based partialling out	
#' 	
#' 	
ivfit.rt<- tsls(y=rY,d=rD, x=NULL, z=rZ,int=FALSE)	
#' 	
#' 	
#' 	
#' 	
#' # Experiments | Forest-based partialling out	
#' 	
#' 	
library(randomForest)	
set.seed(1)	
rY  <-  y-randomForest(w,y)$predicted 	
set.seed(1)	
rD  <-  d-randomForest(w,d)$predicted 	
set.seed(1)	
rZ  <-  z-randomForest(w,z)$predicted 	
ivfit.rf<- tsls(y=rY,d=rD, x=NULL, z=rZ,int=FALSE)	
#' 	
#' 	
#' # Experiments | Comparisons	
#' 	
#' 	
#' 	
#' 	
library(xtable)	
table<- matrix(0, 4, 2)	
table[1,1]<- ivfit.lm$coef	
table[1,2]<- sqrt(ivfit.lm$vcov)	
table[2,1]<- ivfit.lasso$coef	
table[2,2]<- sqrt(ivfit.lasso$vcov)	
table[3,1]<- ivfit.rt$coef	
table[3,2]<- sqrt(ivfit.rt$vcov)	
table[4,1]<- ivfit.rf$coef	
table[4,2]<- sqrt(ivfit.rf$vcov)	
colnames(table)<- c("est", "se")	
rownames(table)<- c("IV no selection", "IV double lasso", "IV double tree", "IV double Forest")	
tab<- xtable(table, digits =2)	
#' 	
#' 	
#' ---	
#' 	
#' # Experiments | Comparisons	
#' 	
#' - print results	
#' 	
#' 	
#' 	
print(tab, type="html")	
#' 	
#' 	
#' - reg trees and random forests (with default options) seem to overfitting	
#' 	
#' 	
#' ---	
#' 	
#' # Experiments | Do the trees and random forests overfit?	
#' 	
#' to demonstrate the point that reg trees overfit	
#' 	
#' 	
set.seed(1)	
x<- cbind(runif(30), runif(30))	
y<- runif(30)	
  trees          <- rpart(y~x)	
  bestcp        <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]	
  ptree          <- prune(trees,cp=bestcp)	
  prp(ptree)	
py <-  predict(ptree)	
#' 	
#' 	
#' -----------------	
#' 	
#' # Experiments | Do the trees and random forests overfit?	
#' 	
#' 	
#' 	
#' 	
library(glmnet)	
ov.rt<- var(py)/var(y)  	
py  <-  randomForest(x,y)$predicted 	
plot(y,randomForest(x,y)$predicted)	
ov.rf<- var(py)/var(y)	
py  <-  y-rlasso(y~x)$res 	
ov.rlasso<- var(py)/var(y)	
py<- predict(cv.glmnet(x,y),newx=x)	
ov.glmnet.lasso<- var(py)/var(y)	
#' 	
#' ----------------------	
#' 	
#' # Experiments | Do the trees and random forests overfit?	
#' 	
#' 	
#' - process results into a table	
#' 	
#' 	
library(xtable)	
table<- matrix(0, 4, 1)	
table[1,1]<- ov.rt	
table[2,1]<- ov.rf	
table[3,1]<- ov.rlasso	
table[4,1]<- ov.glmnet.lasso	
colnames(table)<- c("% overfit")	
rownames(table)<- c("reg tree", "forest", "(post) rlasso", "cv glmnet lasso")	
tab<- xtable(table, digits =2)	
#' 	
#' 	
#' ---	
#' 	
#' # Experiments | Do the trees and random forests overfit?	
#' 	
#' - print results	
#' 	
#' 	
print(tab, type="html")	
#' 	
#' 	
#' - (default) reg trees and random forests seem to overfitting;  	
#' 	
#' so either need to better tune them or use separate sample to avoid estimation	
#' 	
#' ------------	
#' 	
#' 	
#' 	
