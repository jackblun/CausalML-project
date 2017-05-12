#' ---	
#' title: 'Lecture 2.B R Practicum: Double Partialling Out, Basic and With Selection/Regularization'	
#' author: "Victor Chernozhukov"	
#' date: "April, 2016"	
#' output:	
#'   slidy_presentation:	
#'     keep_md: yes	
#' fontsize: 14pt	
#' tidy: true	
#' size: small	
#' ---	
#' 	
#' 	
#' # Illustration of Frisch-Waugh-Lovell Theorem using CPS earnings data	
#' 	
#' 	
#' We are interested in wage discrimination for women here.   	
#' 	
#' 	
#' 	
#' - We first estimate 	
#' $$	
#' Y = D\alpha + W'\beta +\epsilon	
#' $$	
#' using the usual least squares:	
#' 	
#' 	
#'  - We runs a regression of log-wage ($Y$) on	
#'  	
#'  --  female indicator ($D$), and 	
#'  	
#'  -- controls ($X$), which represent education levels, quadratic term in experience, and some	
#' other job-relevant characteristics.	
#' 	
#' *  the target parameter is $\alpha$, which is the gender wage gap	
#' 	
#' ---	
#' 	
#' # Frisch-Waugh-Lovell in CPS | R	
#' 	
#' 	
#' - First, we read in the data	
#' 	
#' 	
rm(list=ls())  #clear working directory	
load("cps2012.Rdata")	#load("cps2012.Rdata") if not online
data = cps2012;	
attach(data);	
colnames(data)	
#' 	
#' * Then run least squares:	
#' 	
#' 	
fmla= lnw ~  female + (widowed + divorced + separated + nevermarried  + hsd08+hsd911+ hsg+cg+ad+mw+so+we+exp1+exp2+exp3)^2	
full.fit= lm(fmla)	# note that ^2 in lm in R doesn't include squares, only cross-terms
#' 	
#' 	
#' * Here $p = \dim(W)$ is 122 $<<$ $n = 29217$	
#' 	
length(full.fit$coef); length(lnw)	
#' 	
#' 	
#' * This is low-dimensional case, and we expect various ways of partialling out to agree.	
#' 	
#' ---	
#' 	
#' # Frisch-Waugh-Lovell in CPS | R	
#' 	
#' 	
#' * Then we estimate $\alpha$ via the double partialling-out:	
#'  $$	
#'  Y = W'\gamma + \tilde Y$$$$ D = W'\pi + \tilde D.$$	
#' * Then by the Frisch-Waugh-Lovell theorem:	
#'  $$	
#'  \tilde Y = \tilde D \alpha + \epsilon.	
#'  $$	
#' 	
fmla.y= lnw~  (widowed + divorced + separated + nevermarried  + hsd08+hsd911+	
  hsg+cg+ad+mw+so+we+exp1+exp2+exp3)^2	
fmla.d= female~  (widowed + divorced + separated + nevermarried  + hsd08+hsd911+ hsg+cg+ad+mw+so+we+exp1+exp2+exp3)^2	
rY= lm(fmla.y)$res    # estimate of tilde Y	
rD= lm(fmla.d)$res    # estimate of tilde D	
partial.fit= lm(rY~rD)  # run OLS of one residual on the other	
#' 	
#' 	
#' - Here "rY" is the name of the residualized $Y$, i.e. the residual that is left from partialling out $W$	
#' 	
#' - Here "rD" is the name of the residualized $D$, i.e. the residual that is left from 	
#' partialling out $W$.	
#' 	
#' - We then regress "rY" on "rD".	
#' 	
#' 	
#' 	
#' 	
#' ---	
#' 	
#' # Frisch-Waugh-Lovell in CPS | R	
#' 	
#' 	
#' - Then we compare the results from full regression and partial regression in a table.	
#' 	
#' 	
library(xtable)	
table= matrix(0, 2, 2)	
table[1,]= summary(full.fit)$coef["female",1:2]   #extract coeff and se on female indicator	
table[2,]= summary(partial.fit)$coef["rD",1:2]  #extract coeff and se on "rD"	
colnames(table)= names(summary(full.fit)$coef["female",])[1:2]	
rownames(table)= c("full reg", "partial reg")	
tab= xtable(table, digits=c(2, 2,7))	
#' 	
#' 	
#' ---	
#' 	
#' # Frisch-Waugh-Lovell in CPS | R	
#' 	
#' 	
#' * Compare the full to partial regression	
#' 	
#' 	
print(tab, type="html")	
#' 	
#' 	
#' 	
#' * The point estimates are numerically equivalent, which is in line with the Frisch-Waugh-Lovell theorem applied to the sample instead of population.	
#' 	
#' * The standard errors are extremely  close -- first-order equivalent -- which is in line with the traditional asymptotics where the number of controls is much smaller than the sample size,  $p << n$.	
#' 	
#' 	
#' ---	
#' 	
#' # What if we try Post-Lasso methods for partialling out? |R 	
#' 	
#' - We load the "hdm" ("High-Dimensional Metrics") package:	
#' 	
#' 	
options(warn=-1)	
library("hdm")	
#' 	
#' 	
#' - Then we do partialling out using (post) lasso.  The "hdm" package uses the post-lasso by default.	
#' 	
#' 	
#' 	
options(warn=-1)	
library(hdm)	
rY= rlasso(fmla.y, post=TRUE)$res  # the default is POST=TRUE	
rD= rlasso(fmla.d, post=TRUE)$res	
partial.fit.lasso= lm(rY~rD)	
#' 	
#' 	
#' -  It is also implemented by the command *rlassoEffect(x,y,d, method="partialling out")* in *hdm*.	
#' 	
#' -  Refs:  Belloni, Chernozhukov, Wang (Annals of Stats, 2014), and Chernozhukov, Hansen, Spindler (2015, Annual Review of Economics).	
#' 	
#' - Here we can do other principled model selection or regularization methods more generally; more	
#' on this later in the course.	
#' 	
#' 	
#' 	
#' ---	
#' 	
#' # What if we try the Lasso method for partialling out? |R 	
#' 	
#' 	
#' - Then we do partialling out using straight-up lasso, instead of post-lasso.	
#' 	
#' 	
options(warn=-1)	
library(hdm)	
rY= rlasso(fmla.y, post=FALSE)$res  # the default is POST=TRUE	
rD= rlasso(fmla.d, post=FALSE)$res	
partial.fit.lasso2= lm(rY~rD)	
#' 	
#' 	
#' -  It is also implemented by the command *rlassoEffect(x,y,d, method="partialling out", post=FALSE)* in *hdm*.	
#' 	
#' -  Refs:  Belloni, Chernozhukov, Wang (Annals of Stats, 2014), and Chernozhukov, Hansen, Spindler (2015, Annual Review of Economics). 	
#' 	
#' - Here we can do other principled model selection or regularization methods more generally.	
#' 	
#' ---	
#' 	
#' #  Put results in the table | R 	
#' 	
#' 	
library(xtable)	
table= matrix(0, 4, 2)	
table[1,]= summary(full.fit)$coef["female",][1:2]	
table[2,]= summary(partial.fit)$coef["rD",][1:2]	
table[3,]= summary(partial.fit.lasso)$coef["rD",][1:2]	
table[4,]= summary(partial.fit.lasso2)$coef["rD",][1:2]	
colnames(table)= names(summary(full.fit)$coef["female",])[1:2]	
rownames(table)= c("full reg", "partial reg", "partial reg via post-lasso", "partial reg via lasso")	
tab= xtable(table, digits=c(2, 2,7))	
#' 	
#' ---	
#' 	
#' # Compare the full reg, partial reg, partial reg via post-lasso and lasso |R	
#' 	
#' 	
print(tab, type="html")	
#' 	
#' 	
#' - We see that all four methods agree on estimates and confidence intervals	
#' 	
#' - This is because in this model the number of controls $p$ is much smaller than the sample size $n$:	
#' $$	
#' p\ll n	
#' $$	
#' 	
#' - When $p$ is comparable to $n$, or larger than $n$, the equivalence will not hold	
#' 	
#' - In particular when $p>n$, the first two methods will break-down, whereas the methods based on lasso will work.	
#' 	
#' 	
#' ---	
#' 	
#' # A More Challenging Example, with $p$ comparable to $n$ | Barro-Lee growth data	
#' 	
#' - Next we try another example, which is more problematic since $p$ is comparable to $n$	
#' 	
#' - This example comes from the Barro-Lee study of cross-country growth rates	
#' 	
#' 	
data(GrowthData)	
growth= GrowthData	
dim(growth)	
colnames(growth)	
attach(growth)	
#' 	
#' 	
#' ---	
#' 	
#' # Barro-Lee Growth Data	
#' 	
#' - Here the outcome ($Y$) is the realized growth rates in per-capita GDP, 	
#' 	
#' - The target variable ($D$) is the initial level of per-capita GDP, labelled as gdpsh465.	
#' 	
#' - The target parameter is $\alpha$, which is the speed of converging, which measures	
#' the speed with which poor countries catch up with rich countries.	
#' 	
#' 	
#' # Analyzing Growth Data | R 	
#' 	
#' -  Perform the fitting by ols and partial ls:	
#' 	
#' 	
fmla=  "Outcome ~ ."	#"." means "all other variables"
full.fit= lm(fmla, data=growth)	
fmla.y= "Outcome ~ . - gdpsh465 "	# "everything but gdpsh465"
fmla.d= "gdpsh465~ . - Outcome"	
# partial fit via ols 	
rY= lm(fmla.y, data =growth)$res	
rD= lm(fmla.d, data =growth)$res	
partial.fit.ls= lm(rY~-1 + rD)	
#' 	
#' 	
#' 	
#' ---	
#' 	
#' # Analyzing Growth Data | R	
#' 	
#' - perform partialling out using (post) lasso	
#' 	
#' 	
# partial ls via lasso	
rY= rlasso(fmla.y, data =growth)$res	
rD= rlasso(fmla.d, data =growth)$res	
partial.fit.lasso= lm(rY~-1+ rD)	
#' 	
#' - This can also be implemented by the command *rlassoEffect(x,y,x, method="partialling out")*	
#' 	
y= as.matrix(growth[,1])          #outcome	
d= as.matrix(growth[,3])          #treatment	
x= as.matrix(growth[,-c(1,2,3)])  #controls	
partial.fit.lasso= rlassoEffect(x, y,d, method="partialling out")	
#' 	
#' 	
#' 	
#' # Analyzing Growth Data | R	
#' 	
#' - A first-order equivalent procedure is implemented by the command *rlassoEffect(x,y,x, method="double selection")*. 	
#' 	
#' - This procedure is using union of controls selected in the lasso regression of Y on X and of D on X	
#' for partiallig out.	
#' 	
#' 	
y= as.matrix(growth[,1])          #outcome	
d= as.matrix(growth[,3])          #treatment	
x= as.matrix(growth[,-c(1,2,3)])  #controls	
partial.fit.lasso2= rlassoEffect(x, y,d, method="double selection")	
#' 	
#' 	
#' ---	
#' 	
#' # Analyzing Growth Data | R	
#' 	
#' - process results (extract results, put them in a table)	
#' 	
#' 	
library(xtable)	
table= matrix(0, 4, 2)	
table[1,]= summary(full.fit)$coef["gdpsh465",1:2]	
table[2,]= summary(partial.fit.ls)$coef["rD",1:2]	
table[3,]= summary(partial.fit.lasso)$coef[1,1:2]	
table[4,]= summary(partial.fit.lasso2)$coef[1,1:2]	
colnames(table)= names(summary(full.fit)$coef["gdpsh465",])[1:2]	
rownames(table)= c("full reg", "partial reg", "partial reg via lasso", "double selection")	
# adjust partial ls standard error by sqrt{n/(n-p)}	
n= dim(growth)[1]; p= dim(growth)[2]	
table[2,2]= table[2,2]*sqrt(n/(n-p))	
tab= xtable(table, digits=c(2, 2,7))	
#' 	
#' 	
#' # # Analyzing Growth Data |  Compare the results 	
#' 	
#' 	
print(tab, type="html")	
#' 	
#' 	
#' * now the results differ dramatically	
#' 	
#' - large p makes least squares without regularization very inefficient	
#' 	
#' - using "double lasso" to partial out twice fixes this problem	
#' 	
#' - using "double selection" as means of regularization fixes this problem	
#' 	
#' - "double selection" and "partialling out" are first-order equivalent	
#' 	
#' # What lies ahead?  | We can use other "high-quality" methods for double partialling out	
#' 	
#' ## Let's try randomforest, why not?  | R	
#' 	
#' * It is a black box (since we haven't introduced it yet), but in the spirit of ML, let's try it out	
#' 	
#' 	
library(randomForest)	
set.seed(1)	
rY  =  y-randomForest(x,y)$predicted 	
set.seed(1)	
rD  =  d-randomForest(x,d)$predicted 	
partial.fit.rf= lm(rY~rD)	
#' 	
#' 	
#' 	
#' - process results (extract results, put them in a table)	
#' 	
#' 	
library(xtable)	
table2= matrix(0, 5, 2)	
table2[1:4,1:2]= table	
table2[5,]= summary(partial.fit.rf)$coef["rD",1:2]	
colnames(table2)= colnames(table)	
rownames(table2)= c("full reg", "partial reg", "partial reg via lasso", "double selection", "partial reg forest")	
tab= xtable(table2, digits=c(2, 2,7))	
#' 	
#' 	
#' ---	
#' 	
#' ## What lies ahead?  | We can use other "high-quality" methods for double partialling out	
#' 	
#' *  print results	
#' 	
#' 	
#' 	
print(tab, type="html")	
#' 	
#' 	
#' * **Appetizer: Can we trust the last line? **	
#' 	
