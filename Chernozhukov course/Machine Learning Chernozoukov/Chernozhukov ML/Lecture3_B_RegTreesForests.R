#' ---	
#' title: "Trees, Boosted Trees, and Forests"	
#' author: "VC"	
#' date: '`r format(Sys.Date())`'	
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
#' ## Introduction	
#' * Idea: Partition the covariate space into a set of rectangles and then fit a simple model (constant) in each one. So the estimated function is the average of outcomes falling in this rectangle.	
#' 	
#' * Greedy Idea: Use recursive binary partitions, i.e. sequentially we choose variable and corresponding split point, which achieve the best improvement in the fit until some stopping criterion is reached.	
#' 	
#' * Example [cf blackboard]	
#' 	
#' ## Regression Trees	
#' 	
#' * $n$ observations $(x_i,y_i), i=1,\ldots,n, \quad x_i=(x_{i1}, \ldots, x_{ip})'$	
#' 	
#' * Given a partition into $M$ regions $R_1,\ldots,R_M$, 	
#' the regression tree is a prediction rule 	
#' of the form:	
#' 	
#' $$ x \mapsto f(x) = \sum_{m=1}^M c_m I(x \in R_m) $$	
#' 	
#' ----	
#' 	
#' * Minimizing the training error, defined as the sum of squared errors: 	
#' $$\min_f \sum_{m=1}^M \sum_{x_i \in R_m} (y_i-f(x_i))^2$$ leads to 	
#' $$\hat{c}_m=\text{average}(y_i \mid x_i \in R_m)$$	
#' 	
#' * Finding the best partition in terms of minimal training error is generally computational infeasible.	
#' 	
#' * Instead: greedy algorithm	
#' 	
#' ## Regression Trees	
#' 	
#' **Algorithm:**  	
#' 	
#' 	
#' -- 1. We consider a splitting variable $j$ and split point $s$, s.t.	
#' 	$$ \min_{j, s} \left[ \min_{c_1} \sum_{x_i \in R_1(j,s)} (y_i-c_1)^2 	
#' 	+ \min_{c_2} \sum_{x_i \in R_2(j,s)} (y_i-c_2)^2 \right] $$	
#' 	with $R_1(j,s)=\{ w |w_{j} \leq s \}$,  $R_2(j,s)=\{ w|w_{j} > s \}$, where $w$ denote the covariate values for $x_i$'s.  	
#' 	
#' -- 2. Repeat 1. on each of the two resulting regions  	
#' 	
#' -- 3. Stop when some criterion is reached  	
#' 	
#' * Determination of the best pair $(j,s)$ is computationally feasible. 	
#' 	
#' ## Regression Trees	
#' * How large should we grow a tree? Tree size is a tuning parameter governing the model's complexity.	
#' 	
#' * We should choose it to optimize predictive performance.	
#' 	
#' * *Strategy 1*: Split the tree node only if the decrease in the training error is sufficiently high. Too short-sighted (Why?)	
#' 	
#' * *Strategy 2* :  Given a large tree, we stop when some minimal node size is reached   Then we prune the tree by *cost complexity pruning*  	
#' 	
#' 	
#' ## Regression Trees	
#' 	
#' Subtree $T \subset T_0$ is any tree that can be obtained by pruning $T_0$, i.e. successively removing terminal nodes.	
#' 	
#' We denote the terminal nodes by $m=1,\ldots,M(T)$.	
#' 	
#' $N_m=|\{x_i \in R_m\}|$  is the number of obs within node $m$	
#' 	
#' $\hat{c}_m = 1/N_m  \sum_{x_i \in R_m} y_i$  is the within-node prediction	
#' 	
#' $Q_m(T) =   \sum_{x_i \in R_m} (y_i - \hat{c}_m)^2$  is the within-node training error	
#' 	
#' Cost-complexity criterion $$C_{\alpha}(T)=\sum_{m=1}^{M(T)} N_m Q_m(T) + \alpha |T|$$	
#' 	
#' 	
#' 	
#' 	
#' ## Regression Trees	
#' 	
#' * Find $T_{\alpha} \subset T_0$ to minimize $C_{\alpha}(T)$.	
#' 	
#' * It turns out that pruning is a tractable optimization problem, and the sequence of optimal trees for each $\alpha$ form a monotone nested sequence with respect to $\alpha \in [0,\infty)$.	
#' 	
#' * The complexity parameter (aka "CP") $\alpha$ governs the trade-off between the tree size and its goodness of fit to the data.  In particular, the choice $\alpha=0$ yields the full tree $T_0$.	
#' 	
#' * We can choose $\alpha$ by cross validation. Final tree is obtained by $$T_{\hat{\alpha}}$$	
#' 	
#' 	
#' ## Boston Data	
#' 	
#' 	
set.seed(123)	
library(MASS)  #Boston housing data	
library(rpart)	
library(rpart.plot)	
data(Boston)  #506 obs,  14 raw regressors	
help(Boston)	
#' 	
#' 	
#' 	
#' "mdev" is median value of owners home in 1000s	
#' 	
#' "lstat" is  lower status of the population (percent).	
#' 	
#' "rm" is the  average number of rooms per dwelling.	
#' 	
#' "dis" is weighted distance to five employment centres	
#' 	
#' ##  Tree Example using Boston Data	
#' 	
#' 	
set.seed(1)	
train = sample(1:nrow(Boston), nrow(Boston)*1/4) 	
tree.boston=rpart(medv~.,data=Boston,subset=train, cp=.001)	
prp(tree.boston)	
#' 	
#' 	
#' ##  Evaluate the Predictive Performance using Test Data	
#' 	
#' * Compute Test Mean Square Error (Test MSE) and its standard error:	
#' 	
#' 	
yhat=predict(tree.boston,newdata=Boston[-train,])	
boston.test=Boston[-train,"medv"]	
summary(lm((yhat-boston.test)^2~1))$coef[1:2]	
#' 	
#' 	
#' 	
#' 	
#' 	
#' ## Pick the best tree by using the best "cp" parameter	
#' 	
#' 	
#' 	
fit.trees.Boston<- rpart(medv~.,data=Boston,subset=train)	
bestcp<- fit.trees.Boston$cptable[which.min(fit.trees.Boston$cptable[,"xerror"]),"CP"]	
prunedtree.Boston <- prune(fit.trees.Boston,cp=bestcp)	
prp(prunedtree.Boston)	
#' 	
#' 	
#' ## Evaluate the Predictive Performance	
#' 	
#' * Compute Test Mean Square Error (Test MSE) and its standard error:	
#' 	
#' 	
yhat=predict(prunedtree.Boston,newdata=Boston[-train,])	
boston.test=Boston[-train,"medv"]	
summary(lm((yhat-boston.test)^2~1))$coef[1:2]	
#' 	
#' 	
#' ## Trees vs Linear Models |  1	
#' 	
#' *   Reg tree fits linear functional forms:	
#' 	
#' $$	
#' x \mapsto \sum_{m=1}^M c_m  1(x \in R_m)	
#' $$	
#' 	
#' * The functional form is linear in parameters, but nonlinear in $x$.	
#' 	
#' * The dictionary of nonlinear transformations 	
#' $$	
#'  1(x \in R_m), \quad m=1,...,M	
#' $$	
#' is found recursively by partitioning the space.	
#' 	
#' 	
#' ## Trees vs Linear Models |  2	
#' 	
#' * The linear regression models fit functional forms:	
#' 	
#' $$	
#'  x \mapsto \sum_{m=1}^M \beta_m B_m(x)	
#' $$	
#' 	
#' * The dictionary of nonlinear transformations	
#' $$	
#' B_m(x), \quad m=1,..., M	
#' $$	
#' is specified by the empirist.  We've used polynomial transformations and interactions.	
#' 	
#' ## Fit a "simple" linear model by rlasso	
#' 	
#' 	
#' 	
library(hdm)	
fmla<- medv~  (poly(lstat,2) + poly(rm, 2) + poly(dis,2) + poly(nox,2) + . ) ^2	
rlasso.Boston<- rlasso(fmla,data=Boston,subset=train)	
summary(rlasso.Boston,all=FALSE)	
#' 	
#' 	
#' ## Evaluate the Predictive Performance	
#' 	
#' * Compute Test Mean Square Error (Test MSE) and its standard error:	
#' 	
#' 	
yhat.lasso=predict(rlasso.Boston,newdata=Boston[-train,])	
boston.test=Boston[-train,"medv"]	
summary(lm((yhat.lasso-boston.test)^2~1))$coef[1:2]	
#' 	
#' 	
#' 	
#' ## Bagging	
#' 	
#' * Bagging: Bootstrap aggregation or bagging averages	
#' 	
#' * Training data $Z=\{(x_1,y_1),\ldots,(x_n,y_n)\}$. Fit a model to $Z$ and obtain $x \mapsto \hat{f}(x)$	
#' 	
#' * Idea: obtain predictions over many bootstrap samples and average them 	
#' 	
#'   -- $Z^{*b}, b=1,\ldots,B$ bootstrap samples	
#' 	
#'   --  For each $b$ fit model a model to $Z^{*b}$ to get $x \mapsto \hat{f}^{*,b}(x)$	
#' 	
#'   --  $\hat{f}_{bag}(x)=1/B \sum_{b=1}^B \hat{f}^{*,b}(x)$	
#'   	
#'   	
#' ## Bagging	
#' 	
#' * This is seemingly crazy at a first sight... but the idea is that if $Z^{*b}$ were independent draws from the true data generating process, averaging over many  independent prediction rules would be perfectly sensible.	
#' 	
#' * Well suited for high-variance and low-bias procedures; so we should be bagging trees that are relatively deep.
#' 	
#' * Main Application: regression trees	
#' 	
#' * Bagged Trees is particular version of Random Forest.	
#' 	
#' 	
#' ## Example of Bagging on Trees	
#' 	
#' 	
#' 	
library(randomForest)	
set.seed(1)	
bag.boston=randomForest(medv~.,data=Boston,subset=train, mtry=13, ntree=5000) 	
# mtry=13 =equal to the number of variables, which implements bagging trees	
#' 	
#' * Compute Test Mean Square Error (Test MSE) and its standard error:	
#' 	
#' 	
yhat= predict(bag.boston,newdata=Boston[-train,]) 	
summary(lm((yhat-boston.test)^2~1))$coef[1:2]	
#' 	
#' 	
#' 	
#' 	
#' ## Random Forests  	
#' * Introduced by Breiman (2001)	
#' * Very powerful, all-purposes method with many applications	
#' * Modified version of bagging	
#' * Idea: Building a large collection of de-correlated trees and then average them (Breiman, 2001)	
#' 	
#' 	
#' ## Random Forests | Procedure	
#' 	
#' * Bootstrap samples $1,\ldots,B$	
#' 	
#' * Build trees and before each split, select $m \leq p$ -- typically $m < p$ -- of the input variables at random as candidates for splitting, for example, $m=\sqrt{p}$ and $m=1$.	
#' 		
#' * This induces the "decorrelation" of trees, since it precludes the "strong" predictors imposing too much structure on the trunk of the tree.	
#' 	
#' * Aggregation:	
#' 	$$\hat{f}_{rf}(x) = \frac{1}{B} \sum_{b=1}^B \hat f^{*,b}_h(x)$$	
#' Here $h$ is the tuning parameter, which could include $h = m$ and other parameters of the tree-building algorithms.	
#' 	
#' ## Example of Random Forest with m=3	
#' 	
#' 	
library(randomForest)	
set.seed(1)	
bag.boston=randomForest(medv~.,data=Boston,subset=train, mtry=3, ntree=5000) 	
# mtry=3 =equal to the number of variables which we consider splitting, which implements bagging trees	
yhat.rf = predict(bag.boston,newdata=Boston[-train,]) 	
#' 	
#' 	
#' * Compute Test Mean Square Error (Test MSE) and its standard error:	
#' 	
#' 	
summary(lm((yhat.rf-boston.test)^2~1))$coef[1:2]	
#' 	
#' 	
#' 	
#' * Works better than $m=p$.	# Actually it doesn't!
#' 	
#' * In the previous lecture we've picked $m$ by cross-validation. Does better than the "simple" linear model.	
#' 	
#' 	
#' ## Boosting	
#' 	
#' * The idea of boosting is that of a recursive fitting: 	
#' 	
#' We fit a tree, then take the residuals and fit another tree to the residuals, etc.	
#' This implicitly defines a prediction rule.	
#' 	
#' * In order to avoid overfitting we can regularize the procedure in many different ways:	
#' 	
#' - By considering how many steps to make	
#' - By only partially updating the fit.	
#' 	
#' ## Boosting | Algorithm	
#' 	
#' -- 1. Initialize the prediction rule $x \mapsto \hat f(x) = 0$ and $r_i = y_i, i=1,...,n$	
#' 	
#' -- 2. For $j=1,...,J$ 	
#' 	
#' -- (a) fit a tree of depth $d$, $x \mapsto \hat f^j(x)$ to the data $(x_i, r_i)_{i=1}^n$;	
#' 	
#' -- (b) update the residuals $r_i = r_i - \lambda \hat f^j(x_i)$	
#' 	
#' -- 3. Ouput the boosted model	
#' 	
#' $$	
#' x \mapsto \hat f(x) = \sum_{j=1}^J  \lambda \hat f^j(x).	
#' $$	
#' 	
#' ##  Boosting Algorithm Parameters	
#' 	
#' Parameters: 	
#' 	
#' * $\lambda \in (0,1)$ - anti-shrinkage parameter; $\lambda=1$ no shrinkage; $\lambda$ small forces only small updates. $J$ - total number of iterations;  	
#' 	
#' * Either of these could be picked by looking at test errors or cross-validation.	
#' 	
#' Interpretations:	
#' 	
#' * Can be interpreted as the "gradient descent" algorithm from numerical optimization.	
#' 	
#' 	
#' ## Boosting: an example in R	
#' 	
#' 	
options(warn=-1,message=-1)	
library(gbm)	
set.seed(1)	
boost.boston=gbm(medv~.,data=Boston[train,],distribution= "gaussian",n.trees=5000, interaction.depth=4)	
yhat.boost =predict(boost.boston,newdata=Boston[-train,],n.trees=5000)	
#' 	
#' * Compute Test Mean Square Error (Test MSE) and its standard error:	
#' 	
#' 	
summary(lm((yhat.boost-boston.test)^2~1))$coef[1:2]	
#' 	
#' 	
#' 	
#' 	
#' ## Boosting: example in R	
#' 	
#' 	
library(gbm)	
set.seed(1)	
boost.boston=gbm(medv~.,data=Boston[train,],distribution= "gaussian",n.trees=5000, interaction.depth=2)	
yhat =predict(boost.boston,newdata=Boston[-train,],n.trees=5000)	
#' 	
#' 	
#' * Compute Test Mean Square Error (Test MSE) and its standard error:	
#' 	
#' 	
summary(lm((yhat-boston.test)^2~1))$coef[1:2]	
#' 	
#' 	
#' ## Combinining Predictions/ Aggregations/ Ensemble Learning |Introduction	
#' 	
#' * Idea: To build a prediction model by combing the strengths of a collection of simpler base models.	
#' * We consider functions of the form	
#' 	$$ f(x) = \sum_{k=1}^K \alpha_k f_k(x) $$	
#' 	where $f_k$'s denote prediction rules, including possibly a constant.	
#' 	
#' * We can build prediction rules from the training data.  We can figure out a good way to combine them using the test or validation data.	
#' 	
#' 	
#' ## Combinining Predictions/ Aggregations/ Ensemble Learning |Introduction	
#' 	
#' 	
#' * If $K$ is small,  we can figure out the optimal linear combo of the rules using test data $Test$, by simply ... running least squares.	
#' $$	
#' \min_{(\alpha_k)_{k=1}^K} \sum_{i \in Test}  (y_i - \sum_{k=1}^K \alpha_k f_k(x))^2.	
#' $$	
#' 	
#' * If $K$ is large, can do Lasso aggregation instead:	
#' $$	
#' \min_{(\alpha_k)_{k=1}^K} \sum_{i \in Test}  (y_i - \sum_{k=1}^K \alpha_k f_k(x))^2 + \lambda \sum_{k=1}^K | \alpha_k|	
#' $$	
#' 	
#' ## Combinining Predictions/ Aggregations/ Ensemble Learning |Example	
#' 	
#' * Least squares aggregation:	
#' 	
#' 	
summary(lm(boston.test~ -1+ yhat.rf+yhat.boost+yhat.lasso))	
#' 	
#' 	
#' ## Combinining Predictions/ Aggregations/ Ensemble Learning |Example	
#' 	
#' * Lasso aggregation:	
#' 	
#' 	
summary(rlasso(boston.test~ -1+ yhat.rf+yhat.boost+yhat.lasso, post=FALSE))	
#' 	
#' 	
#' 	
#' 	
#' ## Summary and Conclusions	
#' 	
#' * We've seen a slightly more systematic approach to building trees, and ways to improve them by aggregation. 	
#' 	
#' * Bootstrap samples lead to different trees, and averaging them seems like a good idea and works well.   Averaging over bootstrap samples is called bagging!	
#' 	
#' * Random forests actually try to average over decorrelated "deep" trees; decorrelation is achieved by choosing randomly on which variables to split during the recursion.  This results in many "different" trees.	
#' 	
#' * Boosting uses recursive fitting of residuals by a sequence of "shallow" tree models.	
#' 	
#' 	
#' 	
#' 	
#' 	
#' 	