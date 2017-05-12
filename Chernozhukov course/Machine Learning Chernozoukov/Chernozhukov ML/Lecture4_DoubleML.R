###########################################################################################
#  Program:  Functions for estimating moments for using Machine Learning Methods.         #
#  Reference:  "Double Machine Learning for Causal and Treatment Effects",  MIT WP 2016   #
#  by V.Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey           #
#  These are preliminary programs that implement a 2-way split version of the estimators. #
###########################################################################################


############### Part I: MC Algorithms: Rlasso, Tree, Neuralnet, Nnet, Boosting, Random Forest, Lava ################### 


rlassoF <- function(datause, dataout, form_x, form_y, post, logit=FALSE){
  
  form            <- as.formula(paste(form_y, "~", form_x));
  
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

tree <- function(datause, dataout, form_x, form_y){
  
  form           <- as.formula(paste(form_y, "~", form_x));
  trees          <- rpart(form, data=datause)
  bestcp         <- trees$cptable[which.min(trees$cptable[,"xerror"]),"CP"]
  ptree          <- prune(trees,cp=bestcp)
  
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=datause); 
  yhatuse        <- predict(ptree, newdata=datause)
  resuse         <- fit.p$y - yhatuse
  xuse           <- fit.p$x
  
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=dataout); 
  yhatout         <- predict(ptree, newdata=dataout)
  resout          <- fit.p$y - yhatout
  xout            <- fit.p$x
  
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=ptree));
  
}

nnetF <- function(datause, dataout, form_x, form_y, clas=FALSE){
  
  linout=FALSE
  if(clas==TRUE){ linout=FALSE}
  maxs <- apply(datause, 2, max) 
  mins <- apply(datause, 2, min)
  
  datause <- as.data.frame(scale(datause, center = mins, scale = maxs - mins))
  dataout <- as.data.frame(scale(dataout, center = mins, scale = maxs - mins))
  
  form           <- as.formula(paste(form_y, "~", form_x))
  
  nn             <- nnet(form, data=datause, size=8,  maxit=1000, decay=0.01, linout = linout, trace=FALSE)
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

boost <- function(datause, dataout, form_x, form_y, bag.fraction = .5, interaction.depth=2, n.trees=1000, shrinkage=.01, distribution='gaussian'){
  
  form           <- as.formula(paste(form_y, "~", form_x));
  boostfit       <- gbm(form,  distribution=distribution, data=datause, bag.fraction = bag.fraction, interaction.depth=interaction.depth, n.trees=n.trees, shrinkage=shrinkage ,verbose = FALSE,cv.folds=10)
  best           <- gbm.perf(boostfit,plot.it=FALSE,method="cv")
  
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=datause); 
  yhatuse        <- predict(boostfit, n.trees=best)
  resuse         <- fit.p$y - yhatuse
  xuse           <- fit.p$x
  
  fit.p          <- lm(form,  x = TRUE, y = TRUE, data=dataout); 
  yhatout        <- predict(boostfit, n.trees=best, newdata=dataout,  type="response")
  resout         <- fit.p$y - yhatout
  xout           <- fit.p$x
  
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, xuse=xuse, xuse=xout, model=boostfit, best=best));
}

RF <- function(datause, dataout,  form_x, form_y, x=NA, y=NA, xout=NA, yout=NA, nodesize, ntree, reg=TRUE, tune=FALSE){
  
  yhatout <- NA
  reuse   <- NA
  yhatuse <- NA
  resout  <- NA
  
  
  if(is.na(x)){
    form            <- as.formula(paste(form_y, "~", form_x));
    
    if(tune==FALSE){
      forest       <- randomForest(form, nodesize=nodesize, ntree=ntree,  na.action=na.omit, data=datause)
    }
    if(tune==TRUE){
      fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause); 
      forest_t        <- tuneRF(x=fit.p$x, y=fit.p$y, mtryStart=floor(sqrt(ncol(fit.p$x))), stepFactor=1.5, improve=0.05, nodesize=5, ntree=ntree, doBest=TRUE, plot=FALSE, trace=FALSE)
      min             <- forest_t$mtry
      forest          <- randomForest(form, nodesize=nodesize, mtry=min, ntree=ntree,  na.action=na.omit, data=datause)
    }
    
    fit.p           <- lm(form,  x = TRUE, y = TRUE, data=datause); 
    yhatuse         <- as.numeric(forest$predicted)
    resuse          <- as.numeric(fit.p$y) -  yhatuse
    fit.p           <- lm(form,  x = TRUE, y = TRUE, data=dataout);    
    if(reg==TRUE)  {yhatout         <- predict(forest, dataout, type="response")}
    if(reg==FALSE) {yhatout         <- predict(forest, dataout, type="prob")[,2]}
    
    resout          <- (as.numeric(fit.p$y)-1) - as.numeric(yhatout)
  }
  
  if(!is.na(x)){    
    forest          <- randomForest(x=x, y=y, nodesize=nodesize, ntree=ntree,  na.action=na.omit)
    yhatuse         <- as.numeric(forest$predicted)   
    resuse          <- y - yhatuse 
    
    if(!is.na(xout)){
      
      if(reg==TRUE)  {yhatout         <- predict(forest, newdata=xout, type="response")}
      if(reg==FALSE) {yhatout         <- predict(forest, newdata=xout, type="prob")[,2]}
      resuse          <- yout - as.numeric(yhatout)
    }  
  }
  return(list(yhatuse = yhatuse, resuse=resuse, yhatout = yhatout, resout=resout, model = forest));
}

############# Auxilary Functions  ########################################################;

error <- function(yhat,y){
  
  err         <- sqrt(mean((yhat-y)^2))
  mis         <- sum(abs(as.numeric(yhat > .5)-(as.numeric(y))))/length(y)   
  
  return(list(err = err, mis=mis));
  
}

formC <- function(form_y,form_x, data){
  
  form            <- as.formula(paste(form_y, "~", form_x));    
  fit.p           <- lm(form,  x = TRUE, y = TRUE, data=data); 
  
  return(list(x = fit.p$x, y=fit.p$y));
}


ATE <- function(y, d, my_d1x, my_d0x, md_x)
{
  return( mean( (d * (y - my_d1x) / md_x) -  ((1 - d) * (y - my_d0x) / (1 - md_x)) + my_d1x - my_d0x ) );
}



SE.ATE <- function(y, d, my_d1x, my_d0x, md_x)
{
  return( sd( (d * (y - my_d1x) / md_x) -  ((1 - d) * (y - my_d0x) / (1 - md_x)) + my_d1x - my_d0x )/sqrt(length(y)) );
}


# this function estimates ATE using interactive linear and partially linear models for given methods

DoubleML_DR <- function(data, y, d, xx, xL, method, ite=1){
  
  TE      <- matrix(0,ite,length(method))
  LTE     <- matrix(0,ite,length(method))
  STE     <- array(0,dim=c(ite,length(method),2))
  result  <- matrix(0,2,length(method))
  resultL <- matrix(0,2,length(method))
  
  for(i in 1:ite){
    
    samp1   <- sample(1:nrow(data),floor(nrow(data)/2))
    
    for(k in 1:length(method)){   
      
      print(paste("  estimating",method[k]))
      
      if (method[k]=="RLasso" || method[k]=="PostRLasso"){
        x=xL
      } else {
        x=xx
      }
      
      datause <- data[samp1,]
      dataout <- data[-samp1,]
      
      for(j in 1:2){   
        
        if(j==2){
          datause = data[-samp1,]
          dataout = data[samp1,]  
        }
        
        cond.comp              <- cond_comp(datause=datause, dataout=dataout, y, d, x, method[k], linear=0);
        
        drop                   <- which(cond.comp$mz_x>0.01 & cond.comp$mz_x<0.99)      
        cond.comp$mz_x         <- cond.comp$mz_x[drop]
        cond.comp$my_z1x       <- cond.comp$my_z1x[drop]
        cond.comp$my_z0x       <- cond.comp$my_z0x[drop]
        yout                   <- dataout[drop,y]
        dout                   <- dataout[drop,d]
        
        TE[i,k]                <- ATE(yout, dout, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x)/2 + TE[i,k];
        STE[i,k,j]             <- (SE.ATE(yout, dout, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x))^2;
        
        lm.fit.ry              <- lm(as.matrix(cond.comp$ry) ~ as.matrix(cond.comp$rz)-1);
        ate                    <- lm.fit.ry$coef;
        HCV.coefs              <- vcovHC(lm.fit.ry, type = 'HC');
        se.ate                 <- (diag(HCV.coefs)); # White std errors
        
        LTE[i,k]               <- ate/2 + LTE[i,k] ;
      }  
    }
  }  
  
  colnames(result)   <- method
  
  rownames(result)   <- cbind("mean", "se")
  
  result[1,] <- colMeans(TE)
  result[2,] <- sqrt(.25*STE[1,,1] + .25*STE[1,,2])
  
  return(result)
}  

# causalML_linear only does partially linear ATE estimation by partialling out using given methods

DoubleML_PO <- function(data, y, d, xx, xL, method, ite=1){
  
  TE      <- matrix(0,ite,length(method))
  LTE     <- matrix(0,ite,length(method))
  STE     <- array(0,dim=c(ite,length(method),2))
  result  <- matrix(0,2,length(method))
  resultL <- matrix(0,2,length(method))
  
  for(i in 1:ite){
    
    samp1   <- sample(1:nrow(data),floor(nrow(data)/2))
    
    for(k in 1:length(method)){   
      
      print(paste("  estimating",method[k]))
      if (method[k]=="RLasso" || method[k]=="PostRLasso"){
        x=xL
      } else {
        x=xx
      }
      
      datause <- data[samp1,]
      dataout <- data[-samp1,] 
      
      for(j in 1:2){   
        
        if(j==2){
          datause = data[-samp1,]
          dataout = data[samp1,]  
        }
        
        cond.comp              <- cond_comp(datause=datause, dataout=dataout, y, d, x, method[k], linear=1);
        
        lm.fit.ry              <- lm(as.matrix(cond.comp$ry) ~ as.matrix(cond.comp$rz)-1);
        ate                    <- lm.fit.ry$coef;
        HCV.coefs              <- vcovHC(lm.fit.ry, type = 'HC');
        STE[i,k,j]             <- (diag(HCV.coefs))
        if(method[k]=="Forest"){STE[i,k,j]             <- (summary(lm.fit.ry)$coef[2])^2}
        
        LTE[i,k]               <- ate/2 + LTE[i,k] ;
      }  
    }
  }  
  
  colnames(resultL) <- method
  
  rownames(resultL)  <- cbind("ATE", "sd")
  
  resultL[1,] <- colMeans(LTE)
  resultL[2,] <- sqrt(.25*STE[1,,1] + .25*STE[1,,2])
  
  return(resultL)
} 

# causalBestML does nonparametric estimation by using the method that performs best in terms of prediction MSE error

causalBestML <- function(data, y, d, xx, xL, method, ite=1){
  
  TE       <- matrix(0,ite,1)
  STE      <- array(0,dim=c(ite,1,2))
  result   <- matrix(0,2,1)
  
  for(i in 1:ite){
    
    samp1   <- sample(1:nrow(data),floor(nrow(data)/2))
    
    
    for(j in 1:2){   
      
      print(paste("  running subsample:",j))
      
      err       <- matrix(0,3,length(method))
      cond.comp <- list()
      
      datause <- data[samp1,]
      dataout <- data[-samp1,]
      
      if(j==2){
        datause = data[-samp1,]
        dataout = data[samp1,]  
      }
      
      for(k in 1:length(method)){  
        
        print(paste("    estimating",method[k]))
        if (method[k]=="RLasso" || method[k]=="PostRLasso"){
          x=xL
        } else {
          x=xx
        }
        
        cond.comp[[k]]              <- cond_comp(datause=datause, dataout=dataout, y, d, x, method[k],linear=0);
        err[1,k]                    <- cond.comp[[k]]$err.yz0
        err[2,k]                    <- cond.comp[[k]]$err.yz1
        err[3,k]                    <- cond.comp[[k]]$err.z
      }  
      
      p <- which.min(err[1,])
      l <- which.min(err[2,])
      m <- which.min(err[3,])
      
      print(paste("  best method for E[Y|X, D=0]:", method[p]))
      print(paste("  best method for E[Y|X, D=1]:", method[l]))
      print(paste("  best method for E[D|X]:", method[m]))
      
      my_z0x <- cond.comp[[p]]$my_z0x
      my_z1x <- cond.comp[[l]]$my_z1x
      mz_x   <- cond.comp[[m]]$mz_x
      
      drop                   <- which(mz_x>0.01 & mz_x<0.99)      
      cond.comp$mz_x         <- mz_x[drop]
      cond.comp$my_z1x       <- my_z1x[drop]
      cond.comp$my_z0x       <- my_z0x[drop]
      yout                   <- dataout[drop,y]
      dout                   <- dataout[drop,d]
      
      TE[i,1]                <- ATE(yout, dout, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x)/2 + TE[i,1];
      STE[i,1,j]             <- (SE.ATE(yout, dout, cond.comp$my_z1x, cond.comp$my_z0x, cond.comp$mz_x))^2;
    }  
  }
  
  rownames(result)  <- cbind("ATE", "sd")
  
  colnames(result)  <- "ATE"
  
  result[1] <- mean(TE)
  result[2] <- sqrt(.25*STE[1,1,1] + .25*STE[1,1,2])
  
  return(result)
}  

# predictML does out of sample prediction and returns prediction error

predictML <- function(data, y, xx, xL, method, ite=1){
  
  MSE    <- matrix(0,ite,length(method))
  SE     <- array(0,dim=c(ite,length(method),2))
  result <- matrix(0,2,length(method))
  
  for(i in 1:ite){
    
    samp1   <- sample(1:nrow(data),floor(nrow(data)/2))
    
    for(k in 1:length(method)){  
      
      print(paste("estimating",method[k]))
      
      if (method[k]=="RLasso" || method[k]=="PostRLasso"){
        x=xL
      } else {
        x=xx
      }
      
      datause <- data[samp1,]
      dataout <- data[-samp1,]
      
      for(j in 1:2){   
        
        if(j==2){
          datause = data[-samp1,]
          dataout = data[samp1,]  
        }
        
        form_y  = y
        form_x  = x
        
        ########################## Boosted  Trees ###################################################;
        
        if(method[k]=="Boosting")   {  fit  <- boost(datause=datause, dataout=dataout, form_x=form_x,  form_y=form_y)}  
        if(method[k]=="Nnet")       {  fit  <- nnetF(datause=datause, dataout=dataout, form_x=form_x,  form_y=form_y)}
        if(method[k]=="RLasso")     {  fit  <- rlassoF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, post=FALSE)}
        if(method[k]=="PostRLasso") {  fit  <- rlassoF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, post=TRUE)}     
        if(method[k]=="Forest")     {  fit  <- RF(datause=datause, dataout=dataout, form_x=form_x,  form_y=form_y, nodesize=5, ntree=1000, tune=FALSE)}
        if(method[k]=="TForest")    {  fit  <- RF(datause=datause, dataout=dataout, form_x=form_x,  form_y=form_y, nodesize=5, ntree=1000, tune=TRUE)} 
        if(method[k]=="Trees")      {  fit  <- tree(datause=datause, dataout=dataout, form_x=form_x,  form_y=form_y)} 
        
        ########################## Regression Trees ###################################################;     
        
        MSE[i,k]                <- ((error(fit$yhatout, dataout[,y])$err)^2)/2 + MSE[i,k] ;
        SE[i,k,j]               <- (summary(lm((dataout[,y]-fit$yhatout)^2~1))$coef[2])^2
        
      }
    }
  }
  
  colnames(result) <- method
  
  rownames(result) <- cbind("MSE", "SE")
  
  result[1,] <- colMeans(MSE)
  result[2,] <- sqrt(.25*SE[1,,1] + .25*SE[1,,2])
  
  return(MSE=result)
}  

# cond_comp computes conditional expectation using machine learning methods

cond_comp <- function(datause, dataout, y, d, x, method, linear){
  
  form_y  = y
  form_d  = d
  form_x  = x
  ind_u   = which(datause[,d]==1)
  ind_o   = which(dataout[,d]==1)
  err.yz1 = NULL
  err.yz0 = NULL
  my_z1x  = NULL
  my_z0x  = NULL
  
  ########################## Boosted  Trees ###################################################;
  
  if(method=="Boosting")
  {
    
    if(linear==0){
      
      fit            <- boost(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit$model, n.trees=fit$best, dataout, type="response") 
      
      fit            <- boost(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit$model, n.trees=fit$best, dataout, type="response") 
      
    }
    
    fit            <- boost(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, distribution='adaboost')
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mis.z          <- error(fit$yhatout, dataout[,d])$mis
    mz_x           <- fit$yhatout       
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- boost(datause=datause, dataout=dataout,  form_x, form_y)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err
    
  }  
  
  
  ########################## Neural Network(Nnet Package) ###################################################;   
  
  
  if(method=="Nnet"){
    
    if(linear==0){
      
      fit            <- nnetF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      dataouts       <- as.data.frame(scale(dataout, center = fit$min, scale = fit$max - fit$min))
      my_z1x         <- predict(fit$model, dataouts)*(fit$max[fit$k]-fit$min[fit$k])+fit$min[fit$k] 
      
      fit            <- nnetF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      dataouts       <- as.data.frame(scale(dataout, center = fit$min, scale = fit$max - fit$min))
      my_z0x         <- predict(fit$model, dataouts)*(fit$max[fit$k]-fit$min[fit$k])+fit$min[fit$k] 
    }  
    
    fit            <- nnetF(datause=datause, dataout=dataout, form_x=form_x, form_y=form_d, clas=TRUE)
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mis.z          <- error(fit$yhatout, dataout[,d])$mis
    mz_x           <- fit$yhatout       
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- nnetF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err    
    
  } 
  
  ########################## Lasso and Post Lasso(Hdm Package) ###################################################;    
  
  if(method=="RLasso" || method=="PostRLasso"){
    
    post = FALSE
    if(method=="PostRLasso"){ post=TRUE }
    
    if(linear==0){
      
      fit            <- rlassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_y, post)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit$model, newdata=formC(form_y, form_x, dataout)$x , type="response") 
      
      fit            <- rlassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_y, post)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit$model, newdata=formC(form_y, form_x, dataout)$x, type="response")   
      
    }
    
    fit            <- rlassoF(datause=datause, dataout=dataout,  form_x, form_d, post, logit=TRUE)
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mis.z          <- error(fit$yhatout, dataout[,d])$mis
    mz_x           <- fit$yhatout    
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- rlassoF(datause=datause, dataout=dataout,  form_x, form_y, post)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err            
  }    
  
  ############# Random Forest ###################################################;
  
  if(method=="Forest" | method=="TForest"){
    
    tune = FALSE
    if(method=="TForest"){tune=TRUE}
    
    
    if(linear==0){
      
      fit            <- RF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, nodesize=5, ntree=1000, tune=tune)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit$model, dataout, type="response") 
      
      fit            <- RF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, nodesize=5, ntree=1000, tune=tune)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit$model, dataout, type="response")
      
    }
    
    fit            <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=paste("as.factor(",form_d,")"), nodesize=1, ntree=1000, reg=FALSE, tune=tune)
    err.z          <- error(as.numeric(fit$yhatout), dataout[,y])$err
    mis.z          <- error(as.numeric(fit$yhatout), dataout[,y])$mis
    mz_x           <- fit$yhatout       
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, nodesize=5, ntree=1000, tune=tune)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err      
  }
  
  ########################## Regression Trees ###################################################;     
  
  if(method=="Trees"){
    
    if(linear==0){
      
      fit            <- tree(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y)
      err.yz1        <- error(fit$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit$model, dataout) 
      
      fit            <- tree(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y)
      err.yz0        <- error(fit$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit$model,dataout)   
      
    }
    
    fit            <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d)
    err.z          <- error(fit$yhatout, dataout[,d])$err
    mis.z          <- error(fit$yhatout, dataout[,d])$mis
    mz_x           <- fit$yhatout       
    
    rz             <- fit$resout 
    err.z          <- error(fit$yhatout, dataout[,d])$err        
    
    fit            <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y)
    ry             <- fit$resout
    err.y          <- error(fit$yhatout, dataout[,y])$err   
  }
  
  return(list(my_z1x=my_z1x, mz_x= mz_x, my_z0x=my_z0x, err.z = err.z,  err.yz0= err.yz0,  err.yz1=err.yz1, mis.z=mis.z, ry=ry , rz=rz, err.y=err.y));
  
}  

#### 401(k) Example #########



# Loading libraries;

library(foreign);
library(quantreg);
library(mnormt);
library(gbm);
library(glmnet);
library(MASS);
library(rpart);
library(sandwich);
library(hdm);
library(randomForest);
library(xtable)
library(nnet)
library(neuralnet)
library(caret)
library(matrixStats)

################ Loading functions and Data ########################


#rm(list = ls())  # Clear everything out so we're starting clean
set.seed(1);
data(Pension)
data<- pension
attach(data)
#data  <- read.dta("sipp1991.dta");


################ Inputs ########################

# Outcome Variable
y.name      <- "net_tfa";

# Treatment Indicator
d.name      <- "e401";    

# Controls
x.name      <- "age + inc + educ + fsize + marr + twoearn + db + pira + hown" # use this for tree-based methods like forests and boosted trees
xl.name     <- "(poly(age, 6) + poly(inc, 8) + poly(educ, 4) + poly(fsize,2) + marr + twoearn + db + pira + hown)^2";  # use this for rlasso etc.

# Method names: Boosting, Nnet, RLasso, PostRLasso, Forest, Trees
method      <- c("Boosting", "Nnet", "RLasso", "PostRLasso", "Forest", "Trees")  

# Estimates ATE using doubly robust score 
DoubleML_DR(data, y.name, d.name, x.name, xl.name, method) 

# Estimates ATE by partialling out partially linear model
DoubleML_PO(data, y.name, d.name, x.name,xl.name, method) 

# Estimates ATE using doubly robust score by picking the best conditional expectation function based on out-of-sample prediction performance
causalBestML(data, y.name, d.name, x.name,xl.name, method) 

# Takes outcome variable y and explanatory variables x and returns average out-of-sample MSE
predictML(data, y.name, x.name, xl.name, method) 


















