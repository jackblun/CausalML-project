#file:train_second_stage.py
#using the DNN trained in the first stage as an input, estimate
#the equation relating the covariates to the structural equation for outcome

import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mdn #my created library of mixture density networks
import deepiv #store deepiv functions in here
import multiprocessing as mp
gitdir = '/home/luis/CausalML-project/'

datadir = '/home/luis/CausalML-project/Data/'

settlers = pd.read_csv(datadir+'colonial_origins_data_missimp.csv')
#remove those missing either outcome, endog institution measure
nomiss = (settlers['mi_avexpr']==0) & (settlers['mi_logpgp95']==0)
settlers = settlers.loc[nomiss,:]

#import first stage network info
first_mdn = np.load(gitdir + 'DeepIV/opt_mdn_ests.npz')

#create outcomes/endog/instrument + all controls
y = np.array(settlers['logpgp95']) # the outcome
y.shape = [y.shape[0],1]
p = np.array(settlers['avexpr']) #the endogenous variable
p.shape = [p.shape[0],1]
z = np.array(settlers.loc[:,['logem4','mi_logem4']]) #the instrument(s)
mi_z = np.array(settlers.loc[:,'mi_logem4'])
mi_z.shape = [mi_z.shape[0],1]
all_covars=np.r_[1:3, 7:8,10:52, 54,58:84]
x = settlers.iloc[:,all_covars] 


features_first =  np.concatenate((z, x), axis=1) #combine x + z
#transform the variables exactly as we did for first stage estimation of NN
features_first=np.delete(features_first,first_mdn['excluded_vars'],axis=1)
for v in range(features_first.shape[1]):
    features_first[:,v] = (features_first[:,v] - first_mdn['stdizing_means'][v] )/ first_mdn['stdizing_sds'][v]
#pd.DataFrame(covars).describe()
#could recover directly the dists with these stdized features + 

#second stage features include controls, the missing dummy for the instrument, and the policy var
features_second = np.concatenate((p, x, mi_z), axis=1)
#remove non-varying vars again, and stdize the means again
#keep p moments for when we draw from the conditional distribution and need to feed it into NN
p_mean = np.mean(p)
p_sd = np.std(p)
features_second,feature2_means,feature2_sds,collin_vars2 = deepiv.process_features(features_second)
features_second=np.delete(features_second,collin_vars2,axis=1)


#test the function
#pi,m,s = deepiv.predict_1stStage_cond_dist(features_first, first_mdn['W_in'], \
#        first_mdn['B_in'],first_mdn['W_out'],first_mdn['B_out'])


#params=deepiv.train_second_stage(y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],num_nodes=5,p_mean=p_mean,p_sd=p_sd)
#deepiv.cv_second_stage(y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],num_nodes=5,p_mean=p_mean,p_sd=p_sd)
#y_hat =   np.dot(np.tanh(np.dot(features_second,params[0]) + params[1]),params[2]) + params[3]
#plt.scatter(y,y_hat)
#plt.show()
#plt.hist(y_hat,alpha=.3, color='r',label='predicted y')
#plt.hist(y,color='b',alpha=.3,label='true y')
#plt.legend()
#plt.show()

#try to multithread it?



cv_output = pd.DataFrame(index=noderange,columns=['nodes','err_mean','err_se'])
for n in range(1,features_second.shape[0]):
    print "node count : " + str(n)
    losses = deepiv.cv_second_stage(y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],num_nodes=n,p_mean=p_mean,p_sd=p_sd,folds=5)
    cv_output.loc[n,'nodes'] = n
    cv_output.loc[n,'err_mean'] = np.mean(losses)
    cv_output.loc[n,'err_se'] = np.std(losses)
    print "test MSE: " + str(np.mean(losses))
    print "test SD of MSE: " + str(np.std(losses))
    cv_output.to_csv(gitdir + 'DeepIV/CV_second_stage.csv')

cv_output.to_csv(gitdir + 'DeepIV/CV_first_stage.csv')
