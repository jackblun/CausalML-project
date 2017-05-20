#train a series of NNs to figure out which one is best
#loop over: learning rate, num nodes, and num components
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mdn #my created library

gitdir = '/home/luis/CausalML-project/'



datadir = '/home/luis/CausalML-project/Data/'

settlers = pd.read_csv(datadir+'colonial_origins_data_missimp.csv')
#remove those missing either outcome, endog institution measure, or exog instrument
nomiss = (settlers['mi_avexpr']==0) & (settlers['mi_logpgp95']==0)
settlers = settlers.loc[nomiss,:]

p = np.array(settlers['avexpr']) #the endogenous variable
p.shape = [p.shape[0],1]#make it 2D
z = np.array(settlers.loc[:,['logem4','mi_logem4']]) #the instrument(s)
all_covars=np.r_[1:3, 7:8,10:52, 54,58:84]
x = settlers.iloc[:,all_covars] #this should be all of them

covars =  np.concatenate((z, x), axis=1) #combine x + z
collin_vars = [] #indices of variables with no variation
covar_means = [] #store means we normalize
covar_sds = [] #store SDs we normalize
#stdize all non-dummy variables to have mean 0 and SD 1
for v in range(covars.shape[1]):
    #remove variables with one unique value- they mess stuff up later
    if len(np.unique(covars[:,v].astype(np.float32)))==1:
        collin_vars.append(v)
        continue
    #skip normalizations for dummies (although I guess it doesn't really matter)
    is_dummy = (np.unique(covars[:,v].astype(np.float32))==np.array([0.,1.]))   
    if isinstance(is_dummy,bool):
        if is_dummy:
            covar_means.append(0) #for dummies do not transform
            covar_sds.append(1)
            continue
    else:
        if is_dummy.all():
            covar_means.append(0) #for dummies do not transform
            covar_sds.append(1)
            continue  
    covar_means.append(np.mean(covars[:,v])) #for dummies do not transform
    covar_sds.append(np.std(covars[:,v]))    
    covars[:,v] = (covars[:,v] - np.mean(covars[:,v]))/np.std(covars[:,v])

covars=np.delete(covars,collin_vars,axis=1)

#store the rules for feature transformation as arrays in numpy
covar_means = np.array(covar_means)
covar_sds = np.array(covar_sds)
collin_vars = np.array(collin_vars)
rates = [.001]
comps=range(1,6)
node_grid = 20 #number of nodes within each loop to search
max_index=len(rates)*len(comps)*node_grid
np.random.seed(1992)
#create mat to store the output of the CV
cv_output = pd.DataFrame(index=range(max_index),columns=['rate','comps','nodes','LL'])
index=0
for r in rates:
    for c in comps:
        min_nodes=c*3
        for h in range(min_nodes,covars.shape[1]+1):
            #for h in np.linspace(min_nodes,covars.shape[1]+1,node_grid,endpoint=True,dtype=np.int32):
            #for h in np.linspace(min_nodes,20,node_grid,endpoint=True,dtype=np.int32):
            print "---------------"
            print "rate: " + str(r) + '; #comps: ' + str(c) + "; nodes: " + str(h)
            cv_output.loc[index,'rate']=r
            cv_output.loc[index,'comps']=c
            cv_output.loc[index,'nodes']=h
            mean_LL = mdn.cv_MDN(p,covars,num_components=c,num_nodes=h,learning_rate=r,folds=5,seed=None)
            cv_output.loc[index,'LL']=mean_LL
            print "Test CV Loss: " + str(mean_LL)
            cv_output.to_csv(gitdir + 'DeepIV/CV_first_stage.csv')
            index+=1

#save output
cv_output.to_csv(gitdir + 'DeepIV/CV_first_stage.csv')

#fit NN that has best fit, and recover the parameters 
best_params = cv_output.sort_values(by=['LL'],ascending=True).iloc[0,:]
print "Optimal fit: "
print best_params
[[W_in_final, B_in_final, W_out_final,B_out_final],[mixprobs,mixmeans,mixsds]] = \
    mdn.fit_MDN(p,covars,num_components=best_params['comps'],num_nodes=best_params['nodes'],learning_rate=best_params['rate'])

np.savez(gitdir + 'DeepIV/opt_mdn_ests' ,W_in=W_in_final ,B_in=B_in_final, W_out=W_out_final,B_out=B_out_final,
    pi=mixprobs,mu=mixmeans,sigma=mixsds, \
    stdizing_means=covar_means,stdizing_sds=covar_sds,excluded_vars = collin_vars)

#finally, simulate the data (only for obs with nonmissing instruments)
nomiss_z = np.array(settlers['mi_logem4']==0)
mdn.sim_mdn(p[nomiss_z],z[nomiss_z,0],covars[nomiss_z,:], \
    mixprobs[nomiss_z,:],mixmeans[nomiss_z,:],mixsds[nomiss_z,:],figdir = '/home/luis/CausalML-project/DeepIV/')
