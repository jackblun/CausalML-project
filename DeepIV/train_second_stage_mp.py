#file:train_second_stage.py
#train the second stage over the node choices 
# with the multiprocessing library to parallelize the tasks
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
folds=5

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


#try to multithread it?
noderange = range(1,features_second.shape[1]+1)
fprename = gitdir + 'DeepIV/cv_mp_output/node'
#arglist = []
#for n in noderange:
    #arglist.append( (y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],num_nodes=n,p_mean,p_sd,folds) )

np.random.seed(1992)
if __name__ == '__main__':
    pool=mp.Pool(processes=5)
    results = [pool.apply_async(deepiv.cv_mp_second_stage, \
    args=(y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],n,p_mean,p_sd,None,0,folds,fprename + str(n) + '.txt')) for n in noderange ]
    output = [p.get() for p in results]
    print output

cv_ests = []
for n in noderange:
    fname=fprename + str(n) + '.txt'
    cvout = pd.read_csv(fname)
    cv_ests.append(cvout)

cv_df = pd.concat(cv_ests)
cv_df.to_csv(gitdir + 'DeepIV/cv_mp_output/cv_secondstage.csv',index=False)
#deepiv.cv_mp_second_stage(y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],n,p_mean,p_sd,None,0,folds,fprename + str(n) + '.txt'))
opt_nodes = cv_df.sort_values(['mean']).iloc[0]
opt_nodes = int(opt_nodes['node'])

opt_params=deepiv.train_second_stage(y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],opt_nodes,p_mean,p_sd,seed=1992,p_index=0)
np.savez(gitdir + 'DeepIV/opt_2ndStage_ests' ,W_in=opt_params[0] ,B_in=opt_params[1], W_out=opt_params[2],B_out=opt_params[3],
    p_mean=p_mean,p_sd=p_sd,stdizing_means=feature2_means,stdizing_sds=feature2_sds,excluded_vars = collin_vars2)
