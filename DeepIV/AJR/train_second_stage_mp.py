#file:train_second_stage.py
#train the second stage over the node choices 
# with the multiprocessing library to parallelize the tasks
import sys
sys.path.append('/home/luis/CausalML-project/DeepIV/')
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
outputdir='/home/luis/CausalML-project/DeepIV/AJR/'
settlers = pd.read_csv(datadir+'colonial_origins_data_missimp.csv')
base = settlers['baseco']==1
settlers = settlers.loc[base,:]
#import first stage network info
first_mdn = np.load(outputdir + 'opt_mdn_ests.npz')

#create outcomes/endog/instrument + all controls
y = np.array(settlers['logpgp95']) # the outcome
y.shape = [y.shape[0],1]
p = np.array(settlers['avexpr'])/10 #the endogenous variable (normalized to be between 0 and 1)
p.shape = [p.shape[0],1]#make it 2D
z = np.array(settlers.loc[:,'logem4'])[:,np.newaxis] #the instrument(s)
all_covars=np.r_[1:4, 7:8,10:52, 54,58:84]
x = settlers.iloc[:,all_covars] #this should be all of them

folds=5


features_first =  np.concatenate((z, x), axis=1) #combine x + z
#transform the variables exactly as we did for first stage estimation of NN
features_first=np.delete(features_first,first_mdn['excluded_vars'],axis=1)
for v in range(features_first.shape[1]):
    features_first[:,v] = (features_first[:,v] - first_mdn['stdizing_means'][v] )/ first_mdn['stdizing_sds'][v]



#pd.DataFrame(covars).describe()
#could recover directly the dists with these stdized features + 

#second stage features include controls, the missing dummy for the instrument, and the policy var
features_second = np.concatenate((p, x), axis=1)
#remove non-varying vars again, and stdize the means again
#keep p moments for when we draw from the conditional distribution and need to feed it into NN

#don't do this since now NN trained on normalized p-var between 0 and 1
p_mean = 0
p_sd = 1
features_second,feature2_means,feature2_sds,collin_vars2 = deepiv.process_features(features_second)
features_second=np.delete(features_second,collin_vars2,axis=1)
#reinput p so that it is still between 0 and 1
features_second[:,0] = p.flatten()

#try to multithread it?

#deepiv.train_second_stage_cont(y,features_second, \
#    first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'], \
#    10,p_mean,p_sd,seed=1992,p_index=0)
#arglist = []
#for n in noderange:
    #arglist.append( (y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],num_nodes=n,p_mean,p_sd,folds) )
fprename = outputdir + 'cv_mp_output/node'
noderange = range(1,51)
np.random.seed(1992)
if __name__ == '__main__':
    pool=mp.Pool(processes=3)
    results = [pool.apply_async(deepiv.cv_mp_second_stage_cont, \
    args=(y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],n,p_mean,p_sd,None,0,folds,0.001,fprename + str(n) + '.txt')) for n in noderange ]
    output = [p.get() for p in results]
    print output

    cv_ests = []
    for n in noderange:
        fname=fprename + str(n) + '.txt'
        cvout = pd.read_csv(fname)
        cv_ests.append(cvout)

    cv_df = pd.concat(cv_ests)
    cv_df.to_csv(outputdir + 'cv_mp_output/cv_secondstage.csv',index=False)
    opt_nodes = cv_df.sort_values(['mean']).iloc[0]
    opt_nodes = int(opt_nodes['node'])
    opt_params=deepiv.train_second_stage_cont(y,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],opt_nodes,p_mean,p_sd,seed=1992,p_index=0,learning_rate=0.001)
    np.savez(outputdir + 'opt_2ndStage_ests' ,W_in=opt_params[0] ,B_in=opt_params[1], W_out=opt_params[2],B_out=opt_params[3],
        p_mean=p_mean,p_sd=p_sd,stdizing_means=feature2_means,stdizing_sds=feature2_sds,excluded_vars = collin_vars2)
