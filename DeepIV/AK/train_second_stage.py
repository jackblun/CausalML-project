#train a series of NNs to figure out which one is best
#loop over: learning rate, num nodes, and num components
import sys
sys.path.append('/home/luis/CausalML-project/DeepIV/')
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mdn #my created library
import deepiv
import multiprocessing as mp
gitdir = '/home/luis/CausalML-project/'



datadir = '/home/luis/CausalML-project/Data/angrist_krueger_1991/'
outputdir = '/home/luis/CausalML-project/DeepIV/AK/'
#look at age 40-49 men  (prime working age) in 1980 census

census = pd.read_csv(datadir+'AK1991.csv')

z_names = [v for v in census.columns.values if v.find('QTR')!=-1 and len(v)!=4]

#exclude age^2 since the NN can handle nonlinearities anyway
x_names = [u'AGEQ',u'MARRIED',u'RACE',  u'SMSA',\
       u'ENOCENT',u'ESOCENT',u'MIDATL', u'MT', u'NEWENG', u'SOATL',u'WNOCENT', u'WSOCENT', \
       u'YR20',u'YR21', u'YR22', u'YR23', u'YR24',\
       u'YR25', u'YR26', u'YR27', u'YR28',u'YR29']

z = np.array(census[z_names])
x = np.array(census[x_names])
p = np.array(census['EDUC'])
p.shape = [p.shape[0],1]
p_ind=pd.Series(p[:,0]).astype('category')
p_ind = np.array(pd.get_dummies(p_ind))
y=np.array(census['LWKLYWGE'])
y.shape = [y.shape[0],1]
#stdize age to be z-score (only continuous input)
age_mean= np.mean(census['AGEQ'])
age_sd = np.std(census['AGEQ'])
x[:,0] = (x[:,0] - age_mean)/age_sd
first_mn = np.load(outputdir + 'opt_mn_ests.npz')
P = first_mn['P']

#create features for each stage
features_first =  np.concatenate((z, x), axis=1) #combine x + z
features_second = np.concatenate((p_ind,x),axis=1)
p_range = range(p_ind.shape[1])
#num_nodes=20
#ss=deepiv.train_second_stage_discrete(y,features_second,P,p_range,num_nodes,seed=None)
#loop over deepIV
noderange = range(1,features_second.shape[1]+1,2)
noderange = range(101,200,2)
folds=5
fprename = outputdir + 'cv_mp_output/node'
np.random.seed(1992)
if __name__ == '__main__':
    pool=mp.Pool(processes=5)
    results = [pool.apply_async(deepiv.cv_mp_second_stage_discrete, \
    args=(y,features_second,P,p_range,h,None,folds,fprename + str(h) + '.txt')) for h in noderange ]
    output = [p.get() for p in results]
    print output

noderange = range(1,200,2)
cv_ests = []
for n in noderange:
    fname=fprename + str(n) + '.txt'
    cvout = pd.read_csv(fname)
    cv_ests.append(cvout)


cv_df = pd.concat(cv_ests)
cv_df.to_csv(outputdir + 'cv_mp_output/cv_secondstage.csv',index=False)

#train optimal one- we use 99 since this one looks to be very low but also relatively stable
#121 als a good choice
opt_nodes = 121
opt_params=deepiv.train_second_stage_discrete(y,features_second,P,p_range,opt_nodes,seed=1992)
np.savez(outputdir + 'opt_2ndStage_ests' ,W_in=opt_params[0] ,B_in=opt_params[1], W_out=opt_params[2],B_out=opt_params[3])
