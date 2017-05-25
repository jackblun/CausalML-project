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

#split up data to be training/validation for counterfactual estimation
prop_train = 1./2.
num_obs = settlers.shape[0]
training_sample = np.argsort((np.random.rand(num_obs))) <= (num_obs- 1)*prop_train
validation_sample = ~training_sample

opt_nodes=121
np.random.seed(1992)
#train the NN on the training sample
trainparams = deepiv.train_second_stage_discrete(y[training_sample,:],features_second[training_sample,:], \
    P[training_sample,:], p_range, \
    opt_nodes,seed=1992)

#estimate treatments & instruments of 2nd stage on validation sample
treat,inst = deepiv.predict_2ndStage_cond_dist(features_first[validation_sample,:], \
    first_mn['W_in'],first_mn['B_in'],first_mn['W_out'],first_mn['B_out'], \
    features_second[validation_sample,:],trainparams[0],trainparams[1],trainparams[2],trainparams[3], \
    p_mean,p_sd, B=10000,p_index=0)
