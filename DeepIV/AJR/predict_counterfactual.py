#file:predict_counterfactual.py
#use the trained NNs to do some simulation 
#for policy counterfactuals

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
secondstage_dnn = np.load(gitdir + 'DeepIV/opt_2ndStage_ests.npz')

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

features_second = np.concatenate((p, x, mi_z), axis=1)
features_second=np.delete(features_second,secondstage_dnn['excluded_vars'],axis=1)
for v in range(features_second.shape[1]):
    features_second[:,v] = (features_second[:,v] - secondstage_dnn['stdizing_means'][v] )/ secondstage_dnn['stdizing_sds'][v]

p_mean = secondstage_dnn['p_mean']
p_sd = secondstage_dnn['p_sd']

z_mean  =z[:,0].mean()
z_sd = z[:,0].std()

opt_nodes= 4

#split up data to be training/validation
prop_train = 1./2.
num_obs = settlers.shape[0]
training_sample = np.argsort((np.random.rand(num_obs))) <= (num_obs- 1)*prop_train
validation_sample = ~training_sample

np.random.seed(1992)
#train the NN on the training sample
trainparams = deepiv.train_second_stage(y[training_sample,:],features_second[training_sample,:], \
    first_mdn['pi'][training_sample,:],first_mdn['mu'][training_sample,:],first_mdn['sigma'][training_sample,:], \
    opt_nodes,p_mean,p_sd,seed=1992,p_index=0)

#estimate treatments & instruments of 2nd stage on validation sample
treat,inst = deepiv.predict_2ndStage_cond_dist(features_first[validation_sample,:], \
    first_mdn['W_in'],first_mdn['B_in'],first_mdn['W_out'],first_mdn['B_out'], \
    features_second[validation_sample,:],trainparams[0],trainparams[1],trainparams[2],trainparams[3], \
    p_mean,p_sd, B=10000,p_index=0)

#plot treat vs instrument
plt.scatter(y[validation_sample],inst,color='r',label='Instruments',alpha=.5)
plt.scatter(y[validation_sample],treat,color='b',label='Treatments',alpha=.5)
plt.plot([min(y[validation_sample]),max(y[validation_sample])],[min([min(inst),min(treat)]),max([max(inst),max(treat)])] , ls="--", c=".3")
plt.xlabel('Observed Outcome')
plt.ylabel('Predicted Outcome')
plt.legend(loc='lower right')
plt.savefig(gitdir + 'DeepIV/treat_inst_outcomes.pdf')
plt.show()


beta,v_beta = deepiv.estimate_iv_coefs(y[validation_sample],treat,inst)

print beta
print v_beta

#counterfactuals: 
#assigning full or zero risk appropriation to countries
num_validation_obs = sum(validation_sample)
#apply the DNN over the different counterfactuals for each geography
africa = np.array(settlers['africa'])[validation_sample]
asia=np.array(settlers['asia'])[validation_sample]
others = np.ones([num_validation_obs]) - africa -asia
all_countries = np.ones([num_validation_obs])
usa = np.array(settlers['shortnam']=="VEN",dtype='int32')[validation_sample]
geovars = [africa,asia,others,all_countries,usa]


z_grid = np.arange(0,8,.1)
new_gdp_geo  = np.zeros([len(z_grid),5]) #the counterfactual
V_new_gdp_geo  = np.zeros([len(z_grid),5]) #the counterfactual variance
new_p_geo = np.zeros([len(z_grid),5])
temp_features = features_first[validation_sample,:]
z_index=0
for index in range(len(z_grid)):
    z_new = (z_grid[index] - z_mean)/z_sd
    temp_features[:,z_index] = z_new
    #ignore the instruments, only care about treatments.
    treat,inst = deepiv.predict_2ndStage_cond_dist(temp_features, \
        first_mdn['W_in'],first_mdn['B_in'],first_mdn['W_out'],first_mdn['B_out'], \
        features_second[validation_sample,:],trainparams[0],trainparams[1],trainparams[2],trainparams[3], \
        p_mean,p_sd, B=1000,p_index=0)
    H_new =  np.concatenate((np.ones([num_validation_obs,1]), inst), axis=1)
    gdp_new = np.zeros([num_validation_obs,1])
    V_gdp_new = np.zeros([num_validation_obs,1])
    for i in range(num_validation_obs):
        gdp_new[i] = np.dot(beta.transpose(),H_new[i,:])
        V_gdp_new[i] = np.dot(np.dot(H_new[i,:].transpose(),v_beta),H_new[i,:])
    #assume they are uncorrelated for now
    #do the deviation from the observed gdp, deviation from observed endogenous variable
    for v in range(len(geovars)):
        g= geovars[v]
        new_gdp_geo[index,v] = np.mean(gdp_new[g==1,0]) - np
        V_new_gdp_geo[index,v] = np.mean(V_gdp_new[g==1,0])/float(sum(g==1))
        new_p_geo[index,v] = np.mean(inst[g==1,0])/float(sum(g==1))


plt.plot(new_p_geo[:,0],new_gdp_geo[:,0],color='g',label='Africa')
plt.plot(new_p_geo[:,1],new_gdp_geo[:,1],color='r',label='Asia')
plt.plot(new_p_geo[:,2],new_gdp_geo[:,2],color='b',label='Rest of World')


plt.plot(new_p_geo[:,3],new_gdp_geo[:,3],color='b',label='Rest of World')
plt.plot(z_grid,new_gdp_geo[:,3],color='b',label='Rest of World')


plt.plot(z_grid,new_gdp_geo[:,0],color='g',label='Africa')
plt.plot(z_grid,new_gdp_geo[:,1],color='r',label='Asia')
plt.plot(z_grid,new_gdp_geo[:,2],color='b',label='Rest of World')
colorvals = ['g','r','b']
for v in range(len(geovars)):
    g= geovars[v]
    plt.scatter(p[validation_sample][g==1],y[validation_sample][g==1],color=colorvals[v],alpha=.3)


plt.legend(loc='bottom left')
plt.show()