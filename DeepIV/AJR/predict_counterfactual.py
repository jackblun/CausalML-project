#file:predict_counterfactual.py
#use the trained NNs to do some simulation 
#for policy counterfactuals
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
outputdir = '/home/luis/CausalML-project/DeepIV/AJR/'
settlers = pd.read_csv(datadir+'colonial_origins_data_missimp.csv')
base = settlers['baseco']==1
settlers = settlers.loc[base,:]

#import first stage network info
first_mdn = np.load(outputdir + 'opt_mdn_ests.npz')
secondstage_dnn = np.load(outputdir + 'opt_2ndStage_ests.npz')

#create outcomes/endog/instrument + all controls
y = np.array(settlers['logpgp95']) # the outcome
y.shape = [y.shape[0],1]
p = np.array(settlers['avexpr'])/10 #the endogenous variable
p.shape = [p.shape[0],1]



z = np.array(settlers.loc[:,['logem4']]) #the instrument(s)
all_covars=np.r_[1:4, 7:8,10:52, 54,58:84]
x = settlers.iloc[:,all_covars] 

features_first =  np.concatenate((z, x), axis=1) #combine x + z
#transform the variables exactly as we did for first stage estimation of NN
features_first=np.delete(features_first,first_mdn['excluded_vars'],axis=1)
for v in range(features_first.shape[1]):
    features_first[:,v] = (features_first[:,v] - first_mdn['stdizing_means'][v] )/ first_mdn['stdizing_sds'][v]


features_second = np.concatenate((p, x), axis=1)
features_second=np.delete(features_second,secondstage_dnn['excluded_vars'],axis=1)
for v in range(features_second.shape[1]):
    features_second[:,v] = (features_second[:,v] - secondstage_dnn['stdizing_means'][v] )/ secondstage_dnn['stdizing_sds'][v]

features_second[:,0] = p.flatten()
#p_mean = secondstage_dnn['p_mean']
#p_sd = secondstage_dnn['p_sd']
p_mean=0
p_sd=1

z_mean  =z[:,0].mean()
z_sd = z[:,0].std()

opt_nodes= secondstage_dnn['W_in'].shape[1]

#split up data to be training/validation
np.random.seed(1992)
prop_train = 1./2.
num_obs = settlers.shape[0]
training_sample = np.argsort((np.random.rand(num_obs))) <= (num_obs- 1)*prop_train
validation_sample = ~training_sample

#train the NN on the training sample
trainparams = deepiv.train_second_stage_cont(y[training_sample,:],features_second[training_sample,:], \
    first_mdn['pi'][training_sample,:],first_mdn['mu'][training_sample,:],first_mdn['sigma'][training_sample,:], \
    opt_nodes,p_mean,p_sd,seed=1992,p_index=0)

#estimate treatments & instruments of 2nd stage on validation sample
treat,inst = deepiv.predict_2ndStage_cont(features_first[validation_sample,:], \
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
plt.savefig(outputdir + 'treat_inst_outcomes.pdf')
plt.show()


beta,v_beta = deepiv.estimate_iv_coefs(y[validation_sample],treat,inst)



#counterfactuals: 
#assigning full or zero risk appropriation to countries
num_validation_obs = sum(validation_sample)
#apply the DNN over the different counterfactuals for each geography
africa = np.array(settlers['africa'])
asia=np.array(settlers['asia'])
others = np.ones([num_obs]) - africa -asia
all_countries = np.ones([num_obs])
usa = np.array(settlers['shortnam']=="NGA",dtype='int32')
geovars = [africa,asia,others]#,all_countries]#,usa]


p_index=0
p_grid = np.arange(0,1.01,.05)
new_gdp_geo  = np.zeros([len(p_grid),len(geovars)]) #the counterfactual
V_new_gdp_geo  = np.zeros([len(p_grid),5]) #the counterfactual variance
#new_p_geo = np.zeros([len(p_grid),5])
temp_features = features_second

for index in range(len(p_grid)):
    temp_features[:,p_index] =  p_grid[index]
    #ignore the instruments, only care about treatments.
    treat,inst = deepiv.predict_2ndStage_cont(features_first, \
        first_mdn['W_in'],first_mdn['B_in'],first_mdn['W_out'],first_mdn['B_out'], \
        temp_features,trainparams[0],trainparams[1],trainparams[2],trainparams[3], \
        p_mean,p_sd, B=1,p_index=0)
    H_new =  np.concatenate((np.ones([num_obs,1]), treat), axis=1)
    gdp_new = np.zeros([num_obs,1])
    V_gdp_new = np.zeros([num_obs,1])
    for i in range(num_obs):
        gdp_new[i] = np.dot(beta.transpose(),H_new[i,:])
        V_gdp_new[i] = np.dot(np.dot(H_new[i,:].transpose(),v_beta),H_new[i,:])
    #assume they are uncorrelated for now
    #do the deviation from the observed gdp, deviation from observed endogenous variable
    for v in range(len(geovars)):
        g= geovars[v]
        new_gdp_geo[index,v] = np.mean(gdp_new[g==1,0]) #- np.mean(y[validation_sample,:][g==1,0]))/np.mean(y[validation_sample,:][g==1,0])
        V_new_gdp_geo[index,v] = np.mean(V_gdp_new[g==1,0])/float(sum(g==1))
   

colorvals = ['g','r','b']
labels = ['Africa','Asia','Rest of World']
for v in range(len(geovars)):
    g= geovars[v]
    plt.plot(p_grid*10,new_gdp_geo[:,v],color=colorvals[v],label=labels[v] + ' (Structural)')
    plt.plot(p_grid*10,new_gdp_geo[:,v]-np.sqrt(V_new_gdp_geo[:,v])*1.96, \
        linestyle='dashed',color=colorvals[v])
    plt.plot(p_grid*10,new_gdp_geo[:,v]+np.sqrt(V_new_gdp_geo[:,v])*1.96, \
        linestyle='dashed',color=colorvals[v])
    plt.scatter(p[g==1]*10,y[g==1], \
        color=colorvals[v],alpha=.5,label=labels[v] + ' (Observed)')


plt.legend(loc='upper left')
plt.xlim([0, 10])
plt.ylim([6, 11])
plt.xlabel('Protection Against Expropriation Index (0-10 Scale)')
plt.ylabel('Log GDP per Capita,1995')
plt.savefig(outputdir + 'cf_riskexprop.pdf')
plt.show()