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
prop_train = 4./5.
num_obs = census.shape[0]
training_sample = np.argsort((np.random.rand(num_obs))) <= (num_obs- 1)*prop_train
validation_sample = ~training_sample

opt_nodes=121
#train the NN on the training sample
trainparams = deepiv.train_second_stage_discrete(y[training_sample,:],features_second[training_sample,:], \
    P[training_sample,:], p_range, \
    opt_nodes,learning_rate=0.01,seed=1992)

print "estimating treatments/instruments"
#estimate treatments & instruments of 2nd stage on validation sample
treat,inst = deepiv.predict_2ndStage_discrete(features_first[validation_sample,:], \
    first_mn['W_in'],first_mn['B_in'],first_mn['W_out'],first_mn['B_out'], \
    features_second[validation_sample,:], \
    trainparams[0],trainparams[1],trainparams[2],trainparams[3], \
    p_range)

plt.scatter(y[validation_sample],inst,color='r',label='Instruments',alpha=.1)
plt.scatter(y[validation_sample],treat,color='b',label='Treatments',alpha=.1)
plt.plot([min(y[validation_sample]),max(y[validation_sample])],[min([min(inst),min(treat)]),max([max(inst),max(treat)])] , ls="--", c=".3")
plt.xlabel('Observed Outcome')
plt.ylabel('Predicted Outcome')
plt.legend(loc='lower right')
plt.savefig(outputdir + 'treat_inst_outcomes.pdf')
plt.show()
print "estimating coefficients"
beta,V_beta=deepiv.estimate_iv_coefs(y[validation_sample],treat,inst)

#########
#Begin counterfactuals
#########

#counterfactuals: 
#look at the education-return profile for the average age 
#for non-married,rural new england dude
#compare by race?
mean_age = np.mean(census['AGEQ'])
num_validation_obs = sum(validation_sample)
est_earnings = []

#sample_feature = np.zeros(shape=[1,features_second.shape[1]])
#sample_feature[:,len(p_range)-1 + x_names.index('AGEQ')] = mean_age
#sample_feature[:,len(p_range)-1 + x_names.index('NEWENG')] = 1
#sample_feature[:,len(p_range)-1 + x_names.index('YR25')] = 1

#do it on a batch of 10% validation obs first
counterfact_sample = np.argsort((np.random.rand(num_validation_obs))) <= (num_validation_obs- 1)*0.1
temp_features = features_second[validation_sample,:][counterfact_sample,:]
cf_features_1 = features_first[validation_sample,:][counterfact_sample,:]
y_cf_sample = y[validation_sample,:][counterfact_sample,:]
num_cf_obs = sum(counterfact_sample)
alt_edu_earnings=[]
for c in p_range:
    print c
    temp_p  = np.zeros(shape=[len(p_range),1])
    temp_p[c] = 1
    temp_features[:,p_range] = temp_p.flatten()
    treat,inst = deepiv.predict_2ndStage_discrete(cf_features_1, \
    first_mn['W_in'],first_mn['B_in'],first_mn['W_out'],first_mn['B_out'], \
    temp_features, \
    trainparams[0],trainparams[1],trainparams[2],trainparams[3], \
    p_range)
    H_new =  np.concatenate((np.ones([num_cf_obs,1]), treat), axis=1)
    earn_new = np.zeros([num_cf_obs,1])
    V_earn_new = np.zeros([num_cf_obs,1])
    for i in range(num_cf_obs):
        #if i%1000 == 0:
        #    print i
        earn_new[i] = np.dot(beta.transpose(),H_new[i,:])
        V_earn_new[i] = np.dot(np.dot(H_new[i,:].transpose(),V_beta),H_new[i,:])
    alt_edu_earnings.append([earn_new,V_earn_new])


#plot the first guy
earn_cf =[]
earn_se=[]
p_cf= []
earn_obs = []
for c in p_range:
    p_cf.append(c)
    earn_cf.append(np.mean(alt_edu_earnings[c][0]))
    earn_se.append(np.mean(np.sqrt(alt_edu_earnings[c][1])))
    if sum(features_second[validation_sample,:][counterfact_sample,c])>0:
        earn_obs.append(np.mean(y_cf_sample[features_second[validation_sample,:][counterfact_sample,c]==1]))
    else:
        print "no obs for " + c
        earn_obs.append(np.nan)


plt.errorbar(p_cf,earn_cf,label='Mean earnings from alternative education length',yerr=earn_se)
plt.plot(p_cf,earn_obs,label='Mean observed wages by education')
plt.legend(loc='upper left')
plt.xlabel('Years Of Education')
plt.ylabel("logged weekly wages")
plt.savefig(outputdir + 'obsVstructural_wages.pdf')
plt.show()

#compare structural returns between blacks and whites in the sample
earn_white=[]
earn_black=[]
se_white=[]
se_black=[]
p_cf=[]
blacks = [l[0][temp_features[:,len(p_range) + x_names.index('RACE')] == 1] for l in alt_edu_earnings]
whites = [l[0][temp_features[:,len(p_range) + x_names.index('RACE')] == 0] for l in alt_edu_earnings]
v_blacks = [l[1][temp_features[:,len(p_range) + x_names.index('RACE')] == 1] for l in alt_edu_earnings]
v_whites = [l[1][temp_features[:,len(p_range) + x_names.index('RACE')] == 0] for l in alt_edu_earnings]
for c in p_range:
    p_cf.append(c)
    earn_white.append(np.mean(whites[c]))
    earn_black.append(np.mean(blacks[c]))
    se_white.append(np.mean(np.sqrt(v_whites[c])))
    se_black.append(np.mean(np.sqrt(v_blacks[c])))


plt.errorbar(p_cf,earn_white,label='Whites',yerr=se_white)
plt.errorbar(p_cf,earn_black,label='Blacks',yerr=se_black)
plt.legend(loc='upper left')
plt.xlabel('Years Of Education')
plt.ylabel("Structural Implied Logged Weekly Wages")
plt.savefig(outputdir + 'blacksvswhites.pdf')
plt.show()



#compare structural returns between 40,45,49 year olds
#need to match identically on variables besides age?!
earn_y=[]
earn_m=[]
earn_o=[]
se_y=[]
se_m=[]
se_o=[]
p_cf=[]
young = [l[0][temp_features[:,len(p_range) + x_names.index('YR29')] == 1] for l in alt_edu_earnings]
middle = [l[0][temp_features[:,len(p_range) + x_names.index('YR25')] == 1] for l in alt_edu_earnings]
old = [l[0][temp_features[:,len(p_range) + x_names.index('YR20')] == 1] for l in alt_edu_earnings]

v_young = [l[1][temp_features[:,len(p_range) + x_names.index('YR29')] == 1] for l in alt_edu_earnings]
v_middle = [l[1][temp_features[:,len(p_range) + x_names.index('YR25')] == 1] for l in alt_edu_earnings]
v_old = [l[1][temp_features[:,len(p_range) + x_names.index('YR29')] == 1] for l in alt_edu_earnings]
for c in p_range:
    p_cf.append(c)
    earn_y.append(np.mean(young[c]))
    earn_m.append(np.mean(middle[c]))
    earn_o.append(np.mean(old[c]))
    se_y.append(np.mean(np.sqrt(v_young[c])))
    se_m.append(np.mean(np.sqrt(v_middle[c])))
    se_o.append(np.mean(np.sqrt(v_old[c])))


plt.errorbar(p_cf,earn_y,label='40 year olds in 1980',yerr=se_y)
plt.errorbar(p_cf,earn_m,label='45 year olds in 1980',yerr=se_m)
plt.errorbar(p_cf,earn_o,label='49 year olds in 1980',yerr=se_o)

plt.legend(loc='upper left')
plt.xlabel('Years Of Education')
plt.ylabel("Structural Implied Logged Weekly Wages")
plt.savefig(outputdir + 'cohortdiffs.pdf')
plt.show()