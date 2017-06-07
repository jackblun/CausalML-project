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
np.random.seed(1992)

prop_train = 1./2.
num_obs = census.shape[0]
training_sample = np.argsort((np.random.rand(num_obs))) <= (num_obs- 1)*prop_train
validation_sample = ~training_sample

opt_nodes=43
#train the NN on the training sample
trainparams = deepiv.train_second_stage_discrete(y[training_sample,:],features_second[training_sample,:], \
    P[training_sample,:], p_range, \
    opt_nodes,learning_rate=0.01,seed=1992)

print "estimating treatments/instruments"
#estimate treatments & instruments & coefs of 2nd stage on validation sample
treat,inst = deepiv.predict_etas_discrete(P[validation_sample,:], \
    features_second[validation_sample,:], \
    trainparams[0],trainparams[1],p_range)


beta,V_beta=deepiv.estimate_iv_coefs(y[validation_sample],treat,inst)





# #########
# #Begin counterfactuals
# #########


#calculate predicted structural wages on the census
counterfact_sample = np.argsort((np.random.rand(num_obs))) <= (num_obs- 1)*1
cf_features = features_second[counterfact_sample,:]
y_cf_sample = y[counterfact_sample,:]
num_cf_obs = sum(counterfact_sample)
alt_edu_earnings=[]
for c in p_range:
    print c
    temp_p  = np.zeros(shape=[num_cf_obs,len(p_range)])
    temp_p[:,c] = 1
    cf_features[:,p_range] = temp_p
    treat_temp =  np.tanh(np.dot(cf_features,trainparams[0]) + trainparams[1])
    H_new =  np.concatenate((np.ones([num_cf_obs,1]), treat_temp), axis=1)
    earn_new = np.dot(H_new,beta)
    alt_edu_earnings.append(earn_new)


#plot the first guy
earn_cf =[]
earn_se=[]
p_cf= []
earn_obs = []
for c in p_range:
    p_cf.append(c)
    earn_cf.append(np.mean(alt_edu_earnings[c]))
    #earn_se.append(np.mean(np.sqrt(alt_edu_earnings[c][1])))
    if sum(features_second[counterfact_sample,c])>0:
        earn_obs.append(np.mean(y_cf_sample[features_second[counterfact_sample,c]==1]))
    else:
        print "no obs for " + c
        earn_obs.append(np.nan)

plt.plot(p_cf,earn_obs,label='Mean observed wages by education')
plt.plot(p_cf,earn_cf,label='Mean earnings from alternative education length')
plt.xlabel('Years Of Education')
plt.ylabel("logged weekly wages")
plt.savefig(outputdir + 'obsVstructural_wages.pdf')
plt.legend()
plt.show()

# #compare structural returns between blacks and whites in the sample
# earn_white=[]
# earn_black=[]
# se_white=[]
# se_black=[]
# p_cf=[]
# blacks = [l[0][temp_features[:,len(p_range) + x_names.index('RACE')] == 1] for l in alt_edu_earnings]
# whites = [l[0][temp_features[:,len(p_range) + x_names.index('RACE')] == 0] for l in alt_edu_earnings]
# v_blacks = [l[1][temp_features[:,len(p_range) + x_names.index('RACE')] == 1] for l in alt_edu_earnings]
# v_whites = [l[1][temp_features[:,len(p_range) + x_names.index('RACE')] == 0] for l in alt_edu_earnings]
# for c in p_range:
#     p_cf.append(c)
#     earn_white.append(np.mean(whites[c]))
#     earn_black.append(np.mean(blacks[c]))
#     se_white.append(np.mean(np.sqrt(v_whites[c])))
#     se_black.append(np.mean(np.sqrt(v_blacks[c])))


# plt.errorbar(p_cf,earn_white,label='Whites',yerr=se_white)
# plt.errorbar(p_cf,earn_black,label='Blacks',yerr=se_black)
# plt.legend(loc='upper left')
# plt.xlabel('Years Of Education')
# plt.ylabel("Structural Implied Logged Weekly Wages")
# plt.savefig(outputdir + 'blacksvswhites.pdf')
# plt.show()



# #compare structural returns between 40,45,49 year olds
# #need to match identically on variables besides age?!
# earn_y=[]
# earn_m=[]
# earn_o=[]
# se_y=[]
# se_m=[]
# se_o=[]
# p_cf=[]
# young = [l[0][temp_features[:,len(p_range) + x_names.index('YR29')] == 1] for l in alt_edu_earnings]
# middle = [l[0][temp_features[:,len(p_range) + x_names.index('YR25')] == 1] for l in alt_edu_earnings]
# old = [l[0][temp_features[:,len(p_range) + x_names.index('YR20')] == 1] for l in alt_edu_earnings]

# v_young = [l[1][temp_features[:,len(p_range) + x_names.index('YR29')] == 1] for l in alt_edu_earnings]
# v_middle = [l[1][temp_features[:,len(p_range) + x_names.index('YR25')] == 1] for l in alt_edu_earnings]
# v_old = [l[1][temp_features[:,len(p_range) + x_names.index('YR29')] == 1] for l in alt_edu_earnings]
# for c in p_range:
#     p_cf.append(c)
#     earn_y.append(np.mean(young[c]))
#     earn_m.append(np.mean(middle[c]))
#     earn_o.append(np.mean(old[c]))
#     se_y.append(np.mean(np.sqrt(v_young[c])))
#     se_m.append(np.mean(np.sqrt(v_middle[c])))
#     se_o.append(np.mean(np.sqrt(v_old[c])))


# plt.errorbar(p_cf,earn_y,label='40 year olds in 1980',yerr=se_y)
# plt.errorbar(p_cf,earn_m,label='45 year olds in 1980',yerr=se_m)
# plt.errorbar(p_cf,earn_o,label='49 year olds in 1980',yerr=se_o)

# plt.legend(loc='upper left')
# plt.xlabel('Years Of Education')
# plt.ylabel("Structural Implied Logged Weekly Wages")
# plt.savefig(outputdir + 'cohortdiffs.pdf')
# plt.show()


#compare black and white men returns by age for a given cohort (most populated of the 10)
#get medians/averages for CF
y_median = 'YR20'
for yr in range(20,30):
    yvar = 'YR'  +str(yr)
    if sum(census[yvar]) > sum(census[y_median]):
        y_median=yvar

area_median = 'ENOCENT'
areas=[u'ENOCENT',u'ESOCENT',u'MIDATL', u'MT', u'NEWENG', u'SOATL',u'WNOCENT', u'WSOCENT']
for g in areas:
    if sum(census[g]) > sum(census[area_median]):
        area_median=g

#input median features w/ HS education (12 years)
cf_features = np.zeros(shape=[1,features_second.shape[1]])
cf_features[:,12] = 1
cf_features[:,len(p_range) + x_names.index('RACE')] = np.median(census['RACE'])
cf_features[:,len(p_range) + x_names.index('SMSA')] = np.median(census['SMSA'])
#cf_features[:,len(p_range) + x_names.index(y_median)] = 1
cf_features[:,len(p_range) + x_names.index(area_median)] = 1
#dummy first features to pass to the deepiv function (doesn't affect output)
age_grid = np.arange(census['AGEQ'].min(),census['AGEQ'].max()+.25,.25)
hs_returns_age = np.zeros(shape=[len(age_grid),2])
V_hs_returns_age = np.zeros(shape=[len(age_grid),2])
for dummy in [0,1]:
    cf_features[:,len(p_range) + x_names.index('MARRIED')] = dummy
    for a in range(len(age_grid)):
        age_new = age_grid[a]
        yr_new = int(1980 - np.floor(age_grid[0]+.75) - 1910)
        cf_features[:,len(p_range) + x_names.index('AGEQ')] = age_new
        cf_features[:,len(p_range) + x_names.index('YR' + str(yr_new))]=1
        treat_temp =  np.tanh(np.dot(cf_features,trainparams[0]) + trainparams[1])
        H_new =  np.concatenate((np.ones([1,1]),treat_temp),axis=1)
        hs_returns_age[a,dummy] = np.dot(H_new,beta)
        V_hs_returns_age[a,dummy] = np.dot(np.dot(H_new,V_beta),H_new.transpose())
        cf_features[:,len(p_range) + x_names.index('YR' + str(yr_new))]=0


#plot it
colorvals = ['g','b']
for dummy in [0,1]:
    plt.plot(age_grid, hs_returns_age[:,dummy],label="MARRIED=" + str(dummy),color=colorvals[dummy])
    plt.plot(age_grid, hs_returns_age[:,dummy] - np.sqrt(V_hs_returns_age[:,dummy]), \
            linestyle='dashed',color=colorvals[dummy])
    plt.plot(age_grid, hs_returns_age[:,dummy] + np.sqrt(V_hs_returns_age[:,dummy]), \
            linestyle='dashed',color=colorvals[dummy])    

plt.xlabel('Age in 1980 Census')
plt.ylabel('Predicted Log Weekly Wages')
plt.legend(loc='lower left')
plt.savefig(outputdir + 'cf_married.pdf')
plt.show()
##########################################
#try regional returns to HS education by race
#fix age to plurality observed
cf_features[:,len(p_range) + x_names.index(y_median)] = 1
cf_features[:,len(p_range) + x_names.index('AGEQ')] = np.median(census.loc[census[y_median]==1, 'AGEQ'])
cf_features[:,len(p_range) + x_names.index('MARRIED')] = np.median(census['MARRIED'])
cf_features[:,len(p_range) + x_names.index(area_median)] = 0
#dummy first features to pass to the deepiv function (doesn't affect output)

hs_returns_region= np.zeros(shape=[len(areas),2])
V_hs_returns_region = np.zeros(shape=[len(areas),2])
for dummy in [0,1]:
    cf_features[:,len(p_range) + x_names.index('RACE')] = dummy
    for a in range(len(areas)):
        area_new = areas[a]
        cf_features[:,len(p_range) + x_names.index(area_new)] = 1
        treat_temp =  np.tanh(np.dot(cf_features,trainparams[0]) + trainparams[1])
        H_new =  np.concatenate((np.ones([1,1]),treat_temp),axis=1)
        hs_returns_region[a,dummy] = np.dot(H_new,beta)
        V_hs_returns_region[a,dummy] = np.dot(np.dot(H_new,V_beta),H_new.transpose())
        cf_features[:,len(p_range) + x_names.index(area_new)] = 0



colorvals = ['g','b']
for dummy in [0,1]:
    #plt.scatter(range(len(areas)), hs_returns_region[:,dummy],label="BLACK=" + str(dummy),color=colorvals[dummy])
    plt.errorbar(np.array(range(len(areas))) +(-.05 +.1*dummy) ,hs_returns_region[:,dummy],yerr= np.sqrt(V_hs_returns_region[:,dummy]), \
        label="BLACK=" + str(dummy),color=colorvals[dummy], ls='none',marker='o')
    #plt.plot(range(len(areas)), hs_returns_region[:,dummy] - np.sqrt(V_hs_returns_region[:,dummy]), \
    #        linestyle='dashed',color=colorvals[dummy])
    #plt.plot(range(len(areas)), hs_returns_region[:,dummy] + np.sqrt(V_hs_returns_region[:,dummy]), \
    #        linestyle='dashed',color=colorvals[dummy])    


plt.legend(loc='lower left')
plt.xticks(range(len(areas)),areas,rotation=25)
plt.ylabel('Estimated Log Weekly Wages')
plt.savefig(outputdir + 'cf_region_race.pdf')
plt.show()

#last one: plot median person against observed wages by education, (rather than average outcome as above)
med_returns_edu = []
V_med_returns_edu=[]
cf_features[:,12] = 0
cf_features[:,len(p_range) + x_names.index('RACE')] = np.median(census['RACE'])
cf_features[:,len(p_range) + x_names.index(area_median)] = 1
for c in p_range:
    cf_features[:,c] = 1
    treat_temp =  np.tanh(np.dot(cf_features,trainparams[0]) + trainparams[1])
    H_new =  np.concatenate((np.ones([1,1]),treat_temp),axis=1)
    med_returns_edu.append(np.dot(H_new,beta)[0,0])
    V_med_returns_edu.append(np.dot(np.dot(H_new,V_beta),H_new.transpose())[0,0])
    cf_features[:,c] = 0

plt.plot(p_cf,earn_obs,label='Mean observed wages by education')
plt.errorbar(p_cf,med_returns_edu,label='Mean earnings from alternative education length for median observation')
plt.xlabel('Years Of Education')
plt.ylabel("logged weekly wages")
plt.savefig(outputdir + 'obsVstructural_wages_med.pdf')
plt.legend()
plt.show()
