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
#stdize age to be z-score
age_mean= np.mean(census['AGEQ'])
age_sd = np.std(census['AGEQ'])

x[:,0] = (x[:,0] - age_mean)/age_sd


covars =  np.concatenate((z, x), axis=1) #combine x + z

#mdn_params,dist_params=mdn.fit_MDN(p,covars,num_components=5,num_nodes=20,learning_rate=0.001,seed=1992)



#mdn.fit_MN(p,covars,num_nodes=10,learning_rate=0.001,seed=None,plot_loss=True)

#############
#FIT MN VIA CV
#mdn.fit_MN(p,covars,num_nodes=h,learning_rate=r,seed=None)
np.random.seed(1992)
min_nodes=21
index=0
r=0.01
node_grid = range(min_nodes,150,4)
cv_output = pd.DataFrame(index=node_grid,columns=['rate','nodes','LL_mean','LL_sd'])
for h in node_grid:
    print "---------------"
    print "rate: " + str(r)  + "; nodes: " + str(h)
    cv_output.loc[h,'rate']=r
    cv_output.loc[h,'nodes']=h
    LL = mdn.cv_MN(p,covars,num_nodes=h,learning_rate=r,folds=5,seed=None)
    cv_output.loc[h,'LL_mean']=np.mean(LL)
    cv_output.loc[h,'LL_sd'] = np.std(LL)
    print "Test CV Loss: " + str(np.mean(LL))
    cv_output.to_csv(outputdir + 'CV_first_stage_MN.csv')





# #store the rules for feature transformation as arrays in numpy
# rates = [.01]
# #comps=np.linspace(1,10,1,endpoint=True,dtype=np.int32)
# comps = range(11,21)
# node_grid = 10 #number of nodes within each loop to search
# max_index=len(rates)*len(comps)*node_grid
# #create mat to store the output of the CV
# cv_output = pd.DataFrame(index=range(max_index),columns=['rate','comps','nodes','LL_mean','LL_sd'])
# index=0
# for r in rates:
#     for c in comps:
#         min_nodes=c*3
#         #for h in range(min_nodes,covars.shape[1]+1):
#         for h in np.linspace(min_nodes,covars.shape[1]+1,node_grid,endpoint=True,dtype=np.int32):
#             #for h in np.linspace(min_nodes,20,node_grid,endpoint=True,dtype=np.int32):
#             print "---------------"
#             print "rate: " + str(r) + '; #comps: ' + str(c) + "; nodes: " + str(h)
#             cv_output.loc[index,'rate']=r
#             cv_output.loc[index,'comps']=c
#             cv_output.loc[index,'nodes']=h
#             LL = mdn.cv_MDN(p,covars,num_components=c,num_nodes=h,learning_rate=r,folds=5,seed=None)
#             cv_output.loc[index,'LL_mean']=np.mean(LL)
#             cv_output.loc[index,'LL_sd'] = np.std(LL)
#             print "Test CV Loss: " + str(np.mean(LL))
#             cv_output.to_csv(outputdir + 'CV_first_stage2.csv')
#             index+=1

# #save output
# cv_output.to_csv(outputdir+'CV_first_stage2.csv')


cv_output = pd.read_csv(outputdir+'CV_first_stage_MN.csv')

#fit NN that has best fit, and recover the parameters 
best_params = cv_output.sort_values(by=['LL_mean'],ascending=True).iloc[0,:]
print "Optimal fit: "
print best_params
[[W_in_final, B_in_final, W_out_final,B_out_final],P_final] = \
    mdn.fit_MN(p,covars,num_nodes=int(best_params['nodes']),learning_rate=best_params['rate'],num_batches=100)

np.savez(outputdir + 'opt_mn_ests' ,W_in=W_in_final ,B_in=B_in_final, W_out=W_out_final,B_out=B_out_final,
    P=P_final)



#np.savez(outputdir + 'opt_mdn_ests' ,W_in=W_in_final ,B_in=B_in_final, W_out=W_out_final,B_out=B_out_final,
#    pi=mixprobs,mu=mixmeans,sigma=mixsds)

#finally, simulate the data 
#need to reorient instruments for plotting since currently dummies
z_plot = np.zeros(shape=[p.shape[0],1])
for y in range(20,30):
    for q in range(1,4):
        v_name = 'QTR' + str(q) + str(y)
        print v_name + ": " + str(sum(np.array(census[v_name]==1)))
        z_plot[np.array(census[v_name]==1),:] = 1910. + y + (q-1.)/4
    v_name = "YR" + str(y)
    q4_logic=np.array( (z_plot[:,0]==0) & (np.array(census[v_name]==1)))
    print v_name + ": " +str(sum(q4_logic))
    z_plot[q4_logic,:] = 1910. + y + .75

#p_sim = mdn.sim_mdn(mixprobs,mixmeans,mixsds,B=10)

p_sim = mdn.sim_mn(np.unique(p),P_final,B=100)
p_true = p

bins=np.arange(0,20,1)
cdf_sim,edges,other =plt.hist(p_sim.flatten(),color='r',normed=True,alpha=.5,bins=bins,label='Simulated Data')        
cdf_true,edges,other=plt.hist(p_true.flatten(),color='b',normed=True,alpha=.5,bins=bins,label='Actual Data')
plt.xlabel('Endogenous Variable')
plt.ylabel('Density')
plt.legend(loc='upper left')
plt.savefig(outputdir + 'hist_endog_MN.pdf')
plt.show()

plt.plot(bins[1:20],np.cumsum(cdf_sim),color='r',alpha=.5,label='Simulated Data')        
plt.plot(bins[1:20],np.cumsum(cdf_true),color='b',alpha=.5,label='Actual Data')
plt.xlabel('Endogenous Variable')
plt.ylabel('CDF')
plt.legend(loc='upper left')
plt.savefig(outputdir + 'cdf_endog_MN.pdf')
plt.show()




sim_means=[]
true_means=[]
marker_list =['o','o','o','o']
for v in sorted(np.unique(z_plot)):
    sim_means.append(np.mean(p_sim[z_plot[:,0]==v,:]))
    true_means.append(np.mean(p_true[z_plot[:,0]==v,:]))  

ec = plt.cm.coolwarm(z_plot[:,0]*4 %4)
markers=marker_list*10
colorlist=['black','white','white','white']*10

plt.plot(sorted(np.unique(z_plot)),sim_means,color='r',label='Simulated Data')
plt.plot(sorted(np.unique(z_plot)),true_means,color='b',label='Actual Data')
for i in range(len(true_means)):
    plt.scatter(sorted(np.unique(z_plot))[i],sim_means[i],s=20,color=colorlist[i],marker=markers[i])
    plt.scatter(sorted(np.unique(z_plot))[i],true_means[i],s=20,color=colorlist[i],marker=markers[i])


plt.legend(loc='upper left')
plt.savefig(outputdir + 'endogVinst.pdf')
plt.show()