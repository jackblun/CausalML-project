#file:train_second_stage.py
#using the DNN trained in the first stage as an input, estimate
#the equation relating the covariates to the structural equation for outcome

import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mdn #my created library of mixture density networks
import deepiv #store deepiv functions in here
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
features_first =  np.concatenate((z, x), axis=1) #combine x + z

#transform the variables exactly as we did for first stage estimation of NN
features_first=np.delete(features_first,first_mdn['excluded_vars'],axis=1)
for v in range(features_first.shape[1]):
    features_first[:,v] = (features_first[:,v] - first_mdn['stdizing_means'][v] )/ first_mdn['stdizing_sds'][v]
#pd.DataFrame(covars).describe()

#second stage features include controls, the missing dummy for the instrument, and the policy var
features_second = np.concatenate((p, x, mi_z), axis=1)
#remove non-varying vars again, and stdize the means again
#keep p moments for when we draw from the conditional distribution and need to feed it into NN
p_mean = np.mean(p)
p_sd = np.std(p)
features_second,feature2_means,feature2_sds,collin_vars2 = deepiv.process_features(features_second)
features_second=np.delete(features_second,collin_vars2,axis=1)


#test the function
pi,m,s = deepiv.predict_1stStage_cond_dist(features_first, first_mdn['W_in'], \
        first_mdn['B_in'],first_mdn['W_out'],first_mdn['B_out'])




#estimate loss function (for validation training)
#args: test_y is true outcome data,
#outcome dnn is the DNN trained to predict y, given inputs
#session is the current tensorflow session being used
#features2 is the set of 2nd stage features,
#probs/means/sds1 are the first stage cond. distro parameters,
#B is the number of simulations per obs
#p_index is the column index of the policy variable we simulate in the feature matrix
def secondstage_loss(outcome,outcome_dnn,session,features2,probs1,means1,sds1,B=1000,p_index=0):
    mc_outcomes = np.zeros(shape = (outcome.shape[0],B))
    mc_policy = mdn.sim_mdn(probs1,means1,sds1,num_sims=B)
    temp_features = features2
    for b in range(B):
        temp_features[:,p_index] = mc_policy[:,b]
        mc_outcomes[:,b] = session.run(outcome_dnn,feed_dict={inputs: temp_features.astype(np.float32)}).flatten()
    pred_y_hat = np.mean(mc_outcomes,axis=1)
    pred_y_hat.shape = [pred_y_hat.shape[0],1]
    return np.mean((pred_y_hat - outcome)**2.)

#fcn for gradient for SGD
#args: outcome is the real data, 
#features2 are second stage features
#p_index is the location of policy variable in the feature matrix
#pi/mu/sigam1 are conditional distro of each obs
#outcome_dnn is the output layer of 2nd stage dnn fcn
#grad_fcn calculates the gradients of the loss
#B is number of simulations for the gradient
#session is cur tf session
#currently accepts just one observation 
#NOT FINISHED; SEE BELOW WHERE B=1 AND N=1
def ind_secondstage_loss_gradient(outcome,features2,pi1,mu1,sigma1, \
        outcome_dnn,grad_fcn,session,p_index=0):
    #correct one obs issue w/ array instead of mat
    #print pi1.shape
    p1 = mdn.sim_mdn(pi1,mu1,sigma1,num_sims=1)
    #print pi1.shape
    p2 = mdn.sim_mdn(pi1,mu1,sigma1,num_sims=1)
    tempfeat_1 = features2
    tempfeat_2 = features2
    tempfeat_1[:,p_index] = p1
    tempfeat_2[:,p_index] = p2
    #print"-----"
    pred_outcome = session.run(outcome_dnn,feed_dict={inputs: tempfeat_1.astype(np.float32)})
    grad = session.run(grad_fcn,feed_dict={inputs: tempfeat_2.astype(np.float32)})
    #print(grad)
    multiplier = -2.* (outcome - pred_outcome)
    newgrad=[]
    for g in range(len(grad)):
        newgrad.append(multiplier*grad[g])
    return newgrad


#workflow:
#need to estimate the 2nd stage loss function; do this by, 
#initialize dnn to random one including all features besides the instruments.
#in loop:
# 1.sampling 1 obs per epoch
# 2.sampling to policy outcomes per obs
# 3.calculating gradient of DNN w.r.t. this obs via tf
# 4.step in that direction

# 5. to evaluate the loss for the CV step, across all obs, draw from the conditional distro
#    a lot of times, calc outcome dnn for each, use this to estimate the integral, then subtract from truth squared
#
#rinse and repeat a million times or whatever
# and from each drawing a policy variable; then calculating the gradient of the DNN of 2nd stage on each 
# obs.

#some test code below
seed=1992
np.random.seed(seed)
num_nodes=5
num_inputs = features_second.shape[1] #the number of input features
num_output = 1 # output layer (currently just one since outcome is one variable)
num_obs = y.shape[0]


#initialize weights and biases for input->hidden layer
W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_in')
b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_in')


#initialize weights and biases for hidden->output layer
W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='W_out')
b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32,seed=seed),name='B_out')

#instantiate data vars
inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
outcome = tf.placeholder(dtype=tf.float32, shape=[None,1], name="outcome")
#define the function for the hidden layer
#use canonical tanh function for intermed, simple linear combo for final layer
hidden_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
outcome_layer = tf.matmul(hidden_layer,W_output) + b_output
#the gradients of the output layer w.r.t. network parameters
nn_gradients = tf.gradients(outcome_layer, [W_input, b_input,W_output,b_output]) #the gradients of the 

#placeholders for gradients I pass from numpy
g_W_in = tf.placeholder(dtype=tf.float32, shape=W_input.get_shape(), name="g_W_in")
g_b_in = tf.placeholder(dtype=tf.float32, shape=b_input.get_shape(), name="g_b_in")
g_W_out = tf.placeholder(dtype=tf.float32, shape=W_output.get_shape(), name="g_W_out")
g_b_out = tf.placeholder(dtype=tf.float32, shape=b_output.get_shape(), name="g_b_out")



#now try to fit the NN against the Loss fcn
#trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
trainer = tf.train.GradientDescentOptimizer(learning_rate=.001)

s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

#testing functions
#test_y = s.run(outcome_layer,feed_dict={inputs: features_second.astype(np.float32)})
#test_grad = s.run(nn_gradients,feed_dict={inputs: features_second.astype(np.float32)})
#L=secondstage_loss(y,outcome_layer,s,features_second,first_mdn['pi'],first_mdn['mu'],first_mdn['sigma'],B=10)

#test out the gradient fcn
#ind=2
#obs_feat= features_second[ind,:]
#obs_y = y[ind]
#pi_i = first_mdn['pi'][ind,:]
#mu_i = first_mdn['mu'][ind,:]
#sd_i = first_mdn['sigma'][ind,:]
#gg = ind_secondstage_loss_gradient(obs_y,obs_feat,pi_i,mu_i,sd_i,outcome_layer,nn_gradients,s)


#apply_grad = trainer.apply_gradients(placeholder_gradients)


grad_var_pairs = zip([g_W_in,g_b_in,g_W_out,g_b_out],[W_input,b_input,W_output,b_output])

validation_losses=[]
validation_indices = np.random.choice(num_obs,num_obs/5)
train_indices = np.ones(len(p), np.bool)
train_indices[validation_indices]=0
y_validation = y[validation_indices]
features_validation = features_second[validation_indices,:]
y_train = p[train_indices]
features_train  = features_second[train_indices,:]
num_train_obs = sum(train_indices)
print "training..."
num_iters = 10000
losses=[]
for i in range(num_iters):
    if i%100==0:
        print i
    g_ind=np.random.choice(num_train_obs,1)[0]
    obs_feat= features_second[train_indices,:][g_ind,:]
    obs_y = y[train_indices][g_ind]
    pi_i = first_mdn['pi'][train_indices,:][g_ind,:]
    mu_i = first_mdn['mu'][train_indices,:][g_ind,:]
    sd_i = first_mdn['sigma'][train_indices,:][g_ind,:]
    #reshape everything so treated as 2d
    for v in [obs_y, obs_feat, pi_i, mu_i ,sd_i]:
        v.shape = [1,len(v)]

    stoch_grad = ind_secondstage_loss_gradient(obs_y,obs_feat,pi_i,mu_i,sd_i,outcome_layer,nn_gradients,s)
    grad_dict={}
    grad_index=0
    for theta in [g_W_in,g_b_in,g_W_out,g_b_out]:
        grad_dict[theta]=stoch_grad[grad_index]
        grad_index+=1
    s.run(trainer.apply_gradients(grad_var_pairs),feed_dict=grad_dict)
    #the gradients of the output layer w.r.t. network parameters
    if i%10==0:
        loss=secondstage_loss(y[validation_indices],outcome_layer,s,\
            features_second[validation_indices,:], \
            first_mdn['pi'][validation_indices,:], \
            first_mdn['mu'][validation_indices,:], \
            first_mdn['sigma'][validation_indices,:],B=100)
        validation_losses.append(loss)
        if len(validation_losses) > 5:
            if max(validation_losses[(len(validation_losses)-6):(len(validation_losses)-2)])< validation_losses[len(validation_losses)-1]:
                print "Exiting at iteration " + str(i) + " due to increase in validation error." 
                break
    #print "----------"

plt.plot(range(len(validation_losses)),validation_losses)
plt.show()