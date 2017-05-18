#Attempt to fit a deep IV model to an empirical dataset
#v0.1 - attempt to fit mixture density network a la Bishop (1994)

#heavily inspired by 
#http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt



datadir = '/home/luis/CausalML-project/Data/'

settlers = pd.read_csv(datadir+'colonial_origins_data_missimp.csv')
#remove those missing either outcome, endog institution measure, or exog instrument
nomiss = (settlers['mi_avexpr']==0) & (settlers['mi_logpgp95']==0)
settlers = settlers.loc[nomiss,:]
num_obs = settlers.shape[0]

p = np.array(settlers['avexpr']) #the endogenous variable
p.shape = [p.shape[0],1]#make it 2D
z = np.array(settlers.loc[:,['logem4','mi_logem4']]) #the instrument
#z.shape = [z.shape[0],] 

all_covars=np.r_[1:3, 7:8,10:52, 54,58:84]
#feature sets of covariates we might consider
#x = np.array(settlers.iloc[:,17:38])
#x = np.array(settlers.iloc[:,10:13])
#x = np.array(settlers.iloc[:,10:46]) 
x = settlers.iloc[:,all_covars] #this should be all of them


covars =  np.concatenate((z, x), axis=1)
collin_vars = [] #indices of variables with no variation
#stdize all non-dummy variables to have mean 0 and SD 1
for v in range(covars.shape[1]):
    
    #remove variables with one unique value- they mess stuff up later
    if len(np.unique(covars[:,v].astype(np.float32)))==1:
        collin_vars.append(v)
        continue
    #skip normalizations for dummies (although I guess it doesn't really matter)
    is_dummy = (np.unique(covars[:,v].astype(np.float32))==np.array([0.,1.]))   
    if isinstance(is_dummy,bool):
        if is_dummy:
            continue
    else:
        if is_dummy.all():
            continue        
    covars[:,v] = (covars[:,v] - np.mean(covars[:,v]))/np.std(covars[:,v])


covars=np.delete(covars,collin_vars,axis=1)


#plt.scatter(z,p)
#plt.show()


#begin to define some of the parameters for the MDN
num_layers = 1 #number of intermediate layers; fixed to 1 layer
num_components = 3 #the number of mixture components; later try to tune on them
num_inputs = covars.shape[1] #the number of input features
#num_nodes =num_inputs  #the number of nodes in the hidden layer
num_nodes=10
num_output = num_components*3


#initialize weights and biases for input->hidden layer
W_input = tf.Variable(tf.random_uniform(shape=[num_inputs,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32))
b_input = tf.Variable(tf.random_uniform(shape=[1,num_nodes],minval=-.1,maxval=.1,dtype=tf.float32))

#initialize weights and biases for hidden->output layer
W_output = tf.Variable(tf.random_uniform(shape=[num_nodes,num_output],minval=-.1,maxval=.1,dtype=tf.float32))
b_output = tf.Variable(tf.random_uniform(shape=[1,num_output],minval=-.1,maxval=.1,dtype=tf.float32))

#instantiate data vars
inputs = tf.placeholder(dtype=tf.float32, shape=[None,num_inputs], name="inputs")
endog = tf.placeholder(dtype=tf.float32, shape=[None,1], name="endog")



#define the function for the hidden layer
#use canonical tanh function for intermed, simple linear combo for final layer
#(note it will be further processed)
intermed_layer = tf.nn.tanh(tf.matmul(inputs, W_input) + b_input)
output_layer = tf.matmul(intermed_layer,W_output) + b_output

#transform the final layer into probabilities, means, and variances
def get_params(output):
    mixprobs = tf.placeholder(dtype=tf.float32, shape=[None,num_components], name="mixprobs")
    mixmeans = tf.placeholder(dtype=tf.float32, shape=[None,num_components], name="mixmeans")
    mixsds = tf.placeholder(dtype=tf.float32, shape=[None,num_components], name="mixsds")
    mixprobs,mixmeans,mixsds = tf.split(output,3,axis=1)
    mixprobs = tf.nn.softmax(mixprobs,dim=1) #normalized to be between 0 and 1 and sum to 1
    mixsds = tf.exp(mixsds) #so it's always positive
    return mixprobs,mixmeans,mixsds
mixprobs,mixmeans,mixsds=get_params(output_layer)
#define the loss function- here the log likelihood of the mixture given parameters

#fcn for getting Lhood of a univariate normal 
def tf_normal(y,mean,sd):
    lhood = -tf.square(tf.subtract(y,mean)) #-(y-u)^2
    lhood = tf.divide(lhood,tf.multiply(2.,tf.square(sd)))
    lhood = tf.multiply(tf.exp(lhood),tf.divide(1./math.sqrt(2.*math.pi),sd))
    return lhood

#now the actual mixture l-lhood
def tf_mixlhood(probs,means,sds,y):
    lhood = tf_normal(y,means,sds)
    lhood = tf.reduce_sum(tf.multiply(probs,lhood),axis=1,keep_dims=True)
    return tf.reduce_mean(-tf.log(lhood))

#now try to fit the NN against the Loss fcn

loss = tf_mixlhood(mixprobs, mixmeans, mixsds, endog)
#trainer = tf.train.AdamOptimizer().minimize(loss)
trainer = tf.train.GradientDescentOptimizer(learning_rate=.001).minimize(loss)
#trainer = tf.train.GradientDescentOptimizer(learning_rate=1)
#trainer_calcGrad = trainer.compute_gradients()
s = tf.InteractiveSession()
s.run(tf.global_variables_initializer())

print "training..."
num_iters = 20000 #the number of gradient descents
losses = np.zeros(num_iters)

for i in range(num_iters):
  losses[i] = s.run(loss,feed_dict={inputs: covars.astype(np.float32), endog: p.astype(np.float32)})
  #s.run(trainer,feed_dict={inputs: covars.astype(np.float32), endog: p.astype(np.float32)})
  #SGD
  subsamp = np.random.choice(num_obs,num_obs/10)
  s.run(trainer,feed_dict={inputs: covars[subsamp,:].astype(np.float32), endog: p[subsamp].astype(np.float32)})

plt.plot(range(num_iters),losses)
plt.show()
#try a stochastic version where we do batches of 10



mixprobs,mixmeans,mixsds=s.run(get_params(output_layer),feed_dict={inputs:covars.astype(np.float32)})
######
#look at how we did