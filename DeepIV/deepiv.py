#Attempt to fit a deep IV model to an empirical dataset
#v0.1 - attempt to fit mixture density network a la Bishop (1994)

#heavily inspired by 
#http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/
import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mdn #library for mixture density network estimation I made


#given set of features, predict our DNN conditional distribution for the first stage.
def predict_1stStage_cond_dist(features,W_features,B_features,W_hidden,B_hidden):
    hidden = np.tanh(np.dot(features,W_features) + B_features)
    output = np.dot(hidden,W_hidden) + B_hidden
    probs,means,sds = np.split(output,3,axis=1)
    sds = np.exp(sds)
    probs = np.exp(probs)/np.sum(np.exp(probs),axis=1)[:,np.newaxis]
    return probs,means,sds


#stdize variables that are not dummies to be stdizing mean 0 and sd 1,
#and remove all covariates that have no variation in the data
#so that all behave well when fed into DNN
def process_features(features):
    collin_vars = [] #indices of variables with no variation
    feature_means = [] #store means we normalize
    feature_sds = [] #store SDs we normalize
    for v in range(features.shape[1]):
        #remove variables with one unique value- they mess stuff up later
        if len(np.unique(features[:,v].astype(np.float32)))==1:
            collin_vars.append(v)
            continue
        #skip normalizations for dummies (although I guess it doesn't really matter)
        is_dummy = (np.unique(features[:,v].astype(np.float32))==np.array([0.,1.]))   
        if isinstance(is_dummy,bool):
            if is_dummy:
                feature_means.append(0) #for dummies do not transform
                feature_sds.append(1)
                continue
        else:
            if is_dummy.all():
                feature_means.append(0) #for dummies do not transform
                feature_sds.append(1)
                continue  
        feature_means.append(np.mean(features[:,v])) #for dummies do not transform
        feature_sds.append(np.std(features[:,v]))    
        features[:,v] = (features[:,v] - np.mean(features[:,v]))/np.std(features[:,v])
    return [features,feature_means,feature_sds,collin_vars]


