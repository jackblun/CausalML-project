#file:train_second_stage.py
#using the DNN trained in the first stage as an input, estimate
#the equation relating the covariates to the structural equation for outcome

import math
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import mdn #my created library
