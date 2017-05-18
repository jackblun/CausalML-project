#deepiv.r
#attempt to recreate the deepIV method in R
#using largely canned techniques

library(deepnet)
#import Settler mortality dataset
setwd('/home/luis/CausalML-project/')

settlermort <- read.csv('Data/colonial_origins_data_missimp.csv')
#keep obs with settler mortality data since it is the instrument of interest
settlermort <- settlermort[settlermort$mi_logem4==0 & settermort$mi_avexpr==0,]
#unlog settler mortality since we are going to do an NP analysis anyway
settlermort$em4 <- exp(settlermort$logem4 )

#outcome is logpgp95
#predict average expropriation using settler mortality + all other covariates
#consider doing the NeweyInstrumental variable estimation of nonparametric models.
#Econometrica, 71(5):1565–1578, 2003. as a baseline comparison
