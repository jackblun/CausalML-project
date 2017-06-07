#deepiv_plots.r
#plot stuff from python in ggplot
library(ggplot2)
#import Settler mortality dataset
setwd('/home/luis/CausalML-project/DeepIV')

settlermort <- read.csv('/../Data/colonial_origins_data_missimp.csv')
#keep obs with settler mortality data since it is the instrument of interest

#plot CV results
cv.first.AJR <- read.csv('/home/luis/CausalML-project/DeepIV/AJR/CV_first_stage_final.csv')
#remove missing/Inf values
cv.first.AJR$finite = is.finite(cv.first.AJR$LL_mean)

cv.first.AJR$LL_mean[!cv.first.AJR$finite]=NA
ggplot(data=cv.first.AJR[cv.first.AJR$comps>=1,],aes(x=nodes,y=LL_mean,colour=factor(comps))) +geom_point(alpha=.7) +
  ggtitle('Test CV over node / mixture choices with Local Smoothers')+scale_colour_discrete(name='# of Normal Mixtures') +geom_smooth(fill=NA) +
  xlab('# of Hidden Layer Nodes') + ylab('Negative Log-Likelihood')
ggsave('/home/luis/CausalML-project/DeepIV/AJR/CV_1stStage_graph.pdf')


ggplot(data=cv.first.AJR[cv.first.AJR$comps==2,],aes(x=nodes,y=LL_mean,colour=factor(comps))) +geom_point(alpha=.7) + geom_line() +
  geom_ribbon(aes(ymin=LL_mean-LL_sd, ymax=LL_mean+LL_sd),fill=NA,linetype='dashed') +
  ylab('MSE') + xlab('# of Hidden Layer Nodes')

cv.second.AJR <- read.csv('/home/luis/CausalML-project/DeepIV/AJR/cv_mp_output/cv_secondstage.csv')
ggplot(data=cv.second.AJR,aes(x=node,y=mean)) + geom_line()+ geom_ribbon(aes(ymin=mean-se, ymax=mean+se),fill=NA,colour='red',linetype='dashed',alpha=0.3) +
ylab('MSE') + xlab('# of Hidden Layer Nodes')  + ggtitle('Test CV over node choices for second stage +/- CV SE') + coord_cartesian(ylim = c(0,3))
ggsave('/home/luis/CausalML-project/DeepIV/AJR/CV_2ndStage_graph.pdf')

#######################
cv.first.AK <-read.csv('/home/luis/CausalML-project/DeepIV/AK/CV_first_stage.csv')
#remove missing/Inf values
cv.first.AK$finite = is.finite(cv.first.AK$LL_mean)
ggplot(data=cv.first.AK[cv.first.AK$comps>=1,],aes(x=nodes,y=LL_mean,colour=factor(comps))) +
  geom_line(alpha=.7) + geom_ribbon(aes(ymin=LL_mean-LL_sd, ymax=LL_mean+LL_sd),fill=NA,linetype='dashed',alpha=0.3) +
  ggtitle('Test CV over node / mixture choices with Local Smoothers')+scale_colour_discrete(name='# of Normal Mixtures') +
  xlab('# of Hidden Layer Nodes') + ylab('Negative Log-Likelihood')


cv.first.AK.MN <-read.csv('/home/luis/CausalML-project/DeepIV/AK/CV_first_stage_MN.csv')
ggplot(data=cv.first.AK.MN,aes(x=nodes,y=LL_mean)) +
  geom_line(alpha=1) + geom_ribbon(aes(ymin=LL_mean-LL_sd, ymax=LL_mean+LL_sd),linetype='dashed',alpha=.5) +
  ggtitle('Test CV over node choices with Local Smoothers')+scale_colour_discrete(name='# of Normal Mixtures') +
  xlab('# of Hidden Layer Nodes') + ylab('Negative Log-Likelihood')


cv.second.AK <- read.csv('/home/luis/CausalML-project/DeepIV/AK/cv_mp_output/cv_secondstage.csv')
ggplot(data=cv.second.AK,aes(x=node,y=mean)) + geom_line()+ geom_ribbon(aes(ymin=mean-se, ymax=mean+se),fill=NA,colour='red',linetype='dashed',alpha=0.3) +
  ylab('MSE') + xlab('# of Hidden Layer Nodes')  + ggtitle('Test CV over node choices for second stage +/- CV SE')
ggsave('/home/luis/CausalML-project/DeepIV/AK/CV_2ndStage_graph.pdf')
