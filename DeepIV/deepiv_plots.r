#deepiv_plots.r
#plot stuff from python in ggplot
library(ggplot2)
#import Settler mortality dataset
setwd('/home/luis/CausalML-project/DeepIV')

settlermort <- read.csv('/../Data/colonial_origins_data_missimp.csv')
#keep obs with settler mortality data since it is the instrument of interest

#plot CV results
cv.first <- read.csv('/home/luis/CausalML-project/DeepIV/CV_first_stage.csv')
#remove missing/Inf values
cv.first$finite = is.finite(cv.first$LL)

cv.first$LL[!cv.first$finite]=NA
ggplot(data=cv.first[cv.first$comps>=1,],aes(x=nodes,y=LL,colour=factor(comps))) +geom_point(alpha=.7) +
  ggtitle('Test CV over node / mixture choices with Local Smoothers')+scale_colour_discrete(name='# of Normal Mixtures') +geom_smooth(fill=NA) +
  xlab('# of Hidden Layer Nodes') + ylab('Negative Log-Likelihood')
ggsave('/home/luis/CausalML-project/DeepIV/CV_1stStage_graph.pdf')

cv.second <- read.csv('/home/luis/CausalML-project/DeepIV/cv_mp_output/cv_secondstage.csv')
ggplot(data=cv.second,aes(x=node,y=mean)) + geom_line()+ geom_ribbon(aes(ymin=mean-se, ymax=mean+se),fill=NA,colour='red',linetype='dashed',alpha=0.3) +
ylab('MSE') + xlab('# of Hidden Layer Nodes')  + ggtitle('Test CV over node choices for second stage +/- CV SE') + coord_cartesian(ylim = c(0,3))
ggsave('/home/luis/CausalML-project/DeepIV/CV_2ndStage_graph.pdf')
