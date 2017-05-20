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
  ggtitle('Test CV Neg LL over node / mixture choices with Local Smoothers')+scale_colour_discrete(name='# of Normal Mixtures') +geom_smooth(se=FALSE)
ggsave('/home/luis/CausalML-project/DeepIV/CV_1stStage_graph.pdf')