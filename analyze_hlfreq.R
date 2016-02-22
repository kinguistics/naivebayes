library(ggplot2)
library(sys)
setwd("~/Documents/other_research/naivebayes")

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

hlfreqs = read.csv("hl_freq_tests.csv")

ggplot(hlfreqs[which(hlfreqs$low_idx<50&hlfreqs$high_idx<10),], aes(x=ncols, y=score, color=lowest_freq)) +
  geom_point() +
  scale_x_log10()

for (hi in 1:100) {
  fname = paste("low_freq_hi_",hi,".png",sep="")
  print(fname)
  png(fname)
  p = ggplot(hlfreqs[which(hlfreqs$high_idx==hi),], aes(x=lowest_freq, y=score)) +
    geom_point() +
    scale_y_continuous(limits=c(0,1)) +
    scale_x_log10(limits=(c(1, max(hlfreqs$lowest_freq))))
  print(p)
  dev.off()
}

ggplot(hlfreqs[which(hlfreqs$high_idx==1),], aes(x=ncols, y=score)) +
  geom_point() +
  scale_x_log10(limits=(c(1, max(hlfreqs$ncols))))

hlfreqs[which(hlfreqs$score<0.1 & hlfreqs$ncols>20000),]

onedown = hlfreqs[which(hlfreqs$high_idx==1),]
lines(lowess(hlfreqs$ncols, hlfreqs$score))
