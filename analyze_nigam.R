library(ggplot2)
library(reshape2)
setwd("~/Documents/other_research/naivebayes")

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

alpha = read.csv("nigam_alpha_tests.csv")

alpha.long  = melt(alpha, id.vars=c("n.labeled", "alpha", "iteration"))

ggplot(alpha.long, aes(x=alpha, y=value, color=variable)) +
  #scale_x_log10() +
  scale_color_manual(values=cbPalette) +
  stat_smooth()


### ACTUAL NIGAM REPLICATION
nigam = read.csv("nigam_et_al_repl.csv")
nigam.long = melt(nigam, id.vars = c("n.labeled", "iteration"))
nigam.recast = dcast(nigam.long, n.labeled ~ variable, fun.aggregate=mean)
# could probably do this with dplyr
nigam.long.mean = melt(nigam.recast, id.vars=c("n.labeled"))

ggplot(nigam.long.mean, aes(x=n.labeled, y=value, color=variable)) +
  scale_x_log10(limits=c(10, max(nigam$n.labeled))) +
  scale_y_continuous(limits=c(0,1)) +
  scale_color_manual(values=cbPalette) +
  geom_line()

ggplot(nigam.long, aes(x=n.labeled, y=value, color=variable)) +
  scale_x_log10(limits=c(10, max(nigam$n.labeled))) +
  scale_y_continuous(limits=c(0,1)) +
  scale_color_manual(values=cbPalette) +
  geom_point()