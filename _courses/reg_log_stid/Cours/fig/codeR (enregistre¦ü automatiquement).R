setwd("/Volumes/lchapel/Mes documents/Enseignement/2017-2018/Régression logistique STID/Cours/fig")



chd <- read.table("chddata.txt")

pdf("dataPoints.pdf", width=6, height=3.5, pointsize=8)
plot(chd[,2], chd[,3], xlab="x", ylab="y")
dev.off()

	pdf("regheart.pdf", width=6, height=3.5, pointsize=8)
plot(chd[,2], chd[,3], xlab="x", ylab="y")
abline(lm(chd[,3]~chd[,2]), col=2)
dev.off()

age.means <- rep(0,10)
chd.means <- rep(0,10)
for(i in 0:9) age.means[i+1]<-mean(chd[(10*i+1):(10*i+10),2])
for(i in 0:9) chd.means[i+1]<-mean(chd[(10*i+1):(10*i+10),3])

pdf("pmaladieage.pdf", width=6, height=3.5, pointsize=8)
plot(age.means,chd.means, xlab="x", ylab="P(Y=1|x)")
lines(lowess(age.means,chd.means,iter=1,f=2/3), col=2)
dev.off()

pdf("logit.pdf", width=6, height=3.5, pointsize=8)
x<-seq(0.01,0.99,0.01)
plot(log(x/(1-x)), x, type='l', ylab="pi(x)", xlab="pi(x) / (1-pi(x))")
dev.off()

pdf("ajustLogit.pdf", width=6, height=3.5, pointsize=8)
fit.chd <- glm(chd[,3] ~chd[,2], family=binomial(link="logit"))
logit.inverse <- function(x){1/(1+exp(-x))}
plot(age.means,chd.means, xlab="x", ylab="P(Y=1|x)", ylim=c(0,1))
lines(chd[,2],logit.inverse(predict(fit.chd)), col=2)
dev.off()



#TD
data("womensrole", package = "HSAUR")
fm1 <- cbind(agree, disagree) ~ sex + education
womensrole_glm_1 <- glm(fm1, data = womensrole,family=binomial())
summary(womensrole_glm_1)


