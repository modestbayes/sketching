library(ggplot2)

path <- rbind(cbind(bootstrap$V1[1:20], bootstrap$V2[1:20]), 
              cbind(gaussian$V1[1:20], gaussian$V2[1:20]), 
              cbind(hadamard$V1[1:20], hadamard$V2[1:20]))

path <- as.data.frame(path)

path$method <- c(rep('subsampling', 20), rep('gaussian', 20), rep('hadamard', 20))

colnames(path)[1:2] <- c('beta1', 'beta2')

dataY <- as.matrix(dataY)
dataX <- as.matrix(dataX)
summary(lm(dataY ~ 0 + dataX))

dens <- matrix(NA, nrow=400, ncol=3)
for (i in 1:20) {
  for (j in 1:20) {
    dens[i + 20 * (j - 1), 1] <- 0.4 + 0.05 * i
    dens[i + 20 * (j - 1), 2] <- -2.4 + 0.05 * j
    dens[i + 20 * (j - 1), 3] <- dnorm(0.4 + 0.05 * i, 0.94, 0.1) * dnorm(-2.4 + 0.05 * j, -1.8, 0.1)
  }
}
dens <- as.data.frame(dens)
colnames(dens) <- c('x', 'y', 'p')

ggplot(data=path) +
  geom_path(aes(x=beta1, y=beta2, colour=method, linetype=method)) +
  geom_contour(data=dens, aes(x=x, y=y, z=p), colour='black')

S <- c(14.56, 6.26, 3.03)
G <- c(13.07, 5.92, 3.16)
H <- c(12.88, 4.82, 2.07)

ratio <- matrix(NA, nrow=9, ncol=2)
ratio[, 1] <- rep(c(128, 256, 512), 3)
ratio[, 2] <- as.numeric(c(S, G, H))

ratio <- as.data.frame(ratio)
colnames(ratio) <- c('r', 'ratio')
ratio$method <- c(rep('subsampling', 3), rep('gaussian', 3), rep('hadamard', 3))

ggplot(data=ratio) +
  geom_point(aes(x=r, y=ratio, colour=method)) +
  geom_path(aes(x=r, y=ratio, colour=method))

