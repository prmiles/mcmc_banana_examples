# Banana Example Using MCMC
#
# Written by Isaac Michaud
# Adapted by Paul Miles
# Referenced from: 
# Laine M (2008). Adaptive MCMC Methods with Applications in Environmental and Models. 
# Finnish Meteorological Institute Contributions 69. ISBN 978-951-697-662-7.

# add/install required package
# install.package('FME')
library(FME) # package for running MCMC simulations

# define inverse banana function
inv_banana <- function(y1,y2,a=1,b=1) {
  x1 = y1/a
  x2 = a*(y2 + b*((a*x1^2)+a^2))
  return(c(x1,x2))
}

# define function to be called in MCMC simulation
fme_banana <- function(y1,y2) {
  return(inv_banana(y1,y2)[2])
}

# Define a function that estimates the probability of a multinormally distributed vector
pmultinorm <- function(vec, mean, Cov) {
  diff <- vec - mean
  ex   <- -0.5*t(diff) %*% solve(Cov) %*% diff
  rdet   <- sqrt(det(Cov))
  power  <- -length(diff)*0.5
  return((2.*pi)^power / rdet * exp(ex))
}


# Define the target function which returns the -2*log(probability) of the value
BananaSS <- function (p)
{
  P <- c(p[1], fme_banana(p[1], p[2]))
  Cov <- matrix(nr = 2, data = c(1, 0.9, 0.9, 1))
  -2*sum(log(pmultinorm(P, mean = 0, Cov = Cov)))
}


# Run MCMC Simulations
# Metropolis Hastings
MCMC_MH <- modMCMC(f = BananaSS, p = c(0, 0.5), jump = diag(nrow = 2, x = 5), niter = 2000)
MCMC_MH$count

# Adaptive Metropolis
MCMC_AM <- modMCMC(f = BananaSS, p = c(0, 0.5), jump = diag(nrow = 2, x = 5), updatecov = 100, niter = 2000)
MCMC_AM$count

# Delayed Rejection
MCMC_DR <- modMCMC(f = BananaSS, p = c(0, 0.5), jump = diag(nrow = 2, x = 5), ntrydr = 2, niter = 2000)
MCMC_DR$count

# Delayed Rejection Adaptive Metropolis
print(system.time(
  MCMC_DRAM <- modMCMC(f = BananaSS, p = c(0, 0.5), jump = diag(nrow = 2, x = 5), updatecov = 100, ntrydr = 2, niter = 2000)
))
MCMC_DRAM$count

# Plot Chains
par(mfrow = c(4, 2))
par(mar = c(2, 2, 4, 2))
plot(MCMC_MH, mfrow = NULL, main = "MH")
plot(MCMC_AM, mfrow = NULL, main = "AM")
plot(MCMC_DR, mfrow = NULL, main = "DR")
plot(MCMC_DRAM, mfrow = NULL, main = "DRAM")
mtext(outer = TRUE, side = 3, line = -2, at = c(0.05, 0.95),
      c("y1", "y2"), cex = 1.25)
par(mar = c(5.1, 4.1, 4.1, 2.1))


# Plot Pairwise Correlation
par(mfrow = c(2, 2))
xl <- c(-3, 3)
yl <- c(-8, 1)
plot(MCMC_MH$pars,  main = "MH", xlim = xl, ylim = yl)
plot(MCMC_AM$pars, main = "AM", xlim = xl, ylim = yl)
plot(MCMC_DR$pars, main = "DR", xlim = xl, ylim = yl)
plot(MCMC_DRAM$pars, main = "DRAM", xlim = xl, ylim = yl)

# Write chains to text file
write.table(MCMC_MH$pars, file="mh.txt", sep = ',', row.names=FALSE, col.names=FALSE)
write.table(MCMC_AM$pars, file="am.txt", sep = ',', row.names=FALSE, col.names=FALSE)
write.table(MCMC_DR$pars, file="dr.txt", sep = ',', row.names=FALSE, col.names=FALSE)
write.table(MCMC_DRAM$pars, file="dram.txt", sep = ',', row.names=FALSE, col.names=FALSE)
