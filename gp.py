import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import arviz

from util import gp_plot

def f(x):
  assert (x >= -1).all() and (x <= 1).all(), f"f({x}) is not in domain of f"
  return torch.sin(20*x) + 2*torch.cos(14*x) - 2*torch.sin(6*x)

x = torch.tensor([-1, -1/2, 0, 1/2, 1])
y = f(x)

# Define Gaussian process
kernel = gp.kernels.RBF(input_dim=1)
# note use of _log_normal: priors have support on the positive reals
kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(-1.0, 1.0))
kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 2.0))
gpr = gp.models.GPRegression(x, y, kernel, noise=torch.tensor(1e-4))


# 1. Use NUTS to sample from the posterior p(theta | D) for the GP parameters
S = 100 # TODO 500
C = 2
nuts_kernel = pyro.infer.NUTS(gpr.model) # jit_compile=True emits warnings
mcmc=pyro.infer.MCMC(nuts_kernel, num_samples=S, num_chains=C, warmup_steps=100)
mcmc.run()
posterior_samples = mcmc.get_samples()

# 2. Check the quality of the MCMC sampling using diagnostics (hint: use Arviz).
# Use the diagnostics to choose the hyperparameters of the sampling (such as
# the number of warmup samples).
# data = arviz.from_pyro(mcmc)
# summary = arviz.summary(data, hdi_prob=0.95)
# print(summary)
# TODO

# 3. Use the obtained MCMC samples from the posterior to obtain estimates of
# mean m(x∗) and variance v(x∗) of p(y∗|x∗,D) at a point x∗ ∈ [−1, 1].
#
# The predictive distribution given by marginalising out hyperparameters theta:
#   p(y*|x*,D) = int p(y*|x*,D,theta)p(theta|D) dtheta
# is intractable. So we use a Monte Carlo estimate.
xstar = torch.linspace(-1,1,101)

# Compute mean and variance of f*|y.
def conditional(xstar, lengthscale, variance):
  with torch.no_grad():
    # Local GP object
    lgp = gp.models.GPRegression(x, y, kernel, noise=torch.tensor(1e-4))
    lgp.kernel.variance = variance
    lgp.kernel.lengthscale = lengthscale

    mean, var = lgp(xstar, full_cov=False, noiseless=False)
  return mean, var

# Monte Carlo estimate of m(x*) and v(x*).
means, vars = zip(*map(
  lambda l, s: conditional(xstar, l, s),
  posterior_samples["kernel.lengthscale"],
  posterior_samples["kernel.variance"],
))
means = torch.cat(means).view(-1, len(xstar)) # torch.vmap where you at?
vars = torch.cat(vars).view(-1, len(xstar))
assert means.shape[0] == S*C
assert vars.shape[0] == S*C

m = means.mean(0)
v = vars.mean(0)
# TODO^ not entirely sure if this is right
# or if we should do the "predicitons" and then mean (is there
# even a difference? its a simple mean both ways)

gp_plot(f, x, y, xstar, m, v)


#
# See plotting code from GP tutorial on how to use the model
# (we get mean, cov; plot mean and use cov to get std to show uncertainty).
# But unsure how to use model when integrating over sampled params.
#
# posterior predictive follow tricks on GLM slides week 1, ends at slide 34--35
S = x
f_S = y
# by pp 45: p(y|f_S) = N(y;f_S,I) since y is noise-free?
# Not even sure, since this follows from y = g(x) + e with e = N(0,I).
# And our y (so far) is y = g(x) meaning f* and y* are interchangable.
#
# Look at https://num.pyro.ai/en/0.7.1/examples/gp.html
# and refer to Script pp. 47, how are the many samples of the posterior
# handled? Are they actually integrated out?
# Not sure if the hint in B.1 excludes running p(f*|x*,D) for each
# parameter and then averaging _afterwards_?
# Also try to make sense of how the link to the slides fits into
# this context---there, theta is actually marginalized.
