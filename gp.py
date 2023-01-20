import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import arviz

from util import gp_plot
from util import posterior_plot

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
#
# Params set after analysis:
S = 500
C = 4
W = 1000
nuts_kernel = pyro.infer.NUTS(gpr.model) # jit_compile=True emits warnings
mcmc=pyro.infer.MCMC(nuts_kernel, num_samples=S, num_chains=C, warmup_steps=W)
mcmc.run()
posterior_samples = mcmc.get_samples()

# 2. Check the quality of the MCMC sampling using diagnostics (hint: use Arviz).
# Use the diagnostics to choose the hyperparameters of the sampling (such as
# the number of warmup samples).
data = arviz.from_pyro(mcmc)
summary = arviz.summary(data, hdi_prob=0.95)
print(summary)

# 3. Use the obtained MCMC samples from the posterior to obtain estimates of
# mean m(x∗) and variance v(x∗) of p(y∗|x∗,D) at a point x∗ ∈ [−1, 1].
#
# The predictive distribution given by marginalising out hyperparameters theta:
#   p(y*|x*,D) = int p(y*|x*,D,theta)p(theta|D) dtheta
# is intractable. So we use a Monte Carlo estimate.
xstar = torch.linspace(-1,1,200)

def conditional(xstar, lengthscale, variance):
  # Compute mean and variance of f*|y.
  with torch.no_grad():
    gpr = gp.models.GPRegression(x, y, kernel, noise=torch.tensor(1e-4))
    gpr.kernel.variance = variance
    gpr.kernel.lengthscale = lengthscale

    mean, var = gpr(xstar, full_cov=False, noiseless=False)
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

gp_plot(f, x, y, xstar, m, v)

# p(\theta|D) using 500 samples
posterior_plot(
  posterior_samples["kernel.lengthscale"][:500],
  posterior_samples["kernel.variance"][:500],
)
