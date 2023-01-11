import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

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
S = 100
nuts_kernel = pyro.infer.NUTS(gpr.model, jit_compile=True)
mcmc=pyro.infer.MCMC(nuts_kernel, num_samples=S, num_chains=2, warmup_steps=100)
mcmc.run()
samples = mcmc.get_samples()

print(y)
print(samples)
