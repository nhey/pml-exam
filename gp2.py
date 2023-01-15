from functools import partial
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

xstar = torch.linspace(-1,1,200).detach()

# TODO Only one sample from MCMC good enough quality?
def sample_posterior(model, S=1, C=1, W=1000):
  nuts_kernel = pyro.infer.NUTS(model)
  mcmc=pyro.infer.MCMC(nuts_kernel, num_samples=S, num_chains=C, warmup_steps=W)
  mcmc.run()
  # Trim "kernel." prefix from variable names and keep only
  # last sample for scalar values.
  return {k[len("kernel."):]: v[-1] for k, v in mcmc.get_samples().items()}

# Bayesian optimisation
def bayesian_optimisation(kernel, kernel_priors, D, xstar, T):
  # kernel is a function that returns the kernel when given args.
  # kernel_priors is the kernel instantiated with priors as args.
  # D is the initial dataset {(x_1, f(x_1)), ..., (x_n, f(x_n))}.
  # xstar is the candidate set {x^*_1, ..., x^*_l}.
  # T number of iterations.
  n = D[0].shape[0]
  # Allocate dataset arrays to hold iteration results.
  x, y = torch.empty(n+T), torch.empty(n+T)
  x[:n], y[:n] = D
  # Iteratively infer hyperparameters given D, predict best local minimum candidate
  # for evaluation and update D.
  kernel = partial(kernel, input_dim=1)
  for k in range(n,n+T):
    # Sample kernel hyperparameters from the posterior given current dataset D.
    pyro.clear_param_store()
    gpr = gp.models.GPRegression(
      x[:k], y[:k], kernel_prior, noise=torch.tensor(1e-4)
    )
    theta = sample_posterior(gpr.model)
    print(theta)
    gpr.kernel = kernel(**theta)

    # Sample f* ~ p(f*|X*,D).
    mean, var = gpr(xstar, full_cov=False, noiseless=False)
    fstar = dist.Normal(mean, var.sqrt()).rsample()
    # Find local minimum from predictions (sampled f*).
    p = torch.argmin(fstar)
    # Add (x^*_p, f(x^*_p)) to the dataset D.
    x[k], y[k] = xstar[p], f(xstar[p])
  return (x[n:], y[n:]), mean.detach(), var.detach()

# Need to instantiate kernel with priors here since the RBF (Isotropy)
# class wraps init arguments in PyroParam and we don't want that.
kernel_prior = gp.kernels.RBF(input_dim=1)
# note use of _log_normal: priors have support on the positive reals
kernel_prior.lengthscale = pyro.nn.PyroSample(dist.LogNormal(-1.0, 1.0))
kernel_prior.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 2.0))

(xp, yp), mean, var = bayesian_optimisation(gp.kernels.RBF, kernel_prior, (x, y), xstar, 5)
print(x, y)
gp_plot(f, x, y, xstar, mean, var, filename="bayesian_optimisation.pdf",
  xnew=xp, ynew=yp,
)

# From anaylising plots from multiple reruns it seems that a single observed
# value at one of the extremem minimums will cause the variance to include
# f(x) at both extremes.
