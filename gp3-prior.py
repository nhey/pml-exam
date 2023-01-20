from functools import partial
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import arviz

from util import bayesian_optimisation_plot
from util import posterior_plot

def f(x):
  assert (x >= -1).all() and (x <= 1).all(), f"f({x}) is not in domain of f"
  return torch.sin(20*x) + 2*torch.cos(14*x) - 2*torch.sin(6*x)

x = torch.tensor([-1, -1/2, 0, 1/2, 1])
y = f(x)

xstar = torch.linspace(-1,1,200).detach()

import datetime
t = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
timestamp = True
t = "-" + str(t) if timestamp else ""
_S, _W = 501, 500

def sample_posterior(model, k=0, S=_S, C=1, W=_W):
  nuts_kernel = pyro.infer.NUTS(model)
  mcmc=pyro.infer.MCMC(nuts_kernel, num_samples=S, num_chains=C, warmup_steps=W)
  mcmc.run()
  posterior_samples = mcmc.get_samples()
  posterior_plot(
    posterior_samples["kernel.lengthscale"],
    posterior_samples["kernel.variance"],
    filename=f"prior{_S+_W}{t}posterior{k}.pdf"
  )
  # Trim "kernel." prefix from variable names and keep only
  # last sample for scalar values.
  return {k[len("kernel."):]: v[-1] for k, v in posterior_samples.items()}

# Bayesian optimisation
def bayesian_optimisation(kernel, D, xstar, T):
  # kernel:        A function that returns the kernel when given args.
  # D:             Initial dataset {(x_1, f(x_1)), ..., (x_n, f(x_n))}.
  # xstar:         Candidate set {x^*_1, ..., x^*_l}.
  # T:             Number of iterations.
  n = D[0].shape[0]
  # Allocate dataset arrays to hold iteration results.
  x, y = torch.empty(n+T), torch.empty(n+T)
  x[:n], y[:n] = D
  means, vars = torch.empty(T+1, len(xstar)), torch.empty(T+1, len(xstar))
  thetas = []
  # Iteratively infer hyperparameters given D, predict best local minimum
  # candidate for evaluation and update D.
  kernel = partial(kernel, input_dim=1)
  min = torch.min(y)
  for k in range(0,T):
    print(f"({k+1}/{T})")
    # Init GP on current D with prior hyperparameters.
    pyro.clear_param_store()
    gpr = gp.models.GPRegression(
      x[:n+k], y[:n+k], kernel_prior, noise=torch.tensor(1e-4)
    )
    # Save (hyper)parameters for plotting at k=0 (which is k=-1 here).
    if k == 0:
      means[k], vars[k] = gpr(xstar, full_cov=False, noiseless=False)
      thetas.append({p[0][:-4]: p[1] for p in kernel_prior.named_pyro_params()})
    # Sample kernel hyperparameters from the posterior.
    theta = sample_posterior(gpr.model, k=k+1)
    gpr.kernel = kernel(**theta)

    # Algorithm 1
    # Sample f* ~ p(f*|X*,D).
    mean, var = gpr(xstar, full_cov=False, noiseless=False)
    fstar = dist.Normal(mean, var.sqrt()).rsample()
    # Find local minimum of predictions (sampled function f*).
    p = torch.argmin(fstar)
    # Add (x^*_p, f(x^*_p)) to the dataset D.
    x[n+k], y[n+k] = xstar[p], f(xstar[p])

    # Save (hyper)parameters for plotting
    means[k+1], vars[k+1] = mean, var
    thetas.append(theta)
  return torch.min(y), ((x[n:], y[n:]), means.detach(), vars.detach(), thetas)

# Need to instantiate kernel with priors here since the RBF (Isotropy)
# class wraps init arguments in PyroParam and we don't want that.
kernel_prior = gp.kernels.RBF(input_dim=1)
kernel_prior.lengthscale = pyro.nn.PyroSample(dist.InverseGamma(2.0, 0.5))
kernel_prior.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 2.0))

ymin, ((xp, yp), means, vars, thetas) = bayesian_optimisation(gp.kernels.RBF, (x, y), xstar, 10)
print(thetas)

print("x*_p", xp)
print("y*_p", yp)
print("ymin", ymin)

bayesian_optimisation_plot(f, x, y, xstar, means, vars,
  xp, yp, [0,5,10], thetas, filename=f"prior{_S+_W}{t}.pdf"
)
