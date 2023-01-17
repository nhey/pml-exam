import matplotlib
import matplotlib.pyplot as plt
import numpy

matplotlib.rcParams.update({'font.size': 18})

# Modified from https://pyro.ai/examples/gp.html.
def gp_plot(f, x, y, xstar, mean, var, filename="gp_plot.pdf"):
  fig, ax = plt.subplots(figsize=(12, 6))

  # Visualize p(f*|x*,D).
  sd = var.sqrt()  # standard deviation at each input point x
  ax.plot(xstar.numpy(), mean.numpy(), "m", lw=3, ls="--", label="$m(x^*)$")
  ax.fill_between(
      xstar.numpy(),  # plot the two-sigma uncertainty about the mean
      (mean - 2.0 * sd).numpy(),
      (mean + 2.0 * sd).numpy(),
      color="C0",
      alpha=0.3,
      label="$m(x^*) \pm 2\sqrt{v(x^*)}$",
  )

  # Plot f(x*).
  ax.plot(xstar.numpy(), f(xstar).numpy(), "k", lw=2, label="$f(x)$")
  # Scatter plot D.
  ax.plot(x.numpy(), y.numpy(), "kx", markersize=15, lw=4, label="$\mathcal{D}$")

  plt.xlabel("$x$")
  plt.ylabel("$y$")

  plt.legend()
  plt.tight_layout()
  plt.savefig(filename)

def bayesian_optimisation_plot(f, x, y, xstar, means, vars, xp, yp, ks, thetas, filename="bayesian_optimisation.pdf"):
  fig, axs = plt.subplots(len(ks), figsize=(12, 12))

  ymin = min(y)
  for ax, k in zip(axs, ks):
    mean, var = means[k], vars[k]
    xnew, ynew = xp[:k], yp[:k]
    ymin = min(ymin, min(ynew)) if len(ynew) > 0 else ymin
    # Visualize p(f*|x*,D).
    sd = var.sqrt()  # standard deviation at each input point x
    ax.plot(xstar.numpy(), mean.numpy(), "m", lw=3, ls="--", label="$m(x^*)$")
    ax.fill_between(
        xstar.numpy(),  # plot the two-sigma uncertainty about the mean
        (mean - 2.0 * sd).numpy(),
        (mean + 2.0 * sd).numpy(),
        color="C0",
        alpha=0.3,
        label="$m(x^*) \pm 2\sqrt{v(x^*)}$",
    )

    # Plot f(x*).
    ax.plot(xstar.numpy(), f(xstar).numpy(), "k", lw=2, label="$f(x)$")
    # Scatter plot D.
    ax.plot(x.numpy(), y.numpy(), "kx", markersize=15, lw=4, label="$\mathcal{D}$")
    # Scatter plot (x*, f(x*)) added to D.
    if xnew is not None:
      assert ynew is not None
      ax.plot(xnew.numpy(), ynew.numpy(), "k*", markersize=16, label="$(x_p^*, f(x_p^*))$")
      theta = ", ".join([f"{k} = {v:.1e}" for k,v in thetas[k].items()])
      ax.set_title("$k=$" + f"{k}" + ",   $y_{min}=$" + f"{ymin:.2f}" + ",  " + theta)
      ax.set_ylabel("$y$")

  axs[-1].set_xlabel("$x$")
  axs[-1].legend(ncol=5, bbox_to_anchor=(0, -0.55, 1, 0), loc="lower left", mode="expand")
  plt.tight_layout()
  plt.savefig(filename)

def posterior_plot(lengthscale, variance, filename="posterior_plot.pdf"):
  fig, ax = plt.subplots(figsize=(8, 8))

  # Scatter plot.
  ax.plot(lengthscale.numpy(), variance.numpy(), "kx", markersize=15, lw=4, alpha=0.65)

  plt.loglog()

  plt.xlabel("lengthscale")
  plt.ylabel("variance")
  plt.tight_layout()
  plt.savefig(filename)
