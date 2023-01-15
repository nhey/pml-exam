import matplotlib
import matplotlib.pyplot as plt
import numpy

matplotlib.rcParams.update({'font.size': 18})

# Modified from https://pyro.ai/examples/gp.html.
def gp_plot(f, x, y, xstar, mean, var, filename="gp_plot.pdf", xnew=None, ynew=None):
  fig, ax = plt.subplots(figsize=(12, 6))

  # Visualize p(f*|x*,D).
  sd = var.sqrt()  # standard deviation at each input point x
  ax.plot(xstar.numpy(), mean.numpy(), "m", lw=3, ls="--", label="$m(x*)$")
  ax.fill_between(
      xstar.numpy(),  # plot the two-sigma uncertainty about the mean
      (mean - 2.0 * sd).numpy(),
      (mean + 2.0 * sd).numpy(),
      color="C0",
      alpha=0.3,
      label="$m(x*) \pm 2\sqrt{v(x*)}$",
  )

  # Plot f(x*).
  ax.plot(xstar.numpy(), f(xstar).numpy(), "k", lw=2, label="$f(x)$")
  # Scatter plot D.
  ax.plot(x.numpy(), y.numpy(), "kx", markersize=15, lw=4, label="$\mathcal{D}$")
  # Scatter plot (x*, f(x*)) added to D.
  if xnew is not None:
    assert ynew is not None
    ax.plot(xnew.numpy(), ynew.numpy(), "k*", markersize=16, label="$(x_p^*, f(x_p^*))$")

  plt.xlabel("x")
  plt.ylabel("y")

  plt.legend()
  plt.tight_layout()
  plt.savefig(filename)
