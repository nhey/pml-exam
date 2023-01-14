import matplotlib
import matplotlib.pyplot as plt
import numpy

matplotlib.rcParams.update({'font.size': 18})

# Modified from https://pyro.ai/examples/gp.html.
def gp_plot(f, x, y, xstar, mean, var):
  fig, ax = plt.subplots(figsize=(12, 6))

  # Visualize p(f*|x*,D).
  sd = var.sqrt()  # standard deviation at each input point x
  ax.plot(xstar.numpy(), mean.numpy(), "m", lw=3, label="$m(x*)$")  # plot the mean
  print(xstar.shape, sd.shape, mean.shape)
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

  plt.legend()
  plt.tight_layout()
  plt.savefig("gp_plot.pdf")
