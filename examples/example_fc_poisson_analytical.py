"""
Analytical solution for Poisson process with background.

Produces Fig. 7 from the Feldman Cousins paper.
"""
from functools import partial
from astropy.utils.console import ProgressBar
import numpy as np
import matplotlib.pyplot as plt
from gammapy.stats import fc_find_acceptance_interval_poisson, fc_fix_limits

if __name__ == "__main__":

    background = 3.0

    n_bins_x = 100
    step_width_mu = 0.005
    mu_min = 0
    mu_max = 50
    cl = 0.90

    x_bins = np.arange(0, n_bins_x)
    mu_bins = np.linspace(
        mu_min, mu_max, int(mu_max / step_width_mu) + 1, endpoint=True
    )

    print("Generating FC confidence belt for %s values of mu." % len(mu_bins))

    partial_func = partial(
        fc_find_acceptance_interval_poisson,
        background=background,
        x_bins=x_bins,
        alpha=cl,
    )

    results = ProgressBar.map(partial_func, mu_bins, multiprocess=True)

    LowerLimitAna, UpperLimitAna = zip(*results)

    LowerLimitAna = np.asarray(LowerLimitAna)
    UpperLimitAna = np.asarray(UpperLimitAna)

    fc_fix_limits(LowerLimitAna, UpperLimitAna)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(LowerLimitAna, mu_bins, ls="-", color="red")
    plt.plot(UpperLimitAna, mu_bins, ls="-", color="red")

    plt.grid(True)
    ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.xticks(range(15))
    plt.yticks(range(15))
    ax.set_xlabel(r"Measured n")
    ax.set_ylabel(r"Signal Mean $\mu$")
    plt.axis([0, 15, 0, 15])
    plt.show()
