"""
Compute numerical solution for Poisson with background.

Produces Fig. 7 from the Feldman & Cousins paper.
"""
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from gammapy.stats import (
    fc_construct_acceptance_intervals_pdfs,
    fc_fix_limits,
    fc_get_limits,
)

background = 3.0

n_bins_x = 100
step_width_mu = 0.005
mu_min = 0
mu_max = 50
cl = 0.90

x_bins = np.arange(0, n_bins_x)
mu_bins = np.linspace(mu_min, mu_max, mu_max / step_width_mu + 1, endpoint=True)

matrix = [poisson(mu + background).pmf(x_bins) for mu in mu_bins]

acceptance_intervals = fc_construct_acceptance_intervals_pdfs(matrix, cl)

LowerLimitNum, UpperLimitNum, _ = fc_get_limits(mu_bins, x_bins, acceptance_intervals)

fc_fix_limits(LowerLimitNum, UpperLimitNum)

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(UpperLimitNum, mu_bins, ls="-", color="red")
plt.plot(LowerLimitNum, mu_bins, ls="-", color="red")

plt.grid(True)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xticks(range(15))
plt.yticks(range(15))
ax.set_xlabel(r"Measured n")
ax.set_ylabel(r"Signal Mean $\mu$")
plt.axis([0, 15, 0, 15])
plt.show()
