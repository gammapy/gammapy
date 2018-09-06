"""
Compute numerical solution for Gaussian with a boundary at the origin.

Produces Fig. 10 from the Feldman & Cousins paper.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from gammapy.stats import (
    fc_construct_acceptance_intervals_pdfs,
    fc_get_limits,
    fc_fix_limits,
)

sigma = 1
n_sigma = 10
n_bins_x = 1000
step_width_mu = 0.005
mu_min = 0
mu_max = 8
cl = 0.90

x_bins = np.linspace(-n_sigma * sigma, n_sigma * sigma, n_bins_x, endpoint=True)
mu_bins = np.linspace(mu_min, mu_max, mu_max / step_width_mu + 1, endpoint=True)

matrix = [
    dist / sum(dist)
    for dist in (norm(loc=mu, scale=sigma).pdf(x_bins) for mu in mu_bins)
]

acceptance_intervals = fc_construct_acceptance_intervals_pdfs(matrix, cl)

LowerLimitNum, UpperLimitNum, _ = fc_get_limits(mu_bins, x_bins, acceptance_intervals)

fc_fix_limits(LowerLimitNum, UpperLimitNum)

fig = plt.figure()
ax = fig.add_subplot(111)

plt.plot(UpperLimitNum, mu_bins, ls="-", color="red")
plt.plot(LowerLimitNum, mu_bins, ls="-", color="red")

plt.grid(True)
ax.xaxis.set_ticks(np.arange(-10, 10, 1))
ax.xaxis.set_ticks(np.arange(-10, 10, 0.2), True)
ax.yaxis.set_ticks(np.arange(0, 8, 0.2), True)
ax.set_xlabel(r"Measured Mean x")
ax.set_ylabel(r"Mean $\mu$")
plt.axis([-2, 4, 0, 6])
plt.show()
