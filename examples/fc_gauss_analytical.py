"""Analytical solution for Gaussian with a boundary at
   the origin. Produces Fig. 10 from the Feldman Cousins
   paper."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from astropy.utils.console import ProgressBar

from gammapy.stats import (
    fc_find_acceptance_region_gauss,
    fc_fix_limits,
)

sigma         = 1
n_sigma       = 10
n_bins_x      = 1000
step_width_mu = 0.005
mu_min        = 0
mu_max        = 8
cl            = 0.90

x_bins  = np.linspace(-n_sigma*sigma, n_sigma*sigma, n_bins_x, endpoint=True)
mu_bins = np.linspace(mu_min, mu_max, mu_max/step_width_mu + 1, endpoint=True)

print("Generating Feldman Cousins confidence belt for " + str(len(mu_bins)) +
      " values of mu.")

UpperLimitAna = []
LowerLimitAna = []

for mu in ProgressBar(mu_bins):
    goodChoice = fc_find_acceptance_region_gauss(mu, sigma, x_bins, cl)
    UpperLimitAna.append(goodChoice[0])
    LowerLimitAna.append(goodChoice[1])

fc_fix_limits(LowerLimitAna, UpperLimitAna)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(LowerLimitAna, mu_bins, ls='-', color='red')
plt.plot(UpperLimitAna, mu_bins, ls='-', color='red')

plt.grid(True)
ax.xaxis.set_ticks(np.arange(-10, 10, 1))
ax.xaxis.set_ticks(np.arange(-10, 10, 0.2), True)
ax.yaxis.set_ticks(np.arange(0, 8, 0.2), True)
ax.set_xlabel(r'Measured Mean x')
ax.set_ylabel(r'Mean $\mu$')
plt.axis([-2, 4, 0, 6])
plt.show()
