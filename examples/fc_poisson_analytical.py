"""Analytical solution for Poisson process with
   background. Produces Fig. 7 from the Feldman Cousins
   paper."""
import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.console import ProgressBar

from gammapy.stats import (
    fc_find_acceptance_region_poisson,
    fc_fix_limits,
)

background    = 3.0

n_bins_x      = 100
step_width_mu = 0.005
mu_min        = 0
mu_max        = 50
cl            = 0.90

x_bins  = np.arange(0, n_bins_x)
mu_bins = np.linspace(mu_min, mu_max, mu_max/step_width_mu + 1, endpoint=True)

print("Generating Feldman Cousins confidence belt for " + str(len(mu_bins)) +
      " values of mu.")

UpperLimitAna = []
LowerLimitAna = []

for mu in ProgressBar(mu_bins):
    goodChoice = fc_find_acceptance_region_poisson(mu, background, x_bins, fCL)
    UpperLimitAna.append(goodChoice[0])
    LowerLimitAna.append(goodChoice[1])

fc_fix_limits(LowerLimitAna, UpperLimitAna)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(LowerLimitAna, mu_bins, ls='-', color='red')
plt.plot(UpperLimitAna, mu_bins, ls='-', color='red')

plt.grid(True)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xticks(range(15))
plt.yticks(range(15))
ax.set_xlabel(r'Measured n')
ax.set_ylabel(r'Signal Mean $\mu$')
plt.axis([0, 15, 0, 15])
plt.show()
