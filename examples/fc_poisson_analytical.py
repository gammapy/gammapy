"""Analytical solution for Poisson process with
   background. Produces Fig. 7 from the Feldman Cousins
   paper."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from gammapy.stats import (
    fc_find_acceptance_region_poisson,
    fc_fix_upper_and_lower_limit,
)

fBackground  = 3.0
fStepWidthMu = 0.005
fMuMin       = 0
fMuMax       = 50
fNBinsX      = 100
fCL          = 0.90

XBins  = np.arange(0, fNBinsX)
MuBins = np.linspace(fMuMin, fMuMax, fMuMax/fStepWidthMu + 1, endpoint=True)

UpperLimitAna = []
LowerLimitAna = []

for mu in MuBins:
   goodChoice = fc_find_acceptance_region_poisson(mu, fBackground, XBins, fCL)
   UpperLimitAna.append(goodChoice[0])
   LowerLimitAna.append(goodChoice[1])

fc_fix_upper_and_lower_limit(UpperLimitAna, LowerLimitAna)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(UpperLimitAna, MuBins, ls='-',color='red')
plt.plot(LowerLimitAna, MuBins, ls='-',color='red')

plt.grid(True)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xticks(range(15))
plt.yticks(range(15))
ax.set_xlabel(r'Measured n')
ax.set_ylabel(r'Signal Mean $\mu$')
plt.axis([0, 15, 0, 15])
plt.show()
