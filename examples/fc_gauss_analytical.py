"""Analytical solution for Gaussian with a boundary at
   the origin. Produces Fig. 10 from the Feldman Cousins
   paper."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from gammapy.stats import (
    fc_find_acceptance_region_gauss,
    fc_fix_upper_and_lower_limit,
)

fSigma       = 1
fStepWidthMu = 0.005
fMuMin       = 0
fMuMax       = 8
fNSigma      = 10
fNStep       = 1000
fCL          = 0.90

XBins  = np.linspace(-fNSigma*fSigma, fNSigma*fSigma, fNStep, endpoint=True)
MuBins = np.linspace(fMuMin, fMuMax, fMuMax/fStepWidthMu + 1, endpoint=True)

matrix = [dist/sum(dist) for dist in (stats.norm(loc=mu, scale=fSigma).pdf(XBins) for mu in MuBins)]

UpperLimitAna = []
LowerLimitAna = []

for mu in MuBins:
   goodChoice = fc_find_acceptance_region_gauss(mu, fSigma, XBins, fCL)
   UpperLimitAna.append(goodChoice[0])
   LowerLimitAna.append(goodChoice[1])

fc_fix_upper_and_lower_limit(UpperLimitAna, LowerLimitAna)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(UpperLimitAna, MuBins, ls='-',color='red')
plt.plot(LowerLimitAna, MuBins, ls='-',color='red')

plt.grid(True)
ax.xaxis.set_ticks(np.arange(-10, 10, 1))
ax.xaxis.set_ticks(np.arange(-10, 10, 0.2), True)
ax.yaxis.set_ticks(np.arange(0, 8, 0.2), True)
ax.set_xlabel(r'Measured Mean x')
ax.set_ylabel(r'Mean $\mu$')
plt.axis([-2, 4, 0, 6])
plt.show()
