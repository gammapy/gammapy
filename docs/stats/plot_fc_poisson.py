"""Compare numerical / analytical solution for Poisson 
   process with background. Produces Fig. 7 from 
   the Feldman Cousins paper."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from gammapy.stats import (
    fc_find_confidence_interval_poisson,
    fc_construct_confidence_belt_pdfs,
    fc_get_upper_and_lower_limit,
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

DistributionsScaled = []

for mu in MuBins:
    dist = stats.poisson(mu+fBackground)
    DistributionsScaled.append(dist.pmf(XBins))

ConfidenceBelt = fc_construct_confidence_belt_pdfs(DistributionsScaled, fCL)

LowerLimitNum, UpperLimitNum = fc_get_upper_and_lower_limit(MuBins, XBins, ConfidenceBelt)

fc_fix_upper_and_lower_limit(UpperLimitNum, LowerLimitNum)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(UpperLimitNum, MuBins, marker='.', ls='-',color='red')
plt.plot(LowerLimitNum, MuBins, marker='.', ls='-',color='red')

plt.grid(True)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xticks(range(15))
plt.yticks(range(15))
ax.set_xlabel(r'Measured n')
ax.set_ylabel(r'Signal Mean $\mu$')
plt.axis([0, 15, 0, 15])

plt.savefig("fc_poisson_numerical.png")
plt.close()

UpperLimitAna = []
LowerLimitAna = []

for mu in mu_bins:
   goodChoice = upper_limits.FindConfidenceIntervalPoisson(mu, background, x_bins, fCL)
   UpperLimitAna.append(goodChoice[0])
   LowerLimitAna.append(goodChoice[1])

fc_find_confidence_interval_poisson(UpperLimitAna, LowerLimitAna)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(UpperLimitAna, MuBins, marker='.', ls='-',color='red')
plt.plot(LowerLimitAna, MuBins, marker='.', ls='-',color='red')

plt.grid(True)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xticks(range(15))
plt.yticks(range(15))
ax.set_xlabel(r'Measured n')
ax.set_ylabel(r'Signal Mean $\mu$')
plt.axis([0, 15, 0, 15])

plt.savefig("fc_poisson_analytical.png")
plt.close()
