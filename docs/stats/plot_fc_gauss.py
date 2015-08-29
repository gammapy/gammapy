"""Compare numerical / analytical solution for Gaussian 
   with a boundary at the origin. Produces Fig. 10 from 
   the Feldman Cousins paper."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from gammapy.stats import (
    fc_find_confidence_interval_gauss,
    fc_construct_confidence_belt_pdfs,
    fc_get_upper_and_lower_limit,
    fc_fix_upper_and_lower_limit,
)

fSigma  = 1
fMuMin  = 0
fMuMax  = 8
fMuStep = 100
fNSigma = 10
fNStep  = 1000
fCL     = 0.90

XBins  = np.linspace(-fNSigma*fSigma, fNSigma*fSigma, fNStep, endpoint=True)
MuBins = np.linspace(fMuMin, fMuMax, fMuStep, endpoint=True)

DistributionsScaled = []

for mu in MuBins:
    dist = stats.norm(loc=mu, scale=sigma)
    dist = dist.pdf(XBins)
    DistributionsScaled.append(dist/sum(dist))

ConfidenceBelt = fc_construct_confidence_belt_pdf(DistributionsScaled, fCL)

LowerLimitNum, UpperLimitNum = fc_get_upper_and_lower_limit(MuBins, XBins, ConfidenceBelt)

fc_fix_upper_and_lower_limit(UpperLimitNum, LowerLimitNum)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(UpperLimitNum, MuBins, marker='.', ls='-',color='red')
plt.plot(LowerLimitNum, MuBins, marker='.', ls='-',color='red')

plt.grid(True)
ax.xaxis.set_ticks(np.arange(-10, 10, 1))
ax.xaxis.set_ticks(np.arange(-10, 10, 0.2), True)
ax.yaxis.set_ticks(np.arange(0, 8, 0.2), True)
ax.set_xlabel(r'Measured Mean x')
ax.set_ylabel(r'Mean $\mu$')
plt.axis([-2, 4, 0, 6])

plt.savefig("fc_gauss_numerical.png")
plt.close()

UpperLimitAna = []
LowerLimitAna = []

for mu in MuBins:
   goodChoice = fc_find_confidence_interval_gauss(mu, fSigma, XBins, fCL)
   UpperLimitAna.append(goodChoice[0])
   LowerLimitAna.append(goodChoice[1])

fc_fix_upper_and_lower_limit(UpperLimitAna, LowerLimitAna)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(UpperLimitAna, MuBins, marker='.', ls='-',color='red')
plt.plot(LowerLimitAna, MuBins, marker='.', ls='-',color='red')

plt.grid(True)
ax.xaxis.set_ticks(np.arange(-10, 10, 1))
ax.xaxis.set_ticks(np.arange(-10, 10, 0.2), True)
ax.yaxis.set_ticks(np.arange(0, 8, 0.2), True)
ax.set_xlabel(r'Measured Mean x')
ax.set_ylabel(r'Mean $\mu$')
plt.axis([-2, 4, 0, 6])

plt.savefig("fc_gauss_analytical.png")
plt.close()
