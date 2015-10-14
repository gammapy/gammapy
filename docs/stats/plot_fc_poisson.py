"""Compute numerical solution for Poisson with
   background. Produces Fig. 7 from the Feldman
   Cousins paper."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

from gammapy.stats import (
    fc_construct_acceptance_intervals_pdfs,
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

Matrix = [poisson(mu+fBackground).pmf(XBins) for mu in MuBins]

AcceptanceIntervals = fc_construct_acceptance_intervals_pdfs(Matrix, fCL)

UpperLimitNum, LowerLimitNum, _ = fc_get_upper_and_lower_limit(MuBins, XBins, AcceptanceIntervals)

fc_fix_upper_and_lower_limit(UpperLimitNum, LowerLimitNum)

fig = plt.figure()
ax  = fig.add_subplot(111)

plt.plot(UpperLimitNum, MuBins, ls='-',color='red')
plt.plot(LowerLimitNum, MuBins, ls='-',color='red')

plt.grid(True)
ax.yaxis.set_label_coords(-0.08, 0.5)
plt.xticks(range(15))
plt.yticks(range(15))
ax.set_xlabel(r'Measured n')
ax.set_ylabel(r'Signal Mean $\mu$')
plt.axis([0, 15, 0, 15])
plt.show()
