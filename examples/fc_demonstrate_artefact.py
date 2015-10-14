"""Demonstrate the artefact that can arise if
   fc_fix_upper_and_lower_limit is not used."""
import numpy as np

from gammapy.stats import fc_find_acceptance_region_poisson

fBackground  = 3.5
fNBinsX      = 100
fCL          = 0.90

XBins = np.arange(0, fNBinsX)

for mu in [0.745, 0.750, 0.755, 1.030, 1.035, 1.040, 1.045, 1.050, 1.055, 1.060, 1.065]:

    iMax, iMin = fc_find_acceptance_region_poisson(mu, fBackground, XBins, fCL)

    print("Mu:               " + str(mu))
    print("Acceptance Start: " + str(iMax))
    print("Acceptance Stop:  " + str(iMin))
    print("----")