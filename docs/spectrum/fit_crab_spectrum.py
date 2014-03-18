# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Fit and plot Crab nebula spectrum."""
import matplotlib.pyplot as plt
from astropy.modeling.models import PowerLaw1D
from astropy.modeling.fitting import NonLinearLSQFitter
from gammapy.datasets import tev_spectrum

data = tev_spectrum('crab')
energy, flux, flux_err  = data['energy'], data['flux'], data['flux_err']

model = PowerLaw1D(4e-11, 1, 2.6)
model.x_0.fixed = True
fitter = NonLinearLSQFitter()

model = fitter(model, energy, flux, weights=(1. / flux_err))
print(model)

plt.errorbar(energy, energy ** 2 * flux, energy ** 2 * flux_err, fmt='o', label='HESS measurement')
plt.errorbar(energy, energy ** 2 * model(energy), fmt='r-', label='Powerlaw fit')
plt.loglog()
plt.xlabel('Energy (TeV)')
plt.ylabel('E^2 * dN / dE (TeV cm^-2 s^-1)')
plt.legend()
plt.savefig('fit_crab_spectrum.png')
