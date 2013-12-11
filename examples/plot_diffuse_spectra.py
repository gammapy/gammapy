"""Plot spectra of cosmic ray protons, electrons and diffuse gammas.

"""
import numpy as np
from astropy.units import Quantity
import matplotlib.pyplot as plt
from gammapy.data import electron_spectrum
from gammapy.spectrum.cosmic_ray import cosmic_ray_flux
from gammapy.spectrum.diffuse import diffuse_gamma_ray_flux

GAMMA = 3.
UNIT = 'm^-2 s^-1 sr^-1 TeV^-1'

# Electron spectrum measurements
electron_spectra = ['Fermi', 'HESS', 'HESS low energy']
for name in electron_spectra:
    data = electron_spectrum(name)
    x = data['energy']
    y = (x ** GAMMA) * data['flux']
    y_err = (x ** GAMMA) * data['flux_err']
    label = 'Electrons {0}'.format(name)
    plt.errorbar(x, y, y_err, fmt='o', label=label)

# Isotropic gamma-ray spectrum measurements
# TODO

# Electron spectrum model
energy = Quantity(np.logspace(-2, 2), 'TeV')
flux = cosmic_ray_flux(energy, particle='electron')
x = energy
y = (energy ** GAMMA) * flux.to(UNIT)
plt.plot(x.value, y.value, lw=3, label='Electron spectrum model')

# Diffuse gamma-ray model spectra


# Make the plot nice
plt.loglog()
plt.xlabel('Energy (TeV)')
plt.ylabel('E^({0}) * dN / dE (TeV^({0}) m^-2 s^-1 sr^-1 TeV^-1)'.format(GAMMA))
plt.xlim(8e-3, 100)
plt.legend()
plt.savefig('diffuse_spectra.png')

