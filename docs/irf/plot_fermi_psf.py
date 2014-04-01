# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Plot Fermi PSF.
"""
import matplotlib.pyplot as plt
from astropy.units import Quantity
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF

filename = FermiGalacticCenter.filenames()['psf']
psf = EnergyDependentTablePSF.read(filename)

energy = Quantity(1, 'GeV')
psf.table_psf(energy=energy).plot_psf_vs_theta()
plt.xlim(1e-2, 10)
plt.show()
