# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Plot Fermi PSF.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from astropy.units import Quantity
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF

filename = FermiGalacticCenter.filenames()['psf']
fermi_psf = EnergyDependentTablePSF.read(filename)

energies = Quantity([1], 'GeV')
for energy in energies:
    print('0')
    psf = fermi_psf.table_psf_at_energy(energy=energy)
    print('1')
    psf.normalize()
    print('2')
    print('3', psf.eval(Angle(0.1, 'deg')))
    kernel = psf.kernel(pixel_size=Angle(0.1, 'deg'))
    print('4')
    
    print(np.nansum(kernel.value))
    plt.imshow(np.log(kernel.value))
    #psf.plot_psf_vs_theta()

#plt.xlim(1e-2, 10)
#plt.gca().set_xscale('linear')
#plt.gca().set_yscale('linear')
plt.show()
