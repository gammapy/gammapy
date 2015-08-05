"""Plot Fermi PSF."""
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from astropy.units import Quantity
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF

filename = FermiGalacticCenter.filenames()['psf']
fermi_psf = EnergyDependentTablePSF.read(filename)

plt.figure(figsize=(6, 6))

energies = Quantity([1], 'GeV')
for energy in energies:
    psf = fermi_psf.table_psf_at_energy(energy=energy)
    psf.normalize()
    kernel = psf.kernel(pixel_size=Angle(0.1, 'deg'))
    plt.imshow(np.log(kernel.value))
    #psf.plot_psf_vs_theta()

#plt.xlim(1e-2, 10)
#plt.gca().set_xscale('linear')
#plt.gca().set_yscale('linear')
plt.show()
