"""Plot Fermi PSF."""
import matplotlib.pyplot as plt
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF
from gammapy.maps import WcsGeom
from gammapy.cube import PSFKernel

filename = FermiGalacticCenter.filenames()['psf']
fermi_psf = EnergyDependentTablePSF.read(filename)

psf = fermi_psf.table_psf_at_energy(energy='1 GeV')
geom = WcsGeom.create(npix=100, binsz=0.01)
kernel = PSFKernel.from_table_psf(psf, geom)

plt.imshow(kernel.data)
plt.colorbar()
plt.show()
