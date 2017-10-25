"""Plot Fermi PSF."""
import matplotlib.pyplot as plt
from astropy import units as u
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF
from gammapy.image import SkyImage

filename = FermiGalacticCenter.filenames()['psf']
fermi_psf = EnergyDependentTablePSF.read(filename)

fig = plt.figure(figsize=(6, 5))

# Compute a PSF kernel image
# TODO: change this example after introducing PSF kernel class
# (using SkyImage this way for kernels is weird)
psf_image = SkyImage.empty()
energy = 1 * u.GeV
psf = fermi_psf.table_psf_at_energy(energy=energy)
psf_image.data = psf.kernel(psf_image, rad_max=1 * u.deg).value
psf_image.plot(fig=fig, add_cbar=True)

plt.show()
