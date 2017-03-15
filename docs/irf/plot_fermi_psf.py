"""Plot Fermi PSF."""
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.visualization import simple_norm
from gammapy.datasets import FermiGalacticCenter
from gammapy.irf import EnergyDependentTablePSF
from gammapy.image import SkyImage

filename = FermiGalacticCenter.filenames()['psf']
fermi_psf = EnergyDependentTablePSF.read(filename)

fig = plt.figure(figsize=(6, 5))

# Create an empty sky image to show the PSF
image_psf = SkyImage.empty()

energies = [1] * u.GeV
for energy in energies:
    psf = fermi_psf.table_psf_at_energy(energy=energy)
    image_psf.data = psf.kernel(image_psf)
    norm = simple_norm(image_psf.data, stretch='log')
    image_psf.plot(fig=fig, add_cbar=True, norm=norm)

plt.show()
