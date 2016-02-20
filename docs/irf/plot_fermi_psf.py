"""Plot Fermi PSF."""
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from astropy.units import Quantity
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
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
    norm = ImageNormalize(vmin=0., vmax=kernel.max(), stretch=LogStretch())
    plt.imshow(kernel.value, norm=norm)
    # psf.plot_psf_vs_theta()

# plt.xlim(1e-2, 10)
# plt.gca().set_xscale('linear')
# plt.gca().set_yscale('linear')
plt.show()
