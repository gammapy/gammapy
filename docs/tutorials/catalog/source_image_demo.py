"""Produces an image from 1FHL catalog point sources.
"""
import matplotlib.pyplot as plt
from gammapy.datasets import FermiGalacticCenter
from gammapy.image import make_empty_image, catalog_image
from gammapy.irf import EnergyDependentTablePSF

# Create image of defined size
reference = make_empty_image(nxpix=300, nypix=100, binsz=1)
psf_file = FermiGalacticCenter.filenames()['psf']
psf = EnergyDependentTablePSF.read(psf_file)

# Create image
image = catalog_image(reference, psf, catalog='1FHL', source_type='point',
                      total_flux='True')

# Plot
fig = plt.figure(figsize=(15, 5))
hdu = image.to_fits()[0]
wcs = WCS(reference.header)
norm = ImageNormalize(vmin=1E30, vmax=1E35, stretch=LogStretch(5E4), clip=True)
axes = fig.add_axes([0.03, 0.1, 0.9, 0.85], projection=wcs)
axes.imshow(hdu.data, origin='lower', cmap='afmhot', norm=norm)

# Axes labels
lon, lat = axes.coords
lon.set_axislabel('Galactic Longitude')
lat.set_axislabel('Galactic Latitude')
lon.set_ticks(spacing=20. * u.deg)
lat.set_ticks(spacing=10. * u.deg)

cax = fig.add_axes([0.92, 0.1, 0.01, 0.85])
cbar = fig.colorbar(axes.images[0], cax=cax, label='Flux (ph s^-1)')
cbar.solids.set_edgecolor('face')
ticks = norm.inverse(np.linspace(0, 1, 6))
cbar.set_ticks(ticks)
tick_labels = ['{:.0e}'.format(_) for _ in ticks]
cax.set_yticklabels(tick_labels)
plt.show()

