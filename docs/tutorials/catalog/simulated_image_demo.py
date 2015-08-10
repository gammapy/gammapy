"""Simulates a galaxy of point sources and produces an image.
"""
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

from astropy.wcs import WCS
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from gammapy.astro import population
from gammapy.datasets import FermiGalacticCenter
from gammapy.image import make_empty_image, catalog_image
from gammapy.irf import EnergyDependentTablePSF
from gammapy.utils.random import sample_powerlaw

np.random.seed(0)

# Create image of defined size
reference = make_empty_image(nxpix=300, nypix=100, binsz=1)

psf_file = FermiGalacticCenter.filenames()['psf']
psf = EnergyDependentTablePSF.read(psf_file)

# Simulation Parameters

# source density at the sun (sources kpc^-1)
rho_sun = 3
# number of sources
n_sources = int(5e2)
# Spatial distribution using Lorimer (2006) model
rad_dis = 'L06'
# Velocity dispersion
vel_dis = 'F06B'
# Includes spiral arms
spiralarms = True
# Creates table
table = population.make_base_catalog_galactic(n_sources=n_sources, rad_dis=rad_dis,
                                              vel_dis=vel_dis, max_age=1e6,
                                              spiralarms=spiralarms)

# Minimum source luminosity (ph s^-1)
luminosity_min = 4e34
# Maximum source luminosity (ph s^-1)
luminosity_max = 4e37
# Luminosity function differential power-law index
luminosity_index = 1.5

# Assigns luminosities to sources
luminosity = sample_powerlaw(luminosity_min, luminosity_max, luminosity_index,
                             n_sources)
table['luminosity'] = luminosity

# Adds parameters to table: distance, glon, glat, flux, angular_extension
table = population.add_observed_parameters(table)
table.meta['Energy Bins'] = np.array([10, 500]) * u.GeV
# Create image
image = catalog_image(reference, psf, catalog='simulation', source_type='point',
                      total_flux=True, sim_table=table)

# Plot
fig = plt.figure(figsize=(15, 5))
hdu = image.to_fits()[0]
wcs = WCS(hdu.header)
norm = ImageNormalize(vmin=1E30, vmax=1E35, stretch=LogStretch(5E4), clip=True)
axes = fig.add_axes([0.03, 0.1, 0.9, 0.85], projection=wcs)
axes.imshow(hdu.data, origin='lower', cmap='afmhot', norm=norm)

# Axes labels
lon, lat = axes.coords
lon.set_axislabel('Galactic Longitude')
lat.set_axislabel('Galactic Latitude')
lon.set_ticks(spacing=20. * u.deg)
lat.set_ticks(spacing=10. * u.deg)

cax = fig.add_axes([0.91, 0.1, 0.02, 0.85])
cbar = fig.colorbar(axes.images[0], cax=cax, label='Flux (ph s^-1)')
cbar.solids.set_edgecolor('face')
ticks = norm.inverse(np.linspace(0, 1, 6))
cbar.set_ticks(ticks)
tick_labels = ['{:.0e}'.format(_) for _ in ticks]
cax.set_yticklabels(tick_labels)
plt.show()
