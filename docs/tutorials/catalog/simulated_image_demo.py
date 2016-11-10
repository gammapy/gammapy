"""Simulates a galaxy of point sources and produces an image.
"""
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from aplpy import FITSFigure
from gammapy.astro import population
from gammapy.datasets import FermiGalacticCenter
from gammapy.image import SkyImage, catalog_image
from gammapy.irf import EnergyDependentTablePSF
from gammapy.utils.random import sample_powerlaw

# Create image of defined size
reference = SkyImage.empty(nxpix=300, nypix=100, binsz=1).to_image_hdu()

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
                                              spiralarms=spiralarms, random_state=0)

# Minimum source luminosity (ph s^-1)
luminosity_min = 4e34
# Maximum source luminosity (ph s^-1)
luminosity_max = 4e37
# Luminosity function differential power-law index
luminosity_index = 1.5

# Assigns luminosities to sources
luminosity = sample_powerlaw(luminosity_min, luminosity_max, luminosity_index,
                             n_sources, random_state=0)
table['luminosity'] = luminosity

# Adds parameters to table: distance, glon, glat, flux, angular_extension
table = population.add_observed_parameters(table)
table.meta['Energy Bins'] = np.array([10, 500]) * u.GeV
# Create image
image = catalog_image(reference, psf, catalog='simulation', source_type='point',
                      total_flux=True, sim_table=table)

# Plot
fig = FITSFigure(image.to_fits(format='fermi-background')[0], figsize=(15, 5))
fig.show_colorscale(interpolation='bicubic', cmap='afmhot', stretch='log', vmin=1E30, vmax=1E35)
fig.tick_labels.set_xformat('ddd')
fig.tick_labels.set_yformat('dd')
ticks = np.logspace(30, 35, 6)
fig.add_colorbar(ticks=ticks, axis_label_text='Flux (ph s^-1)')
fig.colorbar._colorbar_axes.set_yticklabels(['{:.0e}'.format(_) for _ in ticks])
plt.tight_layout()
plt.show()
