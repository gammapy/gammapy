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
reference = SkyImage.empty(nxpix=1000, nypix=200, binsz=0.2).to_image_hdu()

psf_file = FermiGalacticCenter.filenames()['psf']
psf = EnergyDependentTablePSF.read(psf_file)

# Simulation Parameters
n_sources = int(5e2)

table = population.make_base_catalog_galactic(
    n_sources=n_sources,
    rad_dis='L06',
    vel_dis='F06B',
    max_age=1e5 * u.yr,
    spiralarms=True,
    random_state=0,
)

# Minimum source luminosity (s^-1)
luminosity_min = 4e34
# Maximum source luminosity (s^-1)
luminosity_max = 4e37
# Luminosity function differential power-law index
luminosity_index = 1.5

# Assigns luminosities to sources
luminosity = sample_powerlaw(luminosity_min, luminosity_max, luminosity_index,
                             n_sources, random_state=0)
table['luminosity'] = luminosity

# Adds parameters to table: distance, glon, glat, flux, angular_extension
table = population.add_observed_parameters(table)
table.meta['Energy Bins'] = [10, 500] * u.GeV
# Create image
image = catalog_image(reference, psf, catalog='simulation', source_type='point',
                      total_flux=True, sim_table=table)

# Plot
fig = FITSFigure(image.to_fits(format='fermi-background')[0], figsize=(10, 3))
fig.show_colorscale(interpolation='bicubic', cmap='afmhot', stretch='log', vmin=1E30, vmax=1E35)
fig.tick_labels.set_xformat('ddd')
fig.tick_labels.set_yformat('dd')
ticks = np.logspace(30, 35, 6)
fig.add_colorbar(ticks=ticks, axis_label_text='Flux (s^-1)')
fig.colorbar._colorbar_axes.set_yticklabels(['{:.0e}'.format(_) for _ in ticks])
plt.tight_layout()
plt.show()
