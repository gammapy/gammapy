"""Produces an image from 1FHL catalog point sources.
"""
import numpy as np
import matplotlib.pyplot as plt
from aplpy import FITSFigure
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
fig = FITSFigure(image.to_fits()[0], figsize=(15, 5))
fig.show_colorscale(interpolation='bicubic', cmap='afmhot', stretch='log', vmin=1E-12, vmax=1E-8)
fig.tick_labels.set_xformat('ddd')
fig.tick_labels.set_yformat('dd')
ticks = np.logspace(-12, -8, 5)
fig.add_colorbar(ticks=ticks, axis_label_text='Flux (ph s^-1 cm^-2 TeV^-1)')
fig.colorbar._colorbar_axes.set_yticklabels(['{:.0e}'.format(_) for _ in ticks])
plt.tight_layout()
plt.show()
