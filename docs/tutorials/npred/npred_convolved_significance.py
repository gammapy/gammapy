"""Runs commands to produce convolved predicted counts map in current directory."""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gammapy.stats import significance
from gammapy.image.utils import disk_correlate
from npred_general import prepare_images
from aplpy import FITSFigure

model, gtmodel, ratio, counts, header = prepare_images()

# Top hat correlation
correlation_radius = 3

correlated_gtmodel = disk_correlate(gtmodel, correlation_radius)
correlated_counts = disk_correlate(counts, correlation_radius)
correlated_model = disk_correlate(model, correlation_radius)

# Fermi significance
fermi_significance = np.nan_to_num(significance(correlated_counts, gtmodel,
                                                method='lima'))
# Gammapy significance
significance = np.nan_to_num(significance(correlated_counts, correlated_model,
                                          method='lima'))

titles = ['Gammapy Significance', 'Fermi Tools Significance']

# Plot

fig = plt.figure(figsize=(10, 5))
hdu1 = fits.ImageHDU(significance, header)
f1 = FITSFigure(hdu1, figure=fig, convention='wells', subplot=(1,2,1))
f1.set_tick_labels_font(size='x-small')
f1.tick_labels.set_xformat('ddd')
f1.tick_labels.set_yformat('ddd')
f1.show_colorscale(vmin=0, vmax=10, cmap='afmhot')
f1.add_colorbar(axis_label_text='Significance')
f1.colorbar.set_width(0.1)
f1.colorbar.set_location('right')


hdu2 = fits.ImageHDU(fermi_significance, header)
f2 = FITSFigure(hdu2, figure=fig, convention='wells', subplot=(1,2,2))
f2.set_tick_labels_font(size='x-small')
f2.tick_labels.set_xformat('ddd')
f2.hide_ytick_labels()
f2.hide_yaxis_label()
f2.show_colorscale(vmin=0, vmax=10, cmap='afmhot')
f2.add_colorbar(axis_label_text='Significance')
f2.colorbar.set_width(0.1)
f2.colorbar.set_location('right')
fig.text(0.15, 0.92,"Gammapy Significance")
fig.text(0.63, 0.92,"Fermi Tools Significance")
plt.tight_layout()
plt.show()
