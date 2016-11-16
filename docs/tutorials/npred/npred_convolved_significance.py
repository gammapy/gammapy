"""Runs commands to produce convolved predicted counts map in current directory."""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import Tophat2DKernel
from scipy.ndimage import convolve
from gammapy.stats import significance
from aplpy import FITSFigure
from npred_general import prepare_images

model, gtmodel, ratio, counts, header = prepare_images()

# Top hat correlation
tophat = Tophat2DKernel(3)
tophat.normalize('peak')

correlated_gtmodel = convolve(gtmodel, tophat.array)
correlated_counts = convolve(counts, tophat.array)
correlated_model = convolve(model, tophat.array)

# Fermi significance
fermi_significance = np.nan_to_num(significance(correlated_counts, correlated_gtmodel,
                                                method='lima'))
# Gammapy significance
significance = np.nan_to_num(significance(correlated_counts, correlated_model,
                                          method='lima'))

titles = ['Gammapy Significance', 'Fermi Tools Significance']

# Plot

fig = plt.figure(figsize=(10, 5))
hdu1 = fits.ImageHDU(significance, header)
f1 = FITSFigure(hdu1, figure=fig, convention='wells', subplot=(1, 2, 1))
f1.set_tick_labels_font(size='x-small')
f1.tick_labels.set_xformat('ddd')
f1.tick_labels.set_yformat('ddd')
f1.show_colorscale(vmin=0, vmax=20, cmap='afmhot', stretch='sqrt')
f1.add_colorbar(axis_label_text='Significance')
f1.colorbar.set_width(0.1)
f1.colorbar.set_location('right')

hdu2 = fits.ImageHDU(fermi_significance, header)
f2 = FITSFigure(hdu2, figure=fig, convention='wells', subplot=(1, 2, 2))
f2.set_tick_labels_font(size='x-small')
f2.tick_labels.set_xformat('ddd')
f2.hide_ytick_labels()
f2.hide_yaxis_label()
f2.show_colorscale(vmin=0, vmax=20, cmap='afmhot', stretch='sqrt')
f2.add_colorbar(axis_label_text='Significance')
f2.colorbar.set_width(0.1)
f2.colorbar.set_location('right')
fig.text(0.15, 0.92, "Gammapy Significance")
fig.text(0.63, 0.92, "Fermi Tools Significance")
plt.tight_layout()
plt.show()
