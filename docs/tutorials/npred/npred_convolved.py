"""Runs commands to produce convolved predicted counts map in current directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from npred_general import prepare_images
from aplpy import FITSFigure

model, gtmodel, ratio, counts, header = prepare_images()

# Plotting

titles = ['Gammapy Background', 'Fermi Tools Background', 'Ratio: \n Gammapy/Fermi Tools']

# Plot

fig = plt.figure()
hdu1 = fits.ImageHDU(model, header)
f1 = FITSFigure(hdu1, figure=fig, convention='wells', subplot=[0.18,0.25,0.2,0.26])
f1.set_tick_labels_font(size='x-small')
f1.tick_labels.set_xformat('ddd')
f1.tick_labels.set_yformat('ddd')
f1.hide_xaxis_label()
f1.show_colorscale(vmin=0, vmax=0.3)

hdu2 = fits.ImageHDU(gtmodel, header)
f2 = FITSFigure(hdu2, figure=fig, convention='wells', subplot=[0.41,0.25,0.2,0.26])
f2.set_tick_labels_font(size='x-small')
f2.tick_labels.set_xformat('ddd')
f2.hide_ytick_labels()
f2.hide_yaxis_label()
f2.show_colorscale(vmin=0, vmax=0.3)

hdu3 = fits.ImageHDU(ratio, header)
f3 = FITSFigure(hdu2, figure=fig, convention='wells', subplot=[0.62,0.25,0.2,0.26])
f3.set_tick_labels_font(size='x-small')
f3.tick_labels.set_xformat('ddd')
f3.hide_ytick_labels()
f3.hide_yaxis_label()
f3.hide_xaxis_label()
f3.show_colorscale(vmin=0, vmax=2)

fig.canvas.draw()
