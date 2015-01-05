"""Runs commands to produce convolved predicted counts map in current directory.
"""
import matplotlib.pyplot as plt
from astropy.io import fits
from npred_general import prepare_images
from aplpy import FITSFigure

model, gtmodel, ratio, counts, header = prepare_images()

# Plotting

fig = plt.figure()
hdu1 = fits.ImageHDU(model, header)
f1 = FITSFigure(hdu1, figure=fig, convention='wells', subplot=[0.18, 0.264, 0.18, 0.234])
f1.tick_labels.set_font(size='x-small')
f1.tick_labels.set_xformat('ddd')
f1.tick_labels.set_yformat('ddd')
f1.axis_labels.hide_x()
f1.show_colorscale(vmin=0, vmax=0.3)

hdu2 = fits.ImageHDU(gtmodel, header)
f2 = FITSFigure(hdu2, figure=fig, convention='wells', subplot=[0.38, 0.25, 0.2, 0.26])
f2.tick_labels.set_font(size='x-small')
f2.tick_labels.set_xformat('ddd')
f2.tick_labels.hide_y()
f2.axis_labels.hide_y()
f2.show_colorscale(vmin=0, vmax=0.3)
f2.add_colorbar()
f2.colorbar.set_width(0.1)
f2.colorbar.set_location('right')

hdu3 = fits.ImageHDU(ratio, header)
f3 = FITSFigure(hdu3, figure=fig, convention='wells', subplot=[0.67, 0.25, 0.2, 0.26])
f3.tick_labels.set_font(size='x-small')
f3.tick_labels.set_xformat('ddd')
f3.tick_labels.hide_y()
f3.axis_labels.hide()
f3.show_colorscale(vmin=0.9, vmax=1.1)
f3.add_colorbar()
f3.colorbar.set_width(0.1)
f3.colorbar.set_location('right')

fig.text(0.19, 0.53, "Gammapy Background", color='black', size='9')
fig.text(0.39, 0.53, "Fermi Tools Background", color='black', size='9')
fig.text(0.68, 0.53, "Ratio: \n Gammapy/Fermi Tools", color='black', size='9')

fig.canvas.draw()
