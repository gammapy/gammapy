"""Runs commands to produce convolved predicted counts map in current directory.
"""
import matplotlib.pyplot as plt
from astropy.io import fits
from aplpy import FITSFigure
from npred_general import prepare_images

model, gtmodel, ratio, counts, header = prepare_images()

# Plotting
fig = plt.figure(figsize=(15, 5))

image1 = fits.ImageHDU(data=model, header=header)
f1 = FITSFigure(image1, figure=fig, subplot=(1, 3, 1), convention='wells')
f1.show_colorscale(vmin=0, vmax=0.3, cmap='afmhot')
f1.tick_labels.set_xformat('ddd')
f1.tick_labels.set_yformat('dd')

image2 = fits.ImageHDU(data=gtmodel, header=header)
f2 = FITSFigure(image2, figure=fig, subplot=(1, 3, 2), convention='wells')
f2.show_colorscale(vmin=0, vmax=0.3, cmap='afmhot')
f2.tick_labels.set_xformat('ddd')
f2.tick_labels.set_yformat('dd')

image3 = fits.ImageHDU(data=ratio, header=header)
f3 = FITSFigure(image3, figure=fig, subplot=(1, 3, 3), convention='wells')
f3.show_colorscale(vmin=0.95, vmax=1.05, cmap='RdBu')
f3.tick_labels.set_xformat('ddd')
f3.tick_labels.set_yformat('dd')
f3.add_colorbar(ticks=[0.95, 0.975, 1, 1.025, 1.05])

fig.text(0.12, 0.95, "Gammapy Background")
fig.text(0.45, 0.95, "Fermi Tools Background")
fig.text(0.75, 0.95, "Ratio: Gammapy/Fermi Tools")
plt.tight_layout()
plt.show()
