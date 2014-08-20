"""Runs commands to produce convolved predicted counts map in current directory.
"""
import numpy as np
import matplotlib.pyplot as plt
from npred_general import prepare_images

model, gtmodel, ratio, counts = prepare_images()

# Plotting

fig, axes = plt.subplots(nrows=1, ncols=3)

results = [model, gtmodel, ratio]
titles = ['Gammapy Background', 'Fermi Tools Background', 'Ratio: \n Gammapy/Fermi Tools']

vmins = [0, 0, 0.9]
vmaxs = [0.3, 0.3, 1.2]

for i in np.arange(3):
    im = axes.flat[i].imshow(results[i],
                             interpolation='nearest',
                             origin="lower", vmin=vmins[i], vmax=vmaxs[i],
                             cmap=plt.get_cmap())

    axes.flat[i].set_title(titles[i], fontsize=12)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.3, 0.025, 0.4])
fig.colorbar(im, cax=cbar_ax)
a = fig.get_axes()[0]
b = fig.get_axes()[1]
c = fig.get_axes()[2]
a.set_axis_off()
b.set_axis_off()
c.set_axis_off()
