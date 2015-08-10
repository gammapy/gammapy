"""Runs commands to produce convolved predicted counts map in current directory.
"""
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from npred_general import prepare_images

model, gtmodel, ratio, counts, header = prepare_images()

# Plotting
fig = plt.figure()
wcs = WCS(header)

axes_1 = fig.add_axes([0.18, 0.264, 0.18, 0.234], projection=wcs)
axes_1.imshow(model, origin='lower', vmin=0, vmax=0.3)

axes_2 = fig.add_axes([0.38, 0.25, 0.2, 0.26], projection=wcs)
axes_2.imshow(gtmodel, origin='lower', vmin=0, vmax=0.3)

axes_3 = fig.add_axes([0.67, 0.25, 0.2, 0.26], projection=wcs)
axes_3.imshow(ratio, origin='lower', vmin=0.9, vmax=1.1)

fig.text(0.19, 0.53, "Gammapy Background", color='black', size='9')
fig.text(0.39, 0.53, "Fermi Tools Background", color='black', size='9')
fig.text(0.68, 0.53, "Ratio: \n Gammapy/Fermi Tools", color='black', size='9')

plt.show()
