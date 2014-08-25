# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Plot significicance image with HESS- and MILAGRO-colormap.
"""
import numpy as np
import matplotlib.pyplot as plt
from gammapy.datasets import load_poisson_stats_image
from gammapy.image import disk_correlate
from gammapy.stats import significance
from gammapy.image import colormap_hess, colormap_milagro

# Compute an example significance image
counts = load_poisson_stats_image()
counts = disk_correlate(counts, radius=5, mode='reflect')
background = np.median(counts) * np.ones_like(counts)
image = significance(counts, background)

# Plot with the HESS and Milagro colormap
vmin, vmax, vtransition = -5, 15, 5
plt.figure(figsize=(8, 4))

plt.subplot(121)
cmap = colormap_hess(vtransition=vtransition, vmin=vmin, vmax=vmax)
plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
plt.axis('off')
plt.colorbar()
plt.title('HESS-style colormap')

plt.subplot(122)
cmap = colormap_milagro(vtransition=vtransition, vmin=vmin, vmax=vmax)
plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
plt.axis('off')
plt.colorbar()
plt.title('MILAGRO-style colormap')

plt.show()
