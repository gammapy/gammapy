"""Plot significance image with HESS and MILAGRO colormap.
"""
import numpy as np
import matplotlib.pyplot as plt
from gammapy.datasets import load_poisson_stats_image
from gammapy.image import disk_correlate
from gammapy.stats import significance
from gammapy.image import colormap_hess, colormap_milagro
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LinearStretch

# Compute an example significance image
counts = load_poisson_stats_image()
counts = disk_correlate(counts, radius=5, mode='reflect')
background = np.median(counts) * np.ones_like(counts)
image = significance(counts, background)

# Plot with the HESS and Milagro colormap
vmin, vmax, vtransition = -5, 15, 5
plt.figure(figsize=(8, 4))

normalize = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
transition = normalize(vtransition)

plt.subplot(121)
cmap = colormap_hess(transition=transition)
plt.imshow(image, cmap=cmap, norm=normalize)
plt.axis('off')
plt.colorbar()
plt.title('HESS-style colormap')

plt.subplot(122)
cmap = colormap_milagro(transition=transition)
plt.imshow(image, cmap=cmap, norm=normalize)
plt.axis('off')
plt.colorbar()
plt.title('MILAGRO-style colormap')

plt.show()
