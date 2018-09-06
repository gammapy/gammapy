"""Plot significance image with HESS and MILAGRO colormap.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LinearStretch
from gammapy.image import colormap_hess, colormap_milagro
from gammapy.maps import Map

filename = "$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/expected_ts_0.000.fits.gz"
image = Map.read(filename, hdu="SQRT_TS")

# Plot with the HESS and Milagro colormap
vmin, vmax, vtransition = -5, 15, 5
fig = plt.figure(figsize=(15.5, 6))

normalize = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
transition = normalize(vtransition)

ax = fig.add_subplot(121, projection=image.geom.wcs)
cmap = colormap_hess(transition=transition)
image.plot(ax=ax, cmap=cmap, norm=normalize, add_cbar=True)
plt.title("HESS-style colormap")

ax = fig.add_subplot(122, projection=image.geom.wcs)
cmap = colormap_milagro(transition=transition)
image.plot(ax=ax, cmap=cmap, norm=normalize, add_cbar=True)
plt.title("MILAGRO-style colormap")

plt.tight_layout()
plt.show()
