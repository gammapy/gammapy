"""Plot significance image with HESS and MILAGRO colormap.
"""
import numpy as np
import matplotlib.pyplot as plt
from gammapy.image import colormap_hess, colormap_milagro
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LinearStretch
from gammapy.image import SkyImage

filename = '$GAMMAPY_EXTRA/test_datasets/unbundled/poisson_stats_image/expected_ts_0.000.fits.gz'
image = SkyImage.read(filename, hdu='SQRT_TS')

# Plot with the HESS and Milagro colormap
vmin, vmax, vtransition = -5, 15, 5
plt.figure(figsize=(12, 6))

normalize = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
transition = normalize(vtransition)

plt.subplot(121)
cmap = colormap_hess(transition=transition)
plt.imshow(image, cmap=cmap, norm=normalize)
plt.axis('off')
plt.colorbar(shrink=0.7)
plt.title('HESS-style colormap')

plt.subplot(122)
cmap = colormap_milagro(transition=transition)
plt.imshow(image, cmap=cmap, norm=normalize)
plt.axis('off')
plt.colorbar(shrink=0.7)
plt.title('MILAGRO-style colormap')

plt.tight_layout()
plt.show()
