"""Estimate a diffuse emission model from Fermi LAT data.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits
from astropy.convolution import Tophat2DKernel
from gammapy.datasets import FermiGalacticCenter
from gammapy.image import SkyImageList
from gammapy.detect import KernelBackgroundEstimator

# Parameters
CORRELATION_RADIUS = 10  # Pixels
SIGNIFICANCE_THRESHOLD = 5  # Sigma
MASK_DILATION_RADIUS = 10  # Pixels

# Load example images.
filename = ('$GAMMAPY_EXTRA/datasets/source_diffuse_separation/'
            'galactic_simulations/fermi_counts.fits')
images = SkyImageList.read(filename)

source_kernel = Tophat2DKernel(CORRELATION_RADIUS).array
background_kernel = np.ones((10, 100))

kbe = KernelBackgroundEstimator(
    kernel_src=source_kernel,
    kernel_bkg=background_kernel,
    significance_threshold=SIGNIFICANCE_THRESHOLD,
    mask_dilation_radius=MASK_DILATION_RADIUS,
)

niter_max = 4
result = kbe.run(images, niter_max=niter_max)

fig = plt.figure(figsize=(8, 4))

niter_max = len(kbe.images_stack)
crop_width = ((0, 1), (500, 500))
fig = plt.figure(figsize=(12, 6))

for idx, images in enumerate(kbe.images_stack):
    ax_bkg = fig.add_subplot(niter_max + 1, 2, 2 * idx + 1)
    bkg = images['background'].crop(crop_width)
    bkg.plot(ax=ax_bkg, vmin=0, vmax=1)
    ax_bkg.set_title('Background Estimation, Iteration {0}'.format(idx),
                     fontsize='small')
    ax_bkg.set_axis_off()

    ax_sig = fig.add_subplot(niter_max + 1, 2, 2 * idx + 2)
    sig = images['significance'].crop(crop_width)
    sig.plot(ax=ax_sig, vmin=-3, vmax=20)
    ax_sig.set_title('Significance, Iteration {0}'.format(idx),
                     fontsize='small')
    ax_sig.set_axis_off()
    mask = images['exclusion'].crop(crop_width).data
    ax_sig.contour(mask, levels=[0], linewidths=2, colors='green')

plt.tight_layout()
plt.show()