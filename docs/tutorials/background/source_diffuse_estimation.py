"""Estimate a diffuse emission model from Fermi LAT data.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import Tophat2DKernel
from gammapy.datasets import FermiGalacticCenter
from gammapy.image import SkyImageList, SkyImage
from gammapy.detect import KernelBackgroundEstimator

# Parameters
CORRELATION_RADIUS = 10  # Pixels
SIGNIFICANCE_THRESHOLD = 5  # Sigma
MASK_DILATION_RADIUS = 0.5 * u.deg

# Load example images.
filename = ('$GAMMAPY_EXTRA/datasets/source_diffuse_separation/'
            'galactic_simulations/fermi_counts.fits')
counts = SkyImage.read(filename)
center = SkyCoord(0, 0, frame='galactic', unit='deg')

images = SkyImageList()
images['counts'] = counts.cutout(center, (10 * u.deg, 80 * u.deg))

kernel_src = Tophat2DKernel(CORRELATION_RADIUS).array
kernel_bkg = np.ones((10, 150))

kbe = KernelBackgroundEstimator(
    kernel_src=kernel_src,
    kernel_bkg=kernel_bkg,
    significance_threshold=SIGNIFICANCE_THRESHOLD,
    mask_dilation_radius=MASK_DILATION_RADIUS,
)

result = kbe.run(images)
kbe.images_stack_show()
plt.show()

