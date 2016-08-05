"""Estimate a diffuse emission model from Fermi LAT data.
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gammapy import datasets
from gammapy.image import binary_disk
from gammapy.detect import KernelBackgroundEstimator, KernelBackgroundEstimatorData

# *** PREPARATION ***

# Parameters

CORRELATION_RADIUS = 10  # Pixels
SIGNIFICANCE_THRESHOLD = 5  # Sigma
MASK_DILATION_RADIUS = 10  # Pixels

psf_file = datasets.FermiGalacticCenter.psf()

# Load/create example model images.
filename = datasets.gammapy_extra.filename(
    'datasets/source_diffuse_separation/galactic_simulations/fermi_counts.fits')

# *** LOADING INPUT ***

# Counts must be provided as an ImageHDU
counts = fits.open(filename)[0].data
header = fits.open(filename)[0].header
images = KernelBackgroundEstimatorData(counts=counts, header=header)

source_kernel = binary_disk(CORRELATION_RADIUS)

background_kernel = np.ones((10, 100))

# *** ITERATOR ***

kbe = KernelBackgroundEstimator(
    images=images,
    source_kernel=source_kernel,
    background_kernel=background_kernel,
    significance_threshold=SIGNIFICANCE_THRESHOLD,
    mask_dilation_radius=MASK_DILATION_RADIUS,
)

n_iterations = 4

# *** RUN & PLOT ***
plt.figure(figsize=(8, 4))

for iteration in range(n_iterations):
    kbe.run_iteration()
    mask_hdu = kbe.mask_image_hdu
    mask = mask_hdu.data[:, 1400:2000]

    plt.subplot(n_iterations, 2, 2 * iteration + 1)
    background_hdu = kbe.background_image_hdu
    data = background_hdu.data[:, 1400:2000]
    plt.imshow(data, vmin=0, vmax=1)
    plt.contour(mask, levels=[0], linewidths=2, colors='white')
    plt.axis('off')
    plt.title('Background Estimation, Iteration {0}'.format(iteration),
              fontsize='small')

    plt.subplot(n_iterations, 2, 2 * iteration + 2)
    significance_hdu = kbe.significance_image_hdu
    data = significance_hdu.data[:, 1400:2000]
    plt.imshow(data, vmin=-3, vmax=5, cmap=plt.cm.Greys_r)
    plt.contour(mask, levels=[0], linewidths=2, colors='red')
    plt.axis('off')
    plt.title('Significance Image, Iteration {0}'.format(iteration),
              fontsize='small')

plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95)
