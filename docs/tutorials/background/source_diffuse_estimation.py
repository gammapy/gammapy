"""Estimate a diffuse emission model from Fermi LAT data.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import download_file
from gammapy import datasets
from gammapy.background import IterativeKernelBackgroundEstimator, GammaImages
from gammapy.irf import EnergyDependentTablePSF
from gammapy.image import make_empty_image, catalog_image, binary_disk, cube_to_image, solid_angle

# *** PREPARATION ***

# Parameters

CORRELATION_RADIUS = 10 # Pixels
SIGNIFICANCE_THRESHOLD = 5 # Sigma
MASK_DILATION_RADIUS = 10 # Pixels

psf_file = datasets.FermiGalacticCenter.psf()

# Load/create example model images.
filename = datasets.get_path('source_diffuse_separation/galactic_simulations/fermi_counts.fits',
                         location='remote')

# *** LOADING INPUT ***

# Counts must be provided as an ImageHDU
counts = fits.open(filename)[0].data
header = fits.open(filename)[0].header
images = GammaImages(counts=counts, header=header)

source_kernel = binary_disk(CORRELATION_RADIUS)

background_kernel = np.ones((10, 100))

# *** ITERATOR ***

ibe = IterativeKernelBackgroundEstimator(images=images,
                                         source_kernel=source_kernel,
                                         background_kernel=background_kernel,
                                         significance_threshold=SIGNIFICANCE_THRESHOLD,
                                         mask_dilation_radius=MASK_DILATION_RADIUS
                                         )

n_iterations = 4

# *** RUN & PLOT ***
plt.figure(figsize=(8, 4))

for iteration in range(n_iterations):
    ibe.run_iteration()
    mask_hdu = ibe.mask_image_hdu
    mask = mask_hdu.data[:, 1400:2000]

    plt.subplot(n_iterations, 2, 2 * iteration + 1)
    background_hdu = ibe.background_image_hdu
    data = background_hdu.data[:, 1400:2000]
    plt.imshow(data, vmin=0, vmax=1)
    plt.contour(mask, levels=[0], linewidths=2, colors='white')
    plt.axis('off')
    plt.title('Background Estimation, Iteration {0}'.format(iteration),
              fontsize='small')
    
    plt.subplot(n_iterations, 2, 2 * iteration + 2)
    significance_hdu = ibe.significance_image_hdu
    data = significance_hdu.data[:, 1400:2000]
    plt.imshow(data, vmin=-3, vmax=5, cmap = plt.cm.Greys_r)
    plt.contour(mask, levels=[0], linewidths=2, colors='red')
    plt.axis('off')
    plt.title('Significance Image, Iteration {0}'.format(iteration),
              fontsize='small')

plt.subplots_adjust(wspace=0.05, hspace=0.05, left=0.05, right=0.95)
