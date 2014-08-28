"""Estimate a diffuse emission model from Fermi LAT data.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import download_file
from gammapy.datasets import FermiGalacticCenter
from gammapy.background import IterativeKernelBackgroundEstimator, GammaImages
from gammapy.irf import EnergyDependentTablePSF
from gammapy.image import make_empty_image, catalog_image, binary_disk
from gammapy.image.utils import cube_to_image, solid_angle

# *** PREPARATION ***

# Parameters

CORRELATION_RADIUS = 0.3
SIGNIFICANCE_THRESHOLD = 5
MASK_DILATION_RADIUS = 0.3

psf_file = FermiGalacticCenter.filenames()['psf']
psf = EnergyDependentTablePSF.read(psf_file)

# Load/create example model images.
BASE_URL = 'https://github.com/ellisowen/gammapy-extra/blob/diffuse_source_files/datasets/'
file = 'source_diffuse_separation/diffuse.fits.gz?raw=true'
url = BASE_URL + file
filename = download_file(url, cache=True)

# First load the diffuse counts component
diffuse_image_hdu = fits.open(filename)[1]
solid_angle_image = solid_angle(diffuse_image_hdu)
diffuse_image_true = diffuse_image_hdu.data/diffuse_image_hdu.data.sum()

# Then add the source counts component
# Make the reference the same shape as the diffuse counts map.
reference = make_empty_image(nxpix=601, nypix=401, binsz=0.1)

sources = catalog_image(reference, psf, catalog='1FHL',
                        source_type='point', total_flux='True')

source_image_true = sources.data/sources.data.sum()

# Select source & diffuse fraction, and determine number of counts
# This step is for the purposes of this demonstration only and would not be
# done with real data!

total_counts = 1e6
source_frac = 0.9

counts_data = (source_frac * source_image_true + (1-source_frac) * diffuse_image_true) * total_counts

# *** LOADING INPUT ***

# Counts must be provided as an ImageHDU
counts = fits.ImageHDU(data=counts_data, header=reference.header)
# Start with flat background estimate
# Background must be provided as an ImageHDU
background_data=np.ones_like(counts_data, dtype=float)
background = fits.ImageHDU(data=background_data, header=reference.header)
images = GammaImages(counts=counts, background=background)

source_kernel = binary_disk(CORRELATION_RADIUS).astype(float)
source_kernel /= source_kernel.sum()

background_kernel = np.ones((10, 100))

# *** ITERATOR ***

ibe = IterativeKernelBackgroundEstimator(images=images,
                                         source_kernel=source_kernel,
                                         background_kernel=background_kernel,
                                         significance_threshold=SIGNIFICANCE_THRESHOLD,
                                         mask_dilation_radius=MASK_DILATION_RADIUS,
                                         save_intermediate_results=True
                                         )

n_iterations = 6

# *** RUN & PLOT ***

for iteration in range(n_iterations):
    ibe.run_iteration()
    mask_hdu = ibe.mask_image_hdu
    mask = mask_hdu.data[160:240,:]

    plt.subplot(n_iterations, 2, 2 * iteration + 1)
    background_hdu = ibe.background_image_hdu
    data = background_hdu.data[160:240,:]
    plt.imshow(data)#, vmin=0, vmax=0.2)
    plt.contour(mask, levels=[0], linewidths=2, colors='white')
    plt.axis('off')
    plt.title('Background Estimation, Iteration {0}'.format(iteration), fontsize='small')
    
    plt.subplot(n_iterations, 2, 2 * iteration + 2)
    significance_hdu = ibe.significance_image_hdu
    data = significance_hdu.data[160:240,:]
    plt.imshow(data, vmin=-3, vmax=5)
    plt.contour(mask, levels=[0], linewidths=2, colors='white')
    plt.axis('off')
    plt.title('Significance Image, Iteration {0}'.format(iteration), fontsize='small')

plt.tight_layout()
