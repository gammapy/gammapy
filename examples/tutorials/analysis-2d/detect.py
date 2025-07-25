"""
Source detection and significance maps
======================================

Build a list of significant excesses in a Fermi-LAT map.

Context
-------

The first task in a source catalog production is to identify
significant excesses in the data that can be associated to unknown
sources and provide a preliminary parametrization in terms of position,
extent, and flux. In this notebook we will use Fermi-LAT data to
illustrate how to detect candidate sources in counts images with known
background.

**Objective: build a list of significant excesses in a Fermi-LAT map**

Proposed approach
-----------------

This notebook show how to do source detection with Gammapy using the
methods available in `~gammapy.estimators`. We will use images from a
Fermi-LAT 3FHL high-energy Galactic center dataset to do this:

-  perform adaptive smoothing on counts image
-  produce 2-dimensional test-statistics (TS)
-  run a peak finder to detect point-source candidates
-  compute Li & Ma significance images
-  estimate source candidates radius and excess counts

Note that what we do here is a quick-look analysis, the production of
real source catalogs use more elaborate procedures.

We will work with the following functions and classes:

-  `~gammapy.maps.WcsNDMap`
-  `~gammapy.estimators.ASmoothMapEstimator`
-  `~gammapy.estimators.TSMapEstimator`
-  `~gammapy.estimators.utils.find_peaks`

"""

######################################################################
# Setup
# -----
#
# As always, let’s get started with some setup …
#

import numpy as np
import astropy.units as u

# %matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from gammapy.datasets import MapDataset
from gammapy.estimators import ASmoothMapEstimator, TSMapEstimator
from gammapy.estimators.utils import find_peaks, find_peaks_in_flux_map
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map
from gammapy.modeling.models import PointSpatialModel, PowerLawSpectralModel, SkyModel


######################################################################
# Read in input images
# --------------------
#
# We first read the relevant maps:
#

counts = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-counts-cube.fits.gz")
background = Map.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-background-cube.fits.gz"
)

exposure = Map.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-exposure-cube.fits.gz")

psfmap = PSFMap.read(
    "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-psf-cube.fits.gz",
    format="gtpsf",
)

edisp = EDispKernelMap.from_diagonal_response(
    energy_axis=counts.geom.axes["energy"],
    energy_axis_true=exposure.geom.axes["energy_true"],
)

dataset = MapDataset(
    counts=counts,
    background=background,
    exposure=exposure,
    psf=psfmap,
    name="fermi-3fhl-gc",
    edisp=edisp,
)


######################################################################
# Adaptive smoothing
# ------------------
#
# For visualisation purpose it can be nice to look at a smoothed counts
# image. This can be performed using the adaptive smoothing algorithm from
# `Ebeling et
# al. (2006) <https://ui.adsabs.harvard.edu/abs/2006MNRAS.368...65E/abstract>`__.
#
# In the following example the `~gammapy.estimators.ASmoothMapEstimator.threshold`
# argument gives the minimum significance expected, values below are clipped.
#

scales = u.Quantity(np.arange(0.05, 1, 0.05), unit="deg")
smooth = ASmoothMapEstimator(threshold=3, scales=scales, energy_edges=[10, 500] * u.GeV)
images = smooth.run(dataset)

plt.figure(figsize=(9, 5))
images["flux"].plot(add_cbar=True, stretch="asinh")
plt.show()


######################################################################
# TS map estimation
# -----------------
#
# The Test Statistic, :math:`TS = 2 \Delta log L` (`Mattox et
# al. 1996 <https://ui.adsabs.harvard.edu/abs/1996ApJ...461..396M/abstract>`__),
# compares the likelihood function L optimized with and without a given
# source. The TS map is computed by fitting by a single amplitude
# parameter on each pixel as described in Appendix A of `Stewart
# (2009) <https://ui.adsabs.harvard.edu/abs/2009A%26A...495..989S/abstract>`__.
# The fit is simplified by finding roots of the derivative of the fit
# statistics (default settings use `Brent’s
# method <https://en.wikipedia.org/wiki/Brent%27s_method>`__).
#
# We first need to define the model that will be used to test for the
# existence of a source. Here, we use a point source.
#

spatial_model = PointSpatialModel()

# We choose units consistent with the map units here...
spectral_model = PowerLawSpectralModel(amplitude="1e-22 cm-2 s-1 keV-1", index=2)
model = SkyModel(spatial_model=spatial_model, spectral_model=spectral_model)


######################################################################
# Here we show a full configuration of the estimator. We remind the user of the meaning
# of the various quantities:
#
# -  ``model``: a `~gammapy.modeling.models.SkyModel` which is converted to a source model kernel
# -  ``kernel_width``: the width for the above kernel
# -  ``n_sigma``: number of sigma for the flux error
# -  ``n_sigma_ul``: the number of sigma for the flux upper limits
# -  ``selection_optional``: what optional maps to compute
# -  ``n_jobs``: for running in parallel, the number of processes used for the computation
# -  ``sum_over_energy_groups``: to sum over the energy groups or fit the `norm` on the full energy cube


estimator = TSMapEstimator(
    model=model,
    kernel_width="1 deg",
    energy_edges=[10, 500] * u.GeV,
    n_sigma=1,
    n_sigma_ul=2,
    selection_optional=None,
    n_jobs=1,
    sum_over_energy_groups=True,
)


maps = estimator.run(dataset)

######################################################################
# Accessing and visualising results
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Below we print the result of the `~gammapy.estimators.TSMapEstimator`. We have access to a number of
# different quantities, as shown below. We can also access the quantities names
# through ``maps.available_quantities``.
#

print(maps)

######################################################################
#

fig, (ax1, ax2, ax3) = plt.subplots(
    ncols=3,
    figsize=(20, 3),
    subplot_kw={"projection": counts.geom.wcs},
    gridspec_kw={"left": 0.1, "right": 0.98},
)

maps["sqrt_ts"].plot(ax=ax1, add_cbar=True)
ax1.set_title("Significance map")
maps["flux"].plot(ax=ax2, add_cbar=True, stretch="sqrt", vmin=0)
ax2.set_title("Flux map")
maps["niter"].plot(ax=ax3, add_cbar=True)
ax3.set_title("Iteration map")
plt.show()


######################################################################
# The flux in each pixel is obtained by multiplying a reference model with a
# normalisation factor:

print(maps.reference_model)

######################################################################
#
maps.norm.plot(add_cbar=True, stretch="sqrt")
plt.show()


######################################################################
# Source candidates
# -----------------
#
# Let’s run a peak finder on the `sqrt_ts` image to get a list of
# point-sources candidates (positions and peak `sqrt_ts` values). The
# `~gammapy.estimators.utils.find_peaks` function performs a local maximum search in a sliding
# window, the argument `min_distance` is the minimum pixel distance
# between peaks (smallest possible value and default is 1 pixel).
#

sources = find_peaks(maps["sqrt_ts"], threshold=5, min_distance="0.25 deg")
nsou = len(sources)
display(sources)

# Plot sources on top of significance sky image
plt.figure(figsize=(9, 5))
ax = maps["sqrt_ts"].plot(add_cbar=True)

ax.scatter(
    sources["ra"],
    sources["dec"],
    transform=ax.get_transform("icrs"),
    color="none",
    edgecolor="w",
    marker="o",
    s=600,
    lw=1.5,
)
plt.show()

# sphinx_gallery_thumbnail_number = 3


######################################################################
# We can also utilise `~gammapy.estimators.utils.find_peaks_in_flux_map`
# to display various parameters from the FluxMaps

sources_flux_map = find_peaks_in_flux_map(maps, threshold=5, min_distance="0.25 deg")
display(sources_flux_map)


######################################################################
# Note that we used the instrument point-spread-function (PSF) as kernel,
# so the hypothesis we test is the presence of a point source. In order to
# test for extended sources we would have to use as kernel an extended
# template convolved by the PSF. Alternatively, we can compute the
# significance of an extended excess using the Li & Ma formalism, which is
# faster as no fitting is involve.
#


######################################################################
# What next?
# ----------
#
# In this notebook, we have seen how to work with images and compute TS
# and significance images from counts data, if a background estimate is
# already available.
#
# Here’s some suggestions what to do next:
#
# -  Look how background estimation is performed for IACTs with and
#    without the high level interface in
#    :doc:`/tutorials/starting/analysis_1` and
#    :doc:`/tutorials/starting/analysis_2` notebooks,
#    respectively
# -  Learn about 2D model fitting in the :doc:`/tutorials/analysis-2d/modeling_2D` notebook
# -  Find more about Fermi-LAT data analysis in the
#    :doc:`/tutorials/data/fermi_lat` notebook
# -  Use source candidates to build a model and perform a 3D fitting (see
#    :doc:`/tutorials/analysis-3d/analysis_3d`,
#    :doc:`/tutorials/analysis-3d/analysis_mwl` notebooks for some hints)
#
