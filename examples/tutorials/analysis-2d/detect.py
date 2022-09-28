"""
Source detection and significance maps
======================================

Build a list of significant excesses in a Fermi-LAT map.

Context
-------

The first task in a source catalogue production is to identify
significant excesses in the data that can be associated to unknown
sources and provide a preliminary parametrization in term of position,
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
from gammapy.datasets import MapDataset
from gammapy.estimators import ASmoothMapEstimator, TSMapEstimator
from gammapy.estimators.utils import find_peaks
from gammapy.irf import EDispKernelMap, PSFMap
from gammapy.maps import Map
from gammapy.modeling.models import PointSpatialModel, PowerLawSpectralModel, SkyModel

######################################################################
# Check setup
# -----------
from gammapy.utils.check import check_tutorials_setup

check_tutorials_setup()


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
# In the following example the `ASmoothMapEstimator.threshold` argument gives the minimum
# significance expected, values below are clipped.
#

# %%time
scales = u.Quantity(np.arange(0.05, 1, 0.05), unit="deg")
smooth = ASmoothMapEstimator(threshold=3, scales=scales, energy_edges=[10, 500] * u.GeV)
images = smooth.run(dataset)

plt.figure(figsize=(15, 5))
images["flux"].plot(add_cbar=True, stretch="asinh")


######################################################################
# TS map estimation
# -----------------
#
# The Test Statistic, TS = 2 ∆ log L (`Mattox et
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

# %%time
estimator = TSMapEstimator(
    model,
    kernel_width="1 deg",
    energy_edges=[10, 500] * u.GeV,
)
maps = estimator.run(dataset)


######################################################################
# Plot resulting images
# ~~~~~~~~~~~~~~~~~~~~~
#

plt.figure(figsize=(15, 5))
maps["sqrt_ts"].plot(add_cbar=True)

plt.figure(figsize=(15, 5))
maps["flux"].plot(add_cbar=True, stretch="sqrt", vmin=0)

plt.figure(figsize=(15, 5))
maps["niter"].plot(add_cbar=True)


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
sources

# Plot sources on top of significance sky image
plt.figure(figsize=(15, 5))

ax = maps["sqrt_ts"].plot(add_cbar=True)

ax.scatter(
    sources["ra"],
    sources["dec"],
    transform=plt.gca().get_transform("icrs"),
    color="none",
    edgecolor="w",
    marker="o",
    s=600,
    lw=1.5,
)


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
#    `analysis_1 <../../starting/analysis_1.ipynb>`__ and
#    `analysis_2 <../../starting/analysis_2.ipynb>`__ notebooks,
#    respectively
# -  Learn about 2D model fitting in the `modeling
#    2D <modeling_2D.ipynb>`__ notebook
# -  find more about Fermi-LAT data analysis in the
#    `fermi_lat <../../data/fermi_lat.ipynb>`__ notebook
# -  Use source candidates to build a model and perform a 3D fitting (see
#    `analysis_3d <../3D/analysis_3d.ipynb>`__,
#    `analysis_mwl <../3D/analysis_mwl.ipynb>`__ notebooks for some hints)
#
