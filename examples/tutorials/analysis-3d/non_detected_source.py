"""
Computing flux upper limits
===========================

Explore how to compute flux upper limits for a non-detected source.

Prerequisites
-------------

It is advisable to understand the general Gammapy modeling and fitting
framework before proceeding with this notebook, e.g. :doc:`/user-guide/modeling`.

Context
-------

In the study of VHE sources, one often encounters no significant
detection even after long exposures. In that case, it may be useful to
compute flux upper limits (UL) for the said target consistent with the observation.

Proposed approach
-----------------

In this section, we will use an empty observation from the H.E.S.S. DL3 DR1 to understand how
to quantify non-detections. There are two distinct approaches to consider:

- Test for the presence of emission anywhere in a map and compute an integral flux upper limit at
  any position (i.e. UL map).
- Test the presence of emission from a potential source with given position and morphology and compute
  integral and differential UL.
- Use both frequentist and Bayesian approaches for computing the ULs.

"""

######################################################################
# Setup
# -----
#
# As usual, let’s start with some general imports…
#

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from gammapy.datasets import MapDataset, Datasets
from gammapy.estimators import FluxPointsEstimator, ExcessMapEstimator
from gammapy.modeling import select_nested_models
from gammapy.modeling.models import (
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    create_crab_spectral_model,
)
from gammapy.visualization import plot_distribution


######################################################################
# Load observation
# ----------------
#
# For computational purposes, we have
# already created a `~gammapy.datasets.MapDataset` from observation id ``20275`` from the
# public H.E.S.S. data release and stored it in ``$GAMMAPY_DATA``
#

dataset = MapDataset.read("$GAMMAPY_DATA/datasets/empty-dl4/empty-dl4.fits.gz")
dataset.peek()
plt.show()

######################################################################
# Create Upper Limit maps
# -----------------------
#
# We will first use the `~gammapy.estimators.ExcessMapEstimator` for a quick check to see if
# there are any potential sources in the field. The ``correlation_radius`` should be around the
# size of the source you are searching for. You may also use the
# `~gammapy.estimators.TSMapEstimator`.

estimator = ExcessMapEstimator(
    sum_over_energy_groups=True,
    selection_optional="all",
    correlate_off=True,
    correlation_radius=0.1 * u.deg,
)

lima_maps = estimator.run(dataset)

significance_map = lima_maps["sqrt_ts"]
excess_map = lima_maps["npred_excess"]

fig, (ax1, ax2) = plt.subplots(
    figsize=(11, 4), subplot_kw={"projection": lima_maps.geom.wcs}, ncols=2
)
ax1.set_title("Significance map")
significance_map.plot(ax=ax1, add_cbar=True)
ax2.set_title("Excess map")
excess_map.plot(ax=ax2, add_cbar=True)
plt.show()

######################################################################
# The significance map looks featureless. We will plot a histogram of the
# significance distribution and fit it with a standard normal.
# Deviations from a standard normal can suggest the presence of gamma-ray sources,
# or can also originate from incorrect modeling of the residual hadronic background.

ax = plt.subplot()
plot_distribution(
    significance_map,
    func="norm",
    ax=ax,
    kwargs_hist={"bins": 50, "range": (-4, 4), "density": True},
)
plt.show()


######################################################################
# We can also see the correlated upper limits at any position in the map. However, it is important to note
# that this is **not** a source UL, as the containment correction is not applied here. Instead, it gives the
# flux upper limits contained within the ``correlation_radius`` at each pixel. This can be useful
# when making quick look plots to search for the presence of new sources with a field - for example
# in the case of alerts from Gravitational Wave detectors.
#

lima_maps.flux_ul.plot(add_cbar=True, cmap="viridis")
plt.show()


###############################################################################
# .. _compute_upper_lims:
#
# Compute upper limits for a source
# ---------------------------------
#
# Now, we address a more specific question.
# Suppose we were expecting a source at a specific position, say the center of our map.
# We’ll attempt to fit a point source at that location to determine whether the signal
# is significant above the background.
#
# For this, we compare two hypotheses:
#
# - Null hypothesis, :math:`H_0` - only the background is present (i.e., no source)
# - Alternative hypothesis, :math:`H_1` - the background plus a point source
#
# The difference in test statistic (TS) between the two cases indicates the
# significance of the alternative hypothesis.
#
# For this purpose, we can utilise the function `~gammapy.modeling.select_nested_models` which
# performs these comparisons internally.
# As the null hypothesis, :math:`H_0`, corresponds to the case of no source, we set the amplitude to 0, in other words
# the source has no flux.
#
# To prevent the fit from converging to unrelated positions, we freeze the spatial parameters.
# Alternatively, you can constrain the parameter ranges to stay within your expected region.


spectral_model = PowerLawSpectralModel()
spatial_model = PointSpatialModel(lon_0=187 * u.deg, lat_0=2.6 * u.deg, frame="icrs")
spatial_model.lat_0.frozen = True
spatial_model.lon_0.frozen = True

sky_model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="source"
)
dataset.models = sky_model
LLR = select_nested_models(
    datasets=Datasets(dataset),
    parameters=[sky_model.parameters["amplitude"]],
    null_values=[0],
)
print(LLR)

######################################################################
# The fitted parameters under the alternative hypothesis:

print(LLR["fit_results"].parameters.to_table())

######################################################################
# The test statistic is ``ts ~ 4.7``. Note that here we have
# only 1 free parameter, the amplitude, and thus, we can assume the simple conversion
# significance = :math:`\sqrt{ts} \approx 2.2`.
# Therefore, the observed fluctuations are not significant above the background.
# Next, we will estimate the differential upper limits of the source.

###############################################################################
# Differential upper limits
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As source is not significantly detected, a measured flux would mostly
# reflect background fluctuations. Instead, we quote an **upper limit (UL)**:
# the largest flux that is still compatible with the data, at a chosen
# confidence level. We compute it in two ways: first a frequentist UL from the
# profile likelihood, then a Bayesian UL from the posterior distribution.
# The two answer different questions, as discussed at the end of this notebook.
#
# The upper limit is what we actually obtained from this particular
# data set, and is *not* an estimate of the true source flux.
# The upper limits also depends on:
#
# - the **assumed spectral shape**. Here we assume a power law with index 2. The
#   amplitude only sets a reference flux, which is rescaled,
#   so its value does not affect the UL as long as it is positive.
#   In particular, do not use a model with a negative amplitude, such as one
#   returned by the fit above.
# - the **confidence level** and the **method**. Always report both, and, for a
#   Bayesian UL, also report the prior.
#
# Frequentist differential upper limits
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A frequentist UL is the output of a procedure with a guaranteed
# *coverage*: if the observation were repeated many times, the interval
# ``[0, UL]`` would contain the true flux in the stated fraction of the
# repetitions (e.g. 95%). It is **not** the probability that the flux lies
# below the UL. In Gammapy, the UL is the flux at which the fit statistic rises
# by ``n_sigma_ul**2`` above its minimum (on the high-flux side). Asymptotically,
# this corresponds to a one-sided confidence level: ``n_sigma_ul=2`` (the default)
# gives ΔTS = 4, i.e. ~97.7%, and ``n_sigma_ul=1.645`` gives 95%.
# The `~gammapy.estimators.FluxPointsEstimator` computes one UL per energy bin.
# By default (``reoptimize=False``), only the normalisation is fitted in each bin;
# all other parameters, including the index, stay fixed. Here we use
# ``n_sigma_ul=5`` so that the ULs can be compared with the 5-sigma sensitivity
# computed below.
#

sky_model.parameters["amplitude"].value = 1e-14
sky_model.parameters["index"].value = 2.0
sky_model.freeze(model_type="spatial")

energy_edges = dataset.geoms["geom"].axes["energy"].edges
fpe = FluxPointsEstimator(
    selection_optional=["ul"],
    energy_edges=energy_edges,
    n_sigma_ul=5,
    source="source",
)

fp1 = fpe.run(dataset)


fp1.plot(sed_type="dnde")
plt.show()

######################################################################
# Expected upper limits (Brazil bands) and sensitivity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# An observed upper limit comes from a single realisation of the data. To see
# how unusual it is, we compare it with the distribution of upper limits we would
# expect if there were no source. We simulate many *background-only datasets*,
# compute the upper limits for each one in the same way as above, and take the
# median and the 68% and 95% ranges. These form the so-called Brazil bands.
#
#

n_toys = 100  # increase for smoother bands

dnde_ul_toys = []
for idx in range(n_toys):
    toy = dataset.copy(name=f"toy-{idx}")
    toy.models = None
    toy.fake(random_state=idx)
    toy.models = sky_model.copy(name="source")  # same reference model as above
    fp_toy = fpe.run(toy)
    dnde_ul_toys.append(fp_toy.dnde_ul.quantity.squeeze())

unit = u.Unit("cm-2 s-1 TeV-1")
dnde_ul_toys = u.Quantity(dnde_ul_toys).to_value(unit)
p025, p16, p50, p84, p975 = np.nanpercentile(
    dnde_ul_toys, [2.5, 16, 50, 84, 97.5], axis=0
)

energy = fp1.energy_ref.to_value("TeV")
fig, ax = plt.subplots()
ax.fill_between(energy, p025, p975, color="gold", label="Expected 95%")
ax.fill_between(energy, p16, p84, color="limegreen", label="Expected 68%")
ax.plot(energy, p50, "k--", label="Expected median")
ax.plot(energy, fp1.dnde_ul.quantity.squeeze().to_value(unit), "ko-", label="Observed")
ax.loglog()
ax.set_xlabel("Energy [TeV]")
ax.set_ylabel(f"dnde upper limit [{unit.to_string('latex_inline')}]")
ax.legend()
plt.show()

######################################################################
# If the observed upper limits lie inside the bands, the data are consistent with
# background only. If they fall below the yellow band, the background has
# fluctuated down (or is overestimated).

######################################################################
# Integral upper limits
# ~~~~~~~~~~~~~~~~~~~~~
#
# To compute the integral upper limits between certain energies,
# we can simply run  `~gammapy.estimators.FluxPointsEstimator`
# with one bin in energy.

emin = energy_edges[0]
emax = energy_edges[-1]
fpe2 = FluxPointsEstimator(
    selection_optional=["ul"], energy_edges=[emin, emax], source="source"
)
fp2 = fpe2.run(dataset)
print(
    f"Integral upper limit between {emin:.1f} and {emax:.1f} is {fp2.flux_ul.quantity.ravel()[0]:.2e}"
)
