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
import logging

logging.getLogger("ultranest").setLevel(logging.WARNING)

from gammapy.datasets import MapDataset, Datasets
from gammapy.estimators import FluxPointsEstimator, ExcessMapEstimator
from gammapy.modeling import select_nested_models
from gammapy.modeling.models import (
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    create_crab_spectral_model,
    FoVBackgroundModel,
    Models,
)
from gammapy.estimators import FluxCollectionEstimator
from gammapy.modeling import Fit, Parameter, Sampler
from gammapy.modeling.models import UniformPrior
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
    selection_optional="all",
    correlate_off=True,
    correlation_radius=0.2 * u.deg,
)

lima_maps = estimator.run(dataset)

significance_map = lima_maps["sqrt_ts"]
excess_map = lima_maps["npred_excess"]

fig, (ax1, ax2) = plt.subplots(
    figsize=(11, 4), subplot_kw={"projection": lima_maps.geom.wcs}, ncols=2
)
ax1.set_title("Significance map")
significance_map.plot(ax=ax1, add_cbar=True, cmap="coolwarm")
ax2.set_title("Excess map")
excess_map.plot(ax=ax2, add_cbar=True, cmap="coolwarm")
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
# when making quick look plots to search for the presence of new sources within a field - for example
# in the case of alerts from Gravitational Wave detectors.
#

lima_maps.flux_ul.plot(add_cbar=True, cmap="Reds")
plt.show()


###############################################################################
# .. _compute_upper_lims:
#
# Compute upper limits for a source
# ---------------------------------
#
# Now, we address a more specific question.
# Suppose we were expecting a source at a specific position, say at the center of our
# map.
# We’ll attempt to fit a point source at the location to determine whether the signal
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
# To prevent the fit from drifting to unrelated positions, we freeze the spatial parameters.
# Alternatively, you can constrain the parameter ranges to stay within your expected region.
# We also add a background model with a free normalisation to account for uncertainties
# in the background. You can add more complex background corrections like a piecewise # # one, or keep the tilt free, etc
#
#
spectral_model = PowerLawSpectralModel()
spatial_model = PointSpatialModel(lon_0=187 * u.deg, lat_0=2.6 * u.deg, frame="icrs")
spatial_model.lat_0.frozen = True
spatial_model.lon_0.frozen = True

sky_model = SkyModel(
    spatial_model=spatial_model, spectral_model=spectral_model, name="source"
)
bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
dataset.models = Models([sky_model, bkg_model])
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
# The test statistic is ``ts ~ 4.6``. Note that here we have
# only 1 free parameter, the amplitude, and thus, we can assume the simple conversion
# significance = :math:`\sqrt{ts} \approx 2.14`.
# Therefore, the observed fluctuations are not significant above the background at the
# typically required level of 5-sigma.
# Next, we will estimate the differential upper limits of the source.

###############################################################################
# Differential upper limits
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As a source is not significantly detected, a measured flux would mostly
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
# - The **assumed spectral shape**. Here we assume a power law with index 2. The
#   amplitude only sets a reference flux, which is rescaled,
#   so its value does not affect the UL as long as it is positive.
#   In particular, *do not* use a model with a negative amplitude, as obtained above.
#   We will constrain the amplitude to positive values to remain in the physical
#   realm.
# - The **confidence level** and the **method**. Always report both, and, for a
#   Bayesian UL, also report the prior.
#
#
# .. _frequentist_upper_lims:
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
# ``n_sigma_ul=3`` to get the 99.87% UL.
#
# For computation reasons, we keep the background normalisation frozen, but in a real
# analysis, you should keep it free and use `reoptimize=True`.

sky_model.parameters["amplitude"].value = 1e-12
sky_model.parameters["index"].value = 2.0
sky_model.freeze(model_type="spatial")

energy_edges = dataset.geoms["geom"].axes["energy"].edges
fpe = FluxPointsEstimator(
    selection_optional=["ul", "sensitivity"],
    energy_edges=energy_edges,
    n_sigma_ul=3,
    n_sigma_sensitivity=3,
    n_jobs=12,
    source="source",
    reoptimize=True,
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
# Since this is computationally intensive, we will avoid executing it in this
# notebook, but show only the static plot.
#
# Instead of running many simulations, a quick alternative is the Asimov
# approximation of `Cowan et al. (2011) <https://arxiv.org/abs/1007.1727>`__.
# A single fit to the Asimov dataset (counts equal to the predicted background,
# with no fluctuations) gives the median expected upper limit and its ±1σ and ±2σ
# bands.
# The `~gammapy.estimators.FluxPointsEstimator` computes the
# flux that would give an ``n_sigma_sensitivity`` detection (default 5) on an
# Asimov dataset, i.e. in the median case. It answers the question: *would this
# source have been detectable with this exposure and IRFs?* It does not include
# conditions such as a minimum number of gamma rays
# (see :doc:`/tutorials/analysis-1d/cta_sensitivity`), and it depends on the
# analysis configuration.


def run_mc():
    n_toys = 100  # increase for smoother bands; this is the slow step

    dnde_ul_toys = []
    for idx in range(n_toys):
        if idx % 10 == 0:
            print(f"Running for ---------- ${idx}")
        toy = dataset.copy(name=f"toy-{idx}")
        toy.models = None
        toy.fake(random_state=idx)
        toy.models = Models(
            [sky_model.copy(name="source"), FoVBackgroundModel(dataset_name=toy.name)]
        )
        fp_toy = fpe.run(toy)
        dnde_ul_toys.append(fp_toy.dnde_ul.quantity.squeeze())

    unit = u.Unit("cm-2 s-1 TeV-1")
    dnde_ul_toys = u.Quantity(dnde_ul_toys).to_value(unit)
    p025, p16, p50, p84, p975 = np.nanpercentile(
        dnde_ul_toys, [2.5, 16, 50, 84, 97.5], axis=0
    )

    energy = fp1.energy_ref.to_value("TeV")

    ax = fp1.plot(sed_type="dnde", label="Observed UL", color="black", lw=2)
    ax.fill_between(
        energy, p025, p975, color="gold", alpha=0.5, label="Expected UL, 95%"
    )
    ax.fill_between(
        energy, p16, p84, color="limegreen", alpha=0.3, label="Expected UL, 68%"
    )
    ax.plot(energy, p50, "--", label="Expected UL, median", color="blue")
    ax.plot(
        energy,
        fp1.dnde_sensitivity.quantity.squeeze().to_value(unit),
        linestyle="dashdot",
        label="Sensitivity ",
        color="red",
    )
    ax.loglog()
    ax.set_xlabel("Energy [TeV]")
    ax.set_ylabel(f"dnde [{unit.to_string('latex_inline')}]")
    ax.legend()
    plt.show()


#
# .. image:: ../../_static/freq_UL_bands.png
#
#
#
#
######################################################################
# As the observed upper limits lie inside the bands, the data are consistent with
# background only. If they fall below the yellow band, the background has
# fluctuated down (or is overestimated).

# In the limit of high counts, the median expected upper limit and the sensitivity (at the
# same ``n_sigma``) are expected to be equal. With few counts, as here,
# the expected upper limit lies above the sensitivity, and the bands are the
# appropriate reference for the observed upper limits.
#
######################################################################
#
# .. _bayesain_upper_lims:
#
# Bayesian upper limits
# ~~~~~~~~~~~~~~~~~~~~~
#
# An alternative to the frequentist approach above is
# a **Bayesian credible upper limit**, the flux below which a chosen fraction
# of the posterior probability lies.
#
# This requires a prior on every free parameter. To keep the example simple,
# we fix the spectral index and the position, so the amplitude is the only
# free parameter. We use a uniform prior starting at zero. This builds the
# physical boundary (no negative fluxes) directly into the result.
# Its upper bound must lie well above the upper
# limits.
#
# **Note:** a Bayesian upper limit depends on the choice of prior. A prior that is
# uniform in amplitude is the usual choice for upper limits. A log-uniform prior
# would favour small fluxes and give a lower limit, which then depends on the
# lower bound of the prior.
#
# `~gammapy.estimators.FluxCollectionEstimator` fits the same normalisation per
# energy bin as above. It can use either a `~gammapy.modeling.Fit` or a
# `~gammapy.modeling.Sampler` as solver. With a sampler, the upper limit is the
# ``Φ(n_sigma_ul)`` quantile of the posterior of the normalisation.
# While reporting results, always quote the used prior.

norm_max = 40
norm_bayes = Parameter("norm", value=1, prior=UniformPrior(min=0, max=norm_max))

sampler = Sampler(
    backend="ultranest",
    options={
        "min_num_live_points": 300,
        "frac_remain": 0.3,
        "log_dir": None,
        "show_status": False,
        "viz_callback": False,
    },
)

fce_bayes = FluxCollectionEstimator(
    energy_edges=energy_edges,
    models=[sky_model],
    n_sigma_ul=3,
    norm=norm_bayes,
    solver=sampler,
)
fp_bayes = fce_bayes.run(Datasets([dataset]))["flux_points"]["source"]

fig, ax = plt.subplots()
unit = u.Unit("cm-2 s-1 TeV-1")
energy = fp_bayes.energy_ref.to_value("TeV")
fp1.plot(ax=ax, fmt="o-", label="Frequentist")
ax.plot(
    energy,
    fp_bayes.dnde_ul.quantity.squeeze().to_value(unit),
    "s--",
    label="Bayesian (uniform prior, norm ≥ 0)",
)
ax.loglog()
ax.set_xlabel("Energy [TeV]")
ax.set_ylabel(f"dnde upper limit, 99.8% [{unit.to_string('latex_inline')}]")
ax.legend()
plt.show()

# sphinx_gallery_thumbnail_number = 6

######################################################################
# Comparing the two upper limits
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Note that the frequentist and Bayesian upper limits answer different questions, so
# they do not have to give the same number.
#
# - **Frequentist (profile likelihood) upper limit.** This is a statement about
#   the *procedure*. If the experiment were repeated many times, the intervals
#   it produces would contain the true flux in a fixed fraction of the
#   repetitions (this is called *coverage*). It says nothing about how probable
#   any particular flux value is, and it does not use a prior.
#
# - **Bayesian credible upper limit.** This is a statement about the flux,
#   given the data we actually observed and the prior we chose: "there is a 99.8%
#   probability that the flux lies below this value". It depends heavily on the
#   choice of the prior.
#
# In the regime of many counts, with the best fit far from zero and a flat
# prior, the two results converge. In that limit the
# likelihood is close to a Gaussian, and the remaining difference is only the
# confidence-level convention.
#
# They differ most in exactly the situation of this tutorial: a non-detection
# near the physical boundary of zero flux.
#
# - Due to downward fluctuations of the background, frequentist upper limits can
#   become very small, excluding fluxes that the observation could never have detected.
# - The uniform prior starting at zero removes the unphysical region and
#   renormalises the posterior over the physical region. The Bayesian upper
#   limit therefore reacts much less to downward fluctuations, and it stays
#   closer to the sensitivity.
#
# Profiling versus marginalisation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# A real analysis usually has other free parameters besides the one we want to
# constrain, like the other sources in the field, etc. These are called *nuisance
# parameters* (θ). The two approaches treat them in different ways when constraining the
# parameter of  interest (μ, here the amplitude).
#
# - **Profiling (frequentist).** For each value of μ, the nuisance parameters
#   are re-fitted to their best values:
#
#   .. math:: L_{\rm prof}(\mu) = \max_{\theta} L(\mu, \theta)
#
#   In Gammapy this is what ``reoptimize=True`` does, in
#   `~gammapy.modeling.Fit.stat_profile` and in the estimators.
#   The default of `~gammapy.estimators.FluxPointsEstimator` is
#   ``reoptimize=False``. In that case the other parameters are kept *fixed* at
#   their current values. This is neither profiling nor marginalisation, and it
#   makes the upper limits too tight when the nuisance parameters are not well
#   constrained.
#
# - **Marginalisation (Bayesian).** The nuisance parameters are integrated out.
#   Every value of θ contributes, weighted by its likelihood and its prior:
#
#   .. math:: p(\mu \mid D) \propto \int L(\mu, \theta)\, \pi(\mu, \theta)\, {\rm d}\theta
#
#   With a sampler, marginalisation needs no extra work. Each column of
#   ``result.samples`` is already a sample of the marginal posterior of that
#   parameter, so a histogram of one column is its marginal distribution.
#
# Profiling keeps only the *best* nuisance values for each μ. Marginalisation
# also counts *how many* nuisance values are compatible with the data, i.e. the
# volume of the allowed region. The two give the same result when the
# likelihood is close to Gaussian in θ, with a width that does not depend on μ.
# They differ when the likelihood is skewed, when μ and θ are strongly
# correlated in a non-linear way, when a parameter is close to a boundary, or
# when a nuisance parameter is so poorly constrained by the data that its
# prior dominates.
#
######################################################################
#
# .. _integral_upper_lims:
#
# Integral upper limits
# ~~~~~~~~~~~~~~~~~~~~~
#
# To compute the integral upper limits between certain energies,
# one can simply run the estimators, eg `~gammapy.estimators.FluxPointsEstimator`
# with one bin in energy.
#
emin = energy_edges[0]
emax = energy_edges[-1]
fpe2 = FluxPointsEstimator(
    selection_optional=["ul"], energy_edges=[emin, emax], source="source"
)
fp2 = fpe2.run(dataset)
print(
    f"Integral upper limit between {emin:.1f} and {emax:.1f} is {fp2.flux_ul.quantity.ravel()[0]:.2e}"
)
