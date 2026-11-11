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

We will use both frequentist and Bayesian approaches for computing the ULs.

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
logging.getLogger("ultranest").propagate = False

from gammapy.datasets import MapDataset, Datasets
from gammapy.estimators import FluxPointsEstimator, ExcessMapEstimator
from gammapy.modeling import select_nested_models
from gammapy.modeling.models import (
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
    FoVBackgroundModel,
    Models,
)
from gammapy.estimators import FluxCollectionEstimator
from gammapy.modeling import Parameter, Sampler
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
# `~gammapy.estimators.TSMapEstimator` which uses a forward folding approach.

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
significance_map.plot(ax=ax1, add_cbar=True, cmap="coolwarm", vmin=-4, vmax=4)
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
# in the background. You can add more complex background corrections such as letting the tilt free, or a piecewise correction, etc.
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
# significance = :math:`\sqrt{ts} \approx 2.14 \sigma`.
# Therefore, the observed fluctuations are not significant above the background at
# the typically required level of 5-sigma.
# Next, we will estimate the differential upper limits of the source.

###############################################################################
# Differential upper limits
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As the source is not significantly detected, a measured flux would mostly
# reflect background fluctuations. Instead, we quote an **upper limit (UL)**:
# the largest flux that is still compatible with the data, at a chosen
# confidence level. We compute it in two ways: first a frequentist UL from the
# profile likelihood, then a Bayesian UL from the posterior distribution.
# The two answer different questions, as discussed at the end of this notebook.
#
# The upper limit is the maximum flux compatible with this particular
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
#   Bayesian UL, also report the **prior**.
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
# repetitions (e.g. 95%).
#
# In Gammapy, the UL is the flux at which the fit statistic rises
# by ``n_sigma_ul**2`` above its minimum (on the high-flux side). Asymptotically,
# this corresponds to a one-sided confidence level: ``n_sigma_ul=2`` (the default)
# gives ΔTS = 4, i.e. ~97.7%, and ``n_sigma_ul=1.645`` gives 95%.
# The `~gammapy.estimators.FluxPointsEstimator` computes one UL per energy bin.
# By default (``reoptimize=False``), only the normalisation is fitted in each bin;
# all other parameters, including the index, stay fixed. Here we use
# ``n_sigma_ul=1.645`` to get the 95% UL.

# Note that in the case of a detection, n_sigma=2 still corresponds to ΔTS = 4 but is associated to a two-sided confidence level of 95%.
# In order to perform a comparison between the frequentist and Bayesian 95% ULs, we should specify n_sigma_ul=1.645 in the frequentist approach.
#
# For computational reasons here, we keep the background normalisation frozen,
# but **in a real analysis, you should keep it free and use** `reoptimize=True`.

sky_model.parameters["amplitude"].value = 1e-13
sky_model.parameters["index"].value = 2.0
sky_model.freeze(model_type="spatial")

energy_edges = dataset.geoms["geom"].axes["energy"].edges
fpe = FluxPointsEstimator(
    selection_optional="all",
    energy_edges=energy_edges,
    n_sigma_ul=1.645,
    source="source",
)

fp1 = fpe.run(dataset)

######################################################################
# It is useful to plot the upper limits along with fit statistic as a
# density plot to see the TS difference in each bin.

fp1.plot(sed_type="dnde", color="black")
fp1.plot_ts_profiles(sed_type="dnde")
plt.show()

# It can also be useful to visualise the likelihood profile for
# diagnostic purposes, as explained in `:doc:`/tutorials/details/fitting`.

######################################################################
# Expected upper limits (Brazil bands)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# An observed upper limit comes from a single realisation of the data.
# Repeated experiments of the same expected background (e.g. lightcurves) will lead to different Poisson realizations of the background and will yield different upper limit values. To see how unusual it is, we compare it with the distribution of upper limits we would
# expect *if there were no source*. We simulate many *background-only datasets*,
# compute the upper limits for each one in the same way as above, and take the
# median and the 68% and 95% ranges. These form the so-called Brazil bands.
#
# Since this is computationally intensive, we will avoid executing it in this
# notebook, but show only the static plot.
#
# Instead of running many simulations, a quick alternative is the Asimov
# approximation of `Cowan et al. (2011) <https://arxiv.org/abs/1007.1727>`__.
# A single fit to the Asimov dataset (counts equal to the predicted background,
# with no fluctuations) gives the median expected upper limit.
#


def run_mc():
    # Simulate
    n_toys = 100  # increase for smoother bands
    dnde_ul_toys = []
    for idx in range(n_toys):
        if idx % 10 == 0:
            print(f"Running for ---------- ${idx}")
        toy = dataset.copy(name=f"toy-{idx}")
        toy.models = None
        toy.fake(random_state=idx)
        toy.models = sky_model.copy(name="source")
        fp_toy = fpe.run(toy)
        dnde_ul_toys.append(fp_toy.dnde_ul.quantity.squeeze())

    unit = u.Unit("cm-2 s-1 TeV-1")
    dnde_ul_toys = u.Quantity(dnde_ul_toys).to_value(unit)
    p025, p16, p50, p84, p975 = np.nanpercentile(
        dnde_ul_toys, [2.5, 16, 50, 84, 97.5], axis=0
    )

    # Compute the Asimov UL
    asimov = dataset.copy(name="asimov")

    npred_bkg = asimov.npred_background()
    asimov.counts.data[...] = npred_bkg.data

    asimov.models = Models(
        [
            sky_model.copy(name="source"),
            FoVBackgroundModel(dataset_name=asimov.name),
        ]
    )

    fp_asimov = fpe.run(asimov)

    # Plot
    energy = fp1.energy_ref.to_value("TeV")

    ax = fp1.plot(sed_type="dnde", label="Observed UL", color="black", lw=2)
    ax.fill_between(
        energy, p025, p975, color="gold", alpha=0.5, label="Expected UL, 95%"
    )
    ax.fill_between(
        energy, p16, p84, color="limegreen", alpha=0.3, label="Expected UL, 68%"
    )
    ax.plot(energy, p50, linestyle="dotted", label="Expected UL, median", color="blue")
    ax.plot(
        energy,
        fp_asimov.dnde_ul.quantity.squeeze().to_value(unit),
        linestyle="dashdot",
        label="Asimov UL",
        color="red",
    )

    ax.loglog()
    ax.set_xlabel("Energy [TeV]")
    ax.set_ylabel(f"dnde [{unit.to_string('latex_inline')}]")
    ax.legend()
    plt.show()


######################################################################
#
# .. image:: ../../_static/freq_UL_bands.png

######################################################################
# If the observed upper limits lie inside the bands, it implies the data are
# consistent with background only.
# If they fall below the yellow band, the background is overestimated.
# To avoid this, you should always
# *reoptimize the parameters of the other models, specially the background model*.
# If they still occur, such points should be treated with caution as they
# may exclude fluxes the observation was not sensitive to. This usually indicates
# background systematics that have not been accounted for.

######################################################################
# .. _bayesian_upper_lims:
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
# free parameter. We use a uniform prior starting at zero.
# Its upper bound must lie well above the upper limits.
#
# **Note:** a Bayesian upper limit depends on the choice of prior. A prior that is
# uniform in amplitude is the usual choice for upper limits. A log-uniform prior
# would favour small fluxes and give a lower limit, which then depends on the
# lower bound of the prior.
#
# `~gammapy.estimators.FluxCollectionEstimator` fits the same normalisation per
# energy bin as above. It can use either a `~gammapy.modeling.Fit` or a
# `~gammapy.modeling.Sampler` as solver. With a sampler, the upper limit is the
# ``\Phi(n_sigma_ul)`` quantile of the posterior of the normalisation.
# While reporting results, always quote the used prior.

norm_max = 100
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
    n_sigma_ul=1.645,
    norm=norm_bayes,
    solver=sampler,
)
fp_bayes = fce_bayes.run(Datasets([dataset]))["flux_points"]["source"]

fig, ax = plt.subplots()
unit = u.Unit("cm-2 s-1 TeV-1")
energy = fp_bayes.energy_ref.to_value("TeV")
fp1.plot(ax=ax, fmt="o-", label="Frequentist UL")
ax.plot(
    energy,
    fp_bayes.dnde_ul.quantity.squeeze().to_value(unit),
    "s--",
    label="Bayesian Limit (uniform prior, norm ≥ 0)",
)
ax.loglog()
ax.set_xlabel("Energy [TeV]")
ax.set_ylabel(f"dnde upper limit, 95% [{unit.to_string('latex_inline')}]")
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
#   repetitions (this is called *coverage*). In this approach, the other free
#   parameters ("nuisance parameters") are re-fitted to their best values.
#
# - **Bayesian credible upper limit.** This is a statement about the flux,
#   given the data we actually observed and the prior we chose: "there is a 99.8%
#   probability that the flux lies below this value". It depends heavily on the
#   choice of the prior. The nuisance parameters are integrated out.
#   Every value contributes, weighted by its likelihood and its prior
#
# In the regime of many counts, with the best fit far from zero and a flat
# prior, the two results converge. In that limit the
# likelihood is close to a Gaussian, and the remaining difference is only the
# confidence-level convention.
#
######################################################################
# .. _integral_upper_lims:
#
# Integral upper limits
# ~~~~~~~~~~~~~~~~~~~~~
#
# To compute the integral upper limits between certain energies,
# one can simply run the estimators, eg `~gammapy.estimators.FluxPointsEstimator`
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
