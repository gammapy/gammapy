.. include:: ../references.txt

.. _estimators:

**********************************
estimators - High level estimators
**********************************

.. currentmodule:: gammapy.estimators

Introduction
============
The `gammapy.estimators` submodule contains algorithms and classes
for high level flux and significance estimation. This includes
estimation flux points, flux maps, flux points, flux profiles and
flux light curves. All estimators feature a common API and allow
to estimate fluxes in bands of reconstructed energy.

The core of any estimator algorithm is hypothesis testing: a reference
model or counts excess is tested against a null hypothesis. From the
best fit reference model a flux is derived and a corresponding :math:`\Delta TS`
value from the difference in fit statistics to the null hypothesis.
Assuming one degree of freedom, :math:`\sqrt{\Delta TS}` represents an
approximation (`Wilk's theorem <https://en.wikipedia.org/wiki/Wilks%27_theorem>`_)
of the "classical significance". In case of a negative best fit flux,
e.g. when the background is overestimated, the significance is defined
as :math:`-\sqrt{\Delta TS}` by convention.

In general the flux can be estimated using two methods:

#. **Based on model fitting:** given a (global) best fit model with multiple model components,
   the flux of the component of interest is re-fitted in the chosen energy, time or spatial
   region. The new flux is given as a ``norm`` with respect to the global reference model.
   Optionally other component parameters in the global model can be re-optimised. This method
   is also named **forward folding**.

#. **Based on excess:** in the case of having one energy bin, neglecting the PSF and
   not re-optimising other parameters, one can estimate the significance based on the
   analytical solution by [LiMa1983]. In this case the "best fit" flux and significance
   are given by the excess over the null hypothesis. This method is also named
   **backward folding**.


Uniformly for both methods most estimators compute the same basic quantities:

================= =================================================
Quantity          Definition
================= =================================================
norm              Best fit norm with respect to the reference spectral model
norm_err          Symmetric error on the norm derived from the Hessian matrix
stat              Fit statistics value of the best fit hypothesis
stat_null         Fit statistics value of the null hypothesis
ts                Difference in fit statistics (`stat - stat_null` )
sqrt_ts           Square root of ts time sign(norm), in case of one degree of freedom, corresponds to significance (Wilk's theorem)
npred             Predicted counts of the best fit hypothesis, equivalent to correlated counts for backward folding
npred_null        Predicted counts of the null hypothesis, equivalent to correlated null counts for backward folding
npred_excess      Predicted counts of the excess over `npred_null`, equivalent to (`npred - npred_null`), equivalent to correlated counts for backward folding
================= =================================================


In addition the following optional quantities can be computed:

================= =================================================
Quantity          Definition
================= =================================================
norm_errp         Positive error of the norm
norm_errn         Negative error of the norm
norm_ul           Upper limit of the norm
norm_scan         Norm scan
stat_scan         Fit statistics scan
================= =================================================

To compute the error, assymetric errors as well as upper limits one can
specify the arguments ``n_sigma`` and ``n_sigma_ul``. The ``n_sigma``
arguments are translated into a TS difference assuming ``ts = n_sigma ** 2``.

In addition to the norm values a reference spectral model and energy ranges
are given. Using this reference spectral model the norm values can be converted
to the following different SED types:

================= =================================================
Quantity          Definition
================= =================================================
e_ref             Reference energy
e_min             Minimum energy
e_max             Maximum energy
dnde              Differential flux at ``e_ref``
flux              Integrated flux between ``e_min`` and ``e_max``
eflux             Integrated energy flux between ``e_min`` and ``e_max``
e2dnde            Differential energy flux between ``e_ref``
================= =================================================

The same can be applied for the error and upper limit information.
More information can be found on the `likelihood SED type page`_.


Getting Started
===============

.. toctree::
    :maxdepth: 1

    detect

Tutorials
=========

The main tutorial that demonstrates how to use Estimator classes are:

* `Light Curve tutorial (LightCurveEstimator) <../tutorials/analysis/time/light_curve.html>`__
* `Flare tutorial (LightCurveEstimator) <../tutorials/analysis/time/light_curve_flare.html>`__
* `Source detection (TSMapEstimator) <../tutorials/analysis/2D/detect.html>`__
* `Spectral analysis (FluxPointEstimator) <../tutorials/analysis/1D/spectral_analysis.html>`__
* `Detailed 3D analysis (ExcessMapEstimator) <../tutorials/analysis/3D/analysis_3d.html>`__


Reference/API
=============

.. automodapi:: gammapy.estimators
    :no-inheritance-diagram:
    :include-all-objects:


.. _`likelihood SED type page`: https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/binned_likelihoods/index.html


