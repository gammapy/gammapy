.. include:: ../references.txt

.. _gammapy_2p2_release:

2.2 (19th November, 2026)
=========================

- Released 19th November, 2026
- contributors
- pull requests since v (not all listed below)
- closed issues

Summary
-------


API changes
-----------

- The constructor of `~gammapy.data.ObservationTable` now takes an argument ``table`` instead of ``data``. [`#6620 <https://github.com/gammapy/gammapy/issues/6620>`_]
- Deprecated `gammap.utils.make_map_exposure_true_energy`, use the `make_exposure` method on `MapDatasetMaker` instead. [`#6882 <https://github.com/gammapy/gammapy/issues/6882>`_]
- Deprecate `gammapy.makers.utils.make_psf_map`. [`#6893 <https://github.com/gammapy/gammapy/issues/6893>`_]
- The options of `~gammapy.modeling.Sampler` are now given in a single ``options`` dictionary, using the backend's own option names. [`#6898 <https://github.com/gammapy/gammapy/issues/6898>`_]
- Deprecate `gammapy.makers.utils.make_edisp_map` and `gammapy.makers.utils.make_edisp_kernel_map`. [`#6906 <https://github.com/gammapy/gammapy/issues/6906>`_]
- Deprecate `gammapy.makers.make_map_background_irf`. [`#6946 <https://github.com/gammapy/gammapy/issues/6946>`_]


Infrastructure
--------------

- Add support for Python 3.14. Update tox and CI configurations accordingly. [`#6225 <https://github.com/gammapy/gammapy/issues/6225>`_]
- Fix the ``gammapy-dev`` conda environment for compiling ``sherpa`` from source and adding all documentation dependencies. [`#6930 <https://github.com/gammapy/gammapy/issues/6930>`_]


Documentation improvements
--------------------------

- # Add PIG for introducing new plotter classes [`#5726 <https://github.com/gammapy/gammapy/issues/5726>`_]
- Improves the documentation in the user guide on fit statistics. [`#6623 <https://github.com/gammapy/gammapy/issues/6623>`_]
- Fix issue in Fermi-LAT tutorial where dimmest sources are used for fitting rather than brightest. [`#6750 <https://github.com/gammapy/gammapy/issues/6750>`_]
- Add a new tutoial showing detailed Dark Matter analysis [`#6774 <https://github.com/gammapy/gammapy/issues/6774>`_]
- Update of the Dark Matter module main documentation. [`#6777 <https://github.com/gammapy/gammapy/issues/6777>`_]
- Add tutorial for on-off analysis using paired off runs. [`#6835 <https://github.com/gammapy/gammapy/issues/6835>`_]
- Add How-To entry to document ASDF [`#6981 <https://github.com/gammapy/gammapy/issues/6981>`_]
- Update the Upper Limit tutorial. [`#6997 <https://github.com/gammapy/gammapy/issues/6997>`_]
- Add a tutorial about Bayesian model selection using arviz package. [`#7018 <https://github.com/gammapy/gammapy/issues/7018>`_]


New features
------------

- Add `~gammapy.utils.region.CirclePixelRegionArray` and  `~gammapy.utils.region.CircleSkyRegionArray` classes to support array valued regions. [`#5420 <https://github.com/gammapy/gammapy/issues/5420>`_]
- Add `~gammapy.estimators.RegularizedFluxPointsEstimator` to perform estimate of flux points in true energy space using a regularized likelihood approach. [`#5625 <https://github.com/gammapy/gammapy/issues/5625>`_]
- Add `~gammapy.visualization.plot_pulse_profile_3PC` function to plot pulse profiles from the Fermi-LAT 3PC catalog. [`#5761 <https://github.com/gammapy/gammapy/issues/5761>`_]
- Add utility function to creates a 2D grid of `~regions.RectangleSkyRegion` tiles on a given WCS. [`#6006 <https://github.com/gammapy/gammapy/issues/6006>`_]
- Add `gammapy.modeling.models.SamplesKDEPrior` class to create a prior from samples using `scipy.stats.gaussian_kde`. [`#6503 <https://github.com/gammapy/gammapy/issues/6503>`_]
- Add nautilus-sampler as a possible `~gammapy.modeling.Sampler` backend. [`#6596 <https://github.com/gammapy/gammapy/issues/6596>`_]
- Update Fermi-LAT 4FGL catalog and add support for FL16Y list in `gammapy.catalog.SourceCatalog4FGL`. [`#6617 <https://github.com/gammapy/gammapy/issues/6617>`_]
- Add support for creating a `~gammapy.data.DataStore` from the results of a ivoa TapQuery. [`#6653 <https://github.com/gammapy/gammapy/issues/6653>`_]
- Implementation of the option of using custom files as dark matter spectra sources. [`#6660 <https://github.com/gammapy/gammapy/issues/6660>`_]
- Add ASDF serialization support for `~gammapy.maps.MapAxis`. [`#6706 <https://github.com/gammapy/gammapy/issues/6706>`_]
- Add ASDF serialization support for `~gammapy.maps.TimeMapAxis` and `~gammapy.maps.LabelMapAxis` . [`#6722 <https://github.com/gammapy/gammapy/issues/6722>`_]
- Add ASDF serialization support for `~gammapy.maps.MapAxes`. [`#6735 <https://github.com/gammapy/gammapy/issues/6735>`_]
- Add `~gammapy.datasets.MapDataset.downsample` functionality for the ``energy_true`` axis, fix `~gammapy.data.IRFMap.downsample` behaviour, and restrict the axes for downsampling. [`#6740 <https://github.com/gammapy/gammapy/issues/6740>`_]
- Added `~gammapy.astro.darkmatter.utils.add_factor_prior` helper function to support log-normal nuisance parameter treatment for astrophysical J-factors and D-factors. [`#6741 <https://github.com/gammapy/gammapy/issues/6741>`_]
- Fixed JFactory to correctly compute J/D-factors for extragalactic sources by decoupling rmax from distance and propagating the source distance through the line-of-sight integration geometry. [`#6743 <https://github.com/gammapy/gammapy/issues/6743>`_]
- Implementation of parameter validation and code restructuring for Dark Matter spectral models. [`#6744 <https://github.com/gammapy/gammapy/issues/6744>`_]
- Add ASDF serialization support for `~gammapy.maps.WcsGeom`. [`#6751 <https://github.com/gammapy/gammapy/issues/6751>`_]
- Add ASDF serialization support for `~gammapy.maps.HpxGeom`. [`#6761 <https://github.com/gammapy/gammapy/issues/6761>`_]
- Add ASDF serialization support for `~gammapy.maps.RegionGeom`. [`#6771 <https://github.com/gammapy/gammapy/issues/6771>`_]
- Add a dark matter analysis tutorial covering the basics of J/D-factor computation, spectral models, and expected gamma-ray flux. [`#6782 <https://github.com/gammapy/gammapy/issues/6782>`_]
- Add ASDF serialization support for `~gammapy.maps.WcsNDMap`. [`#6790 <https://github.com/gammapy/gammapy/issues/6790>`_]
- Add ASDF serialization support for `~gammapy.maps.HpxNDMap`. [`#6791 <https://github.com/gammapy/gammapy/issues/6791>`_]
- Add ASDF serialization support for `~gammapy.maps.RegionNDMap`. [`#6794 <https://github.com/gammapy/gammapy/issues/6794>`_]
- Addition of `~gammapy.modeling.models.LogNormalPrior` class. [`#6799 <https://github.com/gammapy/gammapy/issues/6799>`_]
- Ensure `gammapy.maps.axes.TimeMapAxis` start and stop values are exported to table in isot format. [`#6805 <https://github.com/gammapy/gammapy/issues/6805>`_]
- Deprecate `gammapy.datasets.Dataset.stat_sum` and make `gammapy.datasets.Dataset.stat_sum_likelihood` and `gammapy.datasets.Datasets.stat_sum_likelihood` public. [`#6812 <https://github.com/gammapy/gammapy/issues/6812>`_]
- Add `~gammapy.makers.OnOffBackgroundMaker` for on-off analysis. [`#6824 <https://github.com/gammapy/gammapy/issues/6824>`_]
- Creation of class `~gammapy.astro.darkmatter.DarkMatterSpectralModel` as the unification of `~gammapy.astro.darkmatter.DarkMatterAnnihilationSpectralModel` and `~gammapy.astro.darkmatter.DarkMatterDecaySpectralModel`. The new class has a parameter `annihilation` to set the type of dark matter process. [`#6825 <https://github.com/gammapy/gammapy/issues/6825>`_]
- Add ASDF serialization support for `~gammapy.maps.Maps`. [`#6830 <https://github.com/gammapy/gammapy/issues/6830>`_]
- Add ASDF serialization support for `~gammapy.data.GTI`. [`#6831 <https://github.com/gammapy/gammapy/issues/6831>`_]
- Add ASDF serialization support for `~gammapy.irf.PSFMap`, `~gammapy.irf.RecoPSFMap`, `~gammapy.irf.EDispMap`, `~gammapy.irf.EDispKernelMap`. [`#6833 <https://github.com/gammapy/gammapy/issues/6833>`_]
- Add Fourth Fermi-LAT Catalog of High-Energy Sources (4FHL).
  Older 2FHL catalog has been deprecated as of next release (v2.2). [`#6838 <https://github.com/gammapy/gammapy/issues/6838>`_]
- Add `~gammapy.visualization.plotters.BasePlotter` class, first building plot for the new Plotters API defined in PIG 31. [`#6845 <https://github.com/gammapy/gammapy/issues/6845>`_]
- Add ASDF serialization support for `~gammapy.datasets.MapDataset` and `~gammapy.datasets.MapDatasetOnOff`. [`#6849 <https://github.com/gammapy/gammapy/issues/6849>`_]
- Add ASDF serialization support for `~gammapy.modeling.models.Models`. [`#6850 <https://github.com/gammapy/gammapy/issues/6850>`_]
- Adapt the default spectra source in `~gammapy.astro.darkmatter.DarkMatterSpectralModel` and `~gammapy.astro.darkmatter.ContinuumPrimaryFlux` from PPPC4DMID to CosmiXs. [`#6885 <https://github.com/gammapy/gammapy/issues/6885>`_]
- Add Eemin, Eemax (for leptonic models) or Epmin, Epmax (for hadronic models) as fit parameters in the `~gammapy.modeling.models.NaimaSpectralModel`s. [`#6889 <https://github.com/gammapy/gammapy/issues/6889>`_]
- Remove the limit of 5 non-spatial axes when reading a `~gammapy.maps.RegionNDMap` with gadf format. [`#6949 <https://github.com/gammapy/gammapy/issues/6949>`_]


Bug Fixes
---------

- Correct `gammapy.datasets.FluxPointsDataset.plot_residuals()` x-axis unit, remove ``ax_spectrum`` and ``ax_residuals`` parameter in `gammapy.datasets.FluxPointsDataset.plot_fit()` and adjust documentation. [`#6615 <https://github.com/gammapy/gammapy/issues/6615>`_]
- `~gammapy.datasets.MapDataset.stack(other)` now skips stacking if other ``mask_safe`` is invalid. This prevents stacking gti of invalid dataset. [`#6621 <https://github.com/gammapy/gammapy/issues/6621>`_]
- Fixes `~gammapy.modeling.models.integrate_spectrum` to allow integration of spectral models without a closed analytical form (e.g. `~gammapy.modeling.models.BrokenPowerLawSpectralModel`) with energy bounds given as arrays of length 1. [`#6622 <https://github.com/gammapy/gammapy/issues/6622>`_]
- Fixes bug to return ``ax`` object for ``TemplateSpatialModel.plot()``. [`#6636 <https://github.com/gammapy/gammapy/issues/6636>`_]
- Fixes a bug in the `~gammapy.estimators.EnergyDependentMorphologyEstimator` to allow for multiple datasets to be handled correctly. [`#6640 <https://github.com/gammapy/gammapy/issues/6640>`_]
- Corrects `~gammapy.makers.utils.make_effective_livetime_map` to use observation pointing to compute livetime map. [`#6643 <https://github.com/gammapy/gammapy/issues/6643>`_]
- Fixes `~gammapy.irf.EffectiveAreaTable2D.plot_offset_dependence()` to remove area being divided twice. [`#6666 <https://github.com/gammapy/gammapy/issues/6666>`_]
- Fix default behavior of `sum_over_energy_groups=False` case in `gammapy.estimators.ExcessMapEstimator`. [`#6685 <https://github.com/gammapy/gammapy/issues/6685>`_]
- Remove dependency to regions deprecated function pixel_scale_angle_at_skycoord in `~gammapy.utils.regions.SkyPolygonRegion`. [`#6705 <https://github.com/gammapy/gammapy/issues/6705>`_]
- Add configuration of interpolation options for  `~gammapy.modeling.models.LightCurveTemplateTemporalModel`, now default is to return zero when evaluating outside the times in the template. [`#6712 <https://github.com/gammapy/gammapy/issues/6712>`_]
- Fixes incorrect calling of dataset for frozen models in `~gammapy.estimators.FluxCollectionEstimator`. [`#6721 <https://github.com/gammapy/gammapy/issues/6721>`_]
- Add unit support to the violins plot from `gammapy.utils.visualization.plot_samples_violin_vs_energy` used with the `~gammapy.estimators.FluxCollectionEstimator` [`#6756 <https://github.com/gammapy/gammapy/issues/6756>`_]
- Fixes `~gammapy.astro.darkmatter.JFactory.compute_differential_jfactor` to use the physical impact parameter in the line-of-sight integration and avoid evaluating the integrable closest-approach singularity directly. [`#6757 <https://github.com/gammapy/gammapy/issues/6757>`_]
- Fix Asimov dataset computation for `~gammapy.datasets.MapDatasetOnOff`: use alpha * N_OFF as expected background. [`#6762 <https://github.com/gammapy/gammapy/issues/6762>`_]
- Fix the `~gammapy.estimators.FluxCollectionEstimator` to support computation over datasets with different spatial geometries, and single flux point. [`#6785 <https://github.com/gammapy/gammapy/issues/6785>`_]
- Fix `~gammapy.maps.RegionGeom` ASDF roundtrip for DS9-formatted regions to clear ``meta`` and ``visual`` on write. [`#6801 <https://github.com/gammapy/gammapy/issues/6801>`_]
- Fix parsing of the `scale` parameter when read from a dictionary in `~gammapy.modeling.models.TemplatePhaseCurveTemporalModel`. [`#6807 <https://github.com/gammapy/gammapy/issues/6807>`_]
- Fix broadcasting in `~gammapy.modeling.models.TemplatePhaseCurveTemporalModel.integral` [`#6813 <https://github.com/gammapy/gammapy/issues/6813>`_]
- Fixes the line-of-sight integration geometry in `~gammapy.astro.darkmatter.JFactory` to account for observers inside or outside the radial integration region. [`#6819 <https://github.com/gammapy/gammapy/issues/6819>`_]
- Fixes `~gammapy.astro.darkmatter.JFactory` when a map pixel center coincides exactly with the dark-matter halo center. [`#6828 <https://github.com/gammapy/gammapy/issues/6828>`_]
- `~gammapy.maps.MapAxis.format_plot_xaxis` no longer produces overlapping scientific-notation minor tick labels on a log x-axis spanning less than one decade. [`#6847 <https://github.com/gammapy/gammapy/issues/6847>`_]
- Makes sure FoV offsets in `_get_fov_coord` are calculated in the correct coordinate frame. [`#6861 <https://github.com/gammapy/gammapy/issues/6861>`_]
- Fix `nan` in sensitivity computation using Asimov dataset. [`#6862 <https://github.com/gammapy/gammapy/issues/6862>`_]
- Add `gammapy.stats.utils.ts_to_sigma_sensitivity` and `gammapy.stats.utils.sigma_to_ts_sensitivity` as separate functions and fix computation with more than one degree of freedom [`#6864 <https://github.com/gammapy/gammapy/issues/6864>`_]
- Fixes an out-of-memory error when computing the covariance of models with a large number of parameters (e.g. a full source catalogue) by building diagonal covariances lazily in `gammapy.modeling.covariance.Covariance`. [`#6867 <https://github.com/gammapy/gammapy/issues/6867>`_]
- Fixes `~gammapy.modeling.models.LogParabola2SpectralModel.e_peak` value for the case where the energy scale E_s is different from the reference energy E_0. [`#6884 <https://github.com/gammapy/gammapy/issues/6884>`_]
- Fixes `~gammapy.irf.PSF3D.containment` when rad is larger than the last bin of the PSF rad axis. [`#6887 <https://github.com/gammapy/gammapy/issues/6887>`_]
- Recompute upper limits using likelihood profiles if the default method using fit condidence fails in `gammapy.estimators.ParameterEstimator`. [`#6900 <https://github.com/gammapy/gammapy/issues/6900>`_]
- Fixes error in getting the `pivot_energy` for a `~gammapy.modeling.models.CompoundSpectralModel` or model which does not contain a `reference`. [`#6916 <https://github.com/gammapy/gammapy/issues/6916>`_]
- Fix error in `gammapy.data.ivoa.make_hdu_table`. [`#6920 <https://github.com/gammapy/gammapy/issues/6920>`_]
- Fix ``to_dict()`` of `~gammapy.modeling.models.TemplateSpectralModel`, `~gammapy.modeling.models.PiecewiseNormSpectralModel`, `~gammapy.modeling.models.EBLAbsorptionNormSpectralModel` and `~gammapy.modeling.models.PiecewiseNormSpatialModel` failing on big-endian arrays. [`#6952 <https://github.com/gammapy/gammapy/issues/6952>`_]
- Fix the normalisation of `~gammapy.stats.compute_fpp` to match Edelson et al. (2002), also when the light curve has NaN values. [`#7014 <https://github.com/gammapy/gammapy/issues/7014>`_]


Known issues
------------


Contributors
------------
