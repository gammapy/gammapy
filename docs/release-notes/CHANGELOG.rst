.. include:: ../references.txt

.. _gammapy_2p0p2_release:

2.0.2 (27th November, 2026)
===========================

- Released 27th November, 2026
- contributors
- pull requests since v (not all listed below)
- closed issues

Summary
-------


API changes
-----------


Infrastructure
--------------

- Move the package configuration from ``setup.cfg`` to ``pyproject.toml`` as recommended by PEP621. [`#6176 <https://github.com/gammapy/gammapy/issues/6176>`_]


Documentation improvements
--------------------------

- Clarify usage of `~gammapy.maps.RegionGeom.from_regions` regarding underlying projection effects. [`#6272 <https://github.com/gammapy/gammapy/issues/6272>`_]
- Remove unused ``obstime`` and ``location`` variables from `~gammapy.utils.coordinates.FoVICRSFrame` docstring example. [`#6460 <https://github.com/gammapy/gammapy/issues/6460>`_]
- Fix issue in Fermi-LAT tutorial where dimmest sources are used for fitting rather than brightest. [`#6750 <https://github.com/gammapy/gammapy/issues/6750>`_]


New features
------------


Bug Fixes
---------

- Fix issue #5783, the `~gammapy.estimators.FluxPoints.to_table` method now always outputs standard flux units. [`#6108 <https://github.com/gammapy/gammapy/issues/6108>`_]
- Adapted the ``size_factor`` default value to be consistent for all spatial models. [`#6137 <https://github.com/gammapy/gammapy/issues/6137>`_]
- Extra sensitivity options were added to `~gammapy.estimators.FluxMaps`. The additional options we now support are: "dnde_sensitivity", "e2dnde_sensitivity", "eflux_sensitivity". [`#6141 <https://github.com/gammapy/gammapy/issues/6141>`_]
- Fixed `gammapy.irf.EDispKernel.plot_bias` so the correct axis is now being called for the ``xaxis.units``. [`#6159 <https://github.com/gammapy/gammapy/issues/6159>`_]
- Fixed the incorrect brackets in the error calculation for `~gammapy.stats.compute_flux_doubling`. [`#6164 <https://github.com/gammapy/gammapy/issues/6164>`_]
- Corrected `~gammapy.modeling.models.TemplatePhaseCurveTemporalModel` integration by performing global normalisation by the duration of the integration window as done in the other temporal models. [`#6182 <https://github.com/gammapy/gammapy/issues/6182>`_]
- Replaced the deprecated keyword ``RADECSYS`` for the coordinate system in the GADF event list fits table with ``RADESYSa``. [`#6189 <https://github.com/gammapy/gammapy/issues/6189>`_]
- Ignore observations without spatial overlap with the target dataset geometry in `~gammapy.makers.DatasetsMaker` instead of raising an error. [`#6194 <https://github.com/gammapy/gammapy/issues/6194>`_]
- Corrected SkyModel `_check_unit` unit requirement for models where exposure is not applied. [`#6227 <https://github.com/gammapy/gammapy/issues/6227>`_]
- Modify default interpolation conditions in `~gammapy.modeling.models.TemplateSpatialModel` to allow using Hpx geometry maps. [`#6248 <https://github.com/gammapy/gammapy/issues/6248>`_]
- Fixed ``plot_error`` for `~gammapy.modeling.models.BrokenPowerLawSpectralModel` and `~gammapy.modeling.models.TemplateNDSpectralModel` [`#6251 <https://github.com/gammapy/gammapy/issues/6251>`_]
- Fix ``plot_error`` for models evaluated with non-finite values. [`#6252 <https://github.com/gammapy/gammapy/issues/6252>`_]
- Fix the addition logic so that `gammapy.modeling.models.Models` and `gammapy.modeling.models.DatasetModels`  can be added together regardless of the order they are given. [`#6253 <https://github.com/gammapy/gammapy/issues/6253>`_]
- Fix map stacking if data type is not float. [`#6254 <https://github.com/gammapy/gammapy/issues/6254>`_]
- Adapted `SpectrumDataset.create()` to avoid creating PSFMap during dataset creation. Moved creation of `SpectrumDataset.create()` into the `SpectrumDataset` class. Added check for `RegionGeom` of given geometry. [`#6259 <https://github.com/gammapy/gammapy/issues/6259>`_]
- Add an optional normalize argument on `~gammapy.modeling.models.TemplatePhaseCurveTemporalModel`. It is set by default to True, for backward compatibility with 2.0. [`#6270 <https://github.com/gammapy/gammapy/issues/6270>`_]
- Fixed kinematics in `gammapy.astro.darkmatter.DarkMatterDecaySpectralModel`. The model now correctly uses mass/2 for the primary flux lookup. [`#6298 <https://github.com/gammapy/gammapy/issues/6298>`_]
- Correct issue #6304. Include CITATION file in the gammapy directory. Modify the `_get_bibtex` and `_get_acknowledgement` functions accordingly. [`#6308 <https://github.com/gammapy/gammapy/issues/6308>`_]
- Add default `~astropy.modeling.Parameter` values on init in gammapy.astro models. This solves docs build issues which appeared recently. [`#6333 <https://github.com/gammapy/gammapy/issues/6333>`_]
- Correct bug with command-line-interface (``gammapy analysis run``) that forces background maker method to be reflected regions. [`#6367 <https://github.com/gammapy/gammapy/issues/6367>`_]
- Fix an off-by-one error in `~gammapy.maps.WcsGeom.pix_to_idx` by clipping indices to npix - 1. [`#6383 <https://github.com/gammapy/gammapy/issues/6383>`_]
- Fix ``hpx.geom.pix_to_idx`` to correctly handle non-spatial pixel coordinates, ensuring consistency with WCS. [`#6388 <https://github.com/gammapy/gammapy/issues/6388>`_]
- ``requests`` and ``tqdm`` are now included as core dependencies so that ``gammapy download datasets`` works out of the box without requiring users to install extra packages manually. [`#6400 <https://github.com/gammapy/gammapy/issues/6400>`_]
- `~gammapy.modeling.Parameter` bounds are now synchronized dynamically with `~gammapy.modeling.models.UniformPrior` and `~gammapy.modeling.models.LogUniformPrior` bounds to prevent Minuit from hitting infinite likelihood. [`#6409 <https://github.com/gammapy/gammapy/issues/6409>`_]
- Fix `gammapy.datasets.utils.set_and_restore_mask_fit` such as it does not ignore the existing dataset.mask_fit by default. [`#6416 <https://github.com/gammapy/gammapy/issues/6416>`_]
- Fix a ``RuntimeWarning`` due to division by zero in `~gammapy.irf.EffectiveAreaTable2D.plot_offset_dependence` when the maximum effective area is zero. [`#6450 <https://github.com/gammapy/gammapy/issues/6450>`_]
- Fix flux estimates in `gammapy.estimators.TSMapEstimator` (now the kernel is properly normalized in each energy bin rather than over the total energy range). [`#6466 <https://github.com/gammapy/gammapy/issues/6466>`_]
- Fix ``select_time`` in `~gammapy.data.EventList`, `~gammapy.data.GTI` and `~gammapy.data.Observation` to always utilise start time (inclusive) and stop time (exclusive). [`#6497 <https://github.com/gammapy/gammapy/issues/6497>`_]
- Fix unit handling in RegionNDMap.plot_mask [`#6520 <https://github.com/gammapy/gammapy/issues/6520>`_]
- Propagated the ``overwrite`` argument in the ``write`` method of `~gammapy.modeling.models.TemplateNDSpectralModel`. [`#6525 <https://github.com/gammapy/gammapy/issues/6525>`_]
- Introduce default threshold value to remove numerical artifacts created when applying `~gammapy.irf.EnergyDispersion2D.to_edisp_kernel()`. [`#6547 <https://github.com/gammapy/gammapy/issues/6547>`_]
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
- Fixes `~gammapy.astro.darkmatter.JFactory.compute_differential_jfactor` to use the physical impact parameter in the line-of-sight integration and avoid evaluating the integrable closest-approach singularity directly. [`#6757 <https://github.com/gammapy/gammapy/issues/6757>`_]
- Fix Asimov dataset computation for `~gammapy.datasets.MapDatasetOnOff`: use alpha * N_OFF as expected background. [`#6762 <https://github.com/gammapy/gammapy/issues/6762>`_]
- Fix parsing of the `scale` parameter when read from a dictionary in `~gammapy.modeling.models.TemplatePhaseCurveTemporalModel`. [`#6807 <https://github.com/gammapy/gammapy/issues/6807>`_]
- Fix broadcasting in `~gammapy.modeling.models.TemplatePhaseCurveTemporalModel.integral` [`#6813 <https://github.com/gammapy/gammapy/issues/6813>`_]
- Makes sure FoV offsets in `_get_fov_coord` are calculated in the correct coordinate frame. [`#6861 <https://github.com/gammapy/gammapy/issues/6861>`_]
- Fix `nan` in sensitivity computation using Asimov dataset. [`#6862 <https://github.com/gammapy/gammapy/issues/6862>`_]
- Fixes an out-of-memory error when computing the covariance of models with a large number of parameters (e.g. a full source catalogue) by building diagonal covariances lazily in `gammapy.modeling.covariance.Covariance`. [`#6867 <https://github.com/gammapy/gammapy/issues/6867>`_]
- Fixes `~gammapy.irf.PSF3D.containment` when rad is larger than the last bin of the PSF rad axis. [`#6887 <https://github.com/gammapy/gammapy/issues/6887>`_]
- Recompute upper limits using likelihood profiles if the default method using fit condidence fails in `gammapy.estimators.ParameterEstimator`. [`#6900 <https://github.com/gammapy/gammapy/issues/6900>`_]
- Fixes error in getting the `pivot_energy` for a `~gammapy.modeling.models.CompoundSpectralModel` or model which does not contain a `reference`. [`#6916 <https://github.com/gammapy/gammapy/issues/6916>`_]
- Fix ``to_dict()`` of `~gammapy.modeling.models.TemplateSpectralModel`, `~gammapy.modeling.models.PiecewiseNormSpectralModel`, `~gammapy.modeling.models.EBLAbsorptionNormSpectralModel` and `~gammapy.modeling.models.PiecewiseNormSpatialModel` failing on big-endian arrays. [`#6952 <https://github.com/gammapy/gammapy/issues/6952>`_]
- Fix the normalisation of `~gammapy.stats.compute_fpp` to match Edelson et al. (2002), also when the light curve has NaN values. [`#7014 <https://github.com/gammapy/gammapy/issues/7014>`_]
- Correct location of HEGRA array in ~gammapy.utils.observers.py [`#7068 <https://github.com/gammapy/gammapy/issues/7068>`_]
- Modify handling of GADF keyword WCSSHAPE in `~gammapy.maps.WcsGeom` to validate its content. [`#7087 <https://github.com/gammapy/gammapy/issues/7087>`_]


Known issues
------------


Contributors
------------

.. include:: ../references.txt

.. _gammapy_2p0px_release:

2.0.x (27th November, 2026)
===========================

- Released 27th November, 2026
- contributors
- pull requests since v (not all listed below)
- closed issues

Summary
-------


API changes
-----------


Infrastructure
--------------

- Move the package configuration from ``setup.cfg`` to ``pyproject.toml`` as recommended by PEP621. [`#6176 <https://github.com/gammapy/gammapy/issues/6176>`_]


Documentation improvements
--------------------------

- Clarify usage of `~gammapy.maps.RegionGeom.from_regions` regarding underlying projection effects. [`#6272 <https://github.com/gammapy/gammapy/issues/6272>`_]
- Remove unused ``obstime`` and ``location`` variables from `~gammapy.utils.coordinates.FoVICRSFrame` docstring example. [`#6460 <https://github.com/gammapy/gammapy/issues/6460>`_]
- Fix issue in Fermi-LAT tutorial where dimmest sources are used for fitting rather than brightest. [`#6750 <https://github.com/gammapy/gammapy/issues/6750>`_]


New features
------------


Bug Fixes
---------


Known issues
------------


Contributors
------------

.. include:: ../references.txt

.. _gammapy_2p0p2_release:

2.0.2 (27th November, 2026)
===========================

- Released 27th November, 2026
- contributors
- pull requests since v (not all listed below)
- closed issues

Summary
-------


API changes
-----------


Infrastructure
--------------

- Move the package configuration from ``setup.cfg`` to ``pyproject.toml`` as recommended by PEP621. [`#6176 <https://github.com/gammapy/gammapy/issues/6176>`_]


Documentation improvements
--------------------------

- Clarify usage of `~gammapy.maps.RegionGeom.from_regions` regarding underlying projection effects. [`#6272 <https://github.com/gammapy/gammapy/issues/6272>`_]
- Remove unused ``obstime`` and ``location`` variables from `~gammapy.utils.coordinates.FoVICRSFrame` docstring example. [`#6460 <https://github.com/gammapy/gammapy/issues/6460>`_]
- Fix issue in Fermi-LAT tutorial where dimmest sources are used for fitting rather than brightest. [`#6750 <https://github.com/gammapy/gammapy/issues/6750>`_]


New features
------------


Bug Fixes
---------


Known issues
------------


Contributors
------------
