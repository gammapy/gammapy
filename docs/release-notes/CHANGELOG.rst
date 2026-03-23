.. include:: ../references.txt

.. _gammapy_2p1_release:

2.1 (xxrd March, 2026)
======================
- Released xxrd March, 2026
- 63 pull requests
- 16 closed issues
- 22 unique contributors


Summary
-------

Gammapy v2.1 is the first feature release since LTS v2.0.
This release includes a number of new features, bug fixes, infrastructural changes and documentation improvements.


API changes
-----------

- Change the argument name from ``model`` to ``kernel_model`` in `~gammapy.estimators.TSMapEstimator` for clarity. [`#6510 <https://github.com/gammapy/gammapy/issues/6510>`_]


Infrastructure
--------------

- Integrate SonarQube checks for quality assurance as required by CTAO standards. [`#6150 <https://github.com/gammapy/gammapy/issues/6150>`_]
- Move the package configuration from ``setup.cfg`` to ``pyproject.toml`` as recommended by PEP621. [`#6176 <https://github.com/gammapy/gammapy/issues/6176>`_]


Documentation improvements
--------------------------

- Clarify usage of `~gammapy.maps.RegionGeom.from_regions` regarding underlying projection effects. [`#6272 <https://github.com/gammapy/gammapy/issues/6272>`_]
- Remove unused ``obstime`` and ``location`` variables from `~gammapy.utils.coordinates.FoVICRSFrame` docstring example. [`#6460 <https://github.com/gammapy/gammapy/issues/6460>`_]


New features
------------

- Switch default SpectrumDataset write/read format to gadf instead of ogip (for better preservation of masks) [`#5812 <https://github.com/gammapy/gammapy/issues/5812>`_]
- Add a FitStatisticPenalty class to support constraints on multiple parameters e.g. regularity constraints, multiparametric priors [`#5838 <https://github.com/gammapy/gammapy/issues/5838>`_]
- Add a log parabola model defined such as the energy scale of the exponent and the reference energy can be different. [`#6147 <https://github.com/gammapy/gammapy/issues/6147>`_]
- Add `axis_name` argument to `~gammapy.estimators.FluxPoints.plot_ts_profiles` to allow the user to select the axis to plot. [`#6197 <https://github.com/gammapy/gammapy/issues/6197>`_]
- Implementation of the option of using ``CosmiXs`` or ``PPPC4`` as a source models for creating dark matter spectra. [`#6232 <https://github.com/gammapy/gammapy/issues/6232>`_]
- Add sample_parameters_from_covariance method on models allowing to create parameters samples from covariance using multivariate normal distribution. [`#6255 <https://github.com/gammapy/gammapy/issues/6255>`_]
- Enable the addition of `~gammapy.datasets.Datasets` and individual `~gammapy.datasets.Dataset` objects to create a new `~gammapy.datasets.Datasets` instance. [`#6256 <https://github.com/gammapy/gammapy/issues/6256>`_]
- Add a new `~gammapy.estimators.FluxCollectionEstimator` that enables the computation of flux points or flux samples jointly for a group of sources. [`#6322 <https://github.com/gammapy/gammapy/issues/6322>`_]
- Add an boolean argument `allow_multiple_telescopes` which allows the user to compute `~gammapy.estimators.FluxPointsEstimator` for multiple telescopes. [`#6339 <https://github.com/gammapy/gammapy/issues/6339>`_]
- Replace `~gammapy.makers.utils.make_theta_squared_table` with `~gammapy.makers.utils.ThetaSquaredTable` class. Overlapping regions are now forbidden. User can define multiple non-overlapping OFF regions. [`#6348 <https://github.com/gammapy/gammapy/issues/6348>`_]
- Add GTI to datasets created by `~gammapy.datasets.FermipyDatasetsReader`. [`#6378 <https://github.com/gammapy/gammapy/issues/6378>`_]
- Make LogScale clipping threshold adaptive to input dtype. This allows interpolation on float64 arrays without hard clipping at the float32 threshold (~1e-38). [`#6387 <https://github.com/gammapy/gammapy/issues/6387>`_]
- Replaced the slow `.coadd()` fallback in `WcsNDMap.crop` and `WcsNDMap.pad` with direct dynamic tensor slicing, drastically improving performance for irregular non-spatial geometries. [`#6428 <https://github.com/gammapy/gammapy/issues/6428>`_]
- Support chained comparisons in ``gammapy.utils.scripts.logic_parser`` (e.g., ``1 < x < 3``). [`#6467 <https://github.com/gammapy/gammapy/issues/6467>`_]


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
- Improve parsing of config file in FermipyDatasetsReader.read() and raise more explicit errors for invalid cases. [`#6421 <https://github.com/gammapy/gammapy/issues/6421>`_]
- Fix a ``RuntimeWarning`` due to division by zero in `~gammapy.irf.EffectiveAreaTable2D.plot_offset_dependence` when the maximum effective area is zero. [`#6450 <https://github.com/gammapy/gammapy/issues/6450>`_]
- Fix flux estimates in `gammapy.estimators.TSMapEstimator` (now the kernel is properly normalized in each energy bin rather than over the total energy range). [`#6466 <https://github.com/gammapy/gammapy/issues/6466>`_]
- Fix ``select_time`` in `~gammapy.data.EventList`, `~gammapy.data.GTI` and `~gammapy.data.Observation` to always utilise start time (inclusive) and stop time (exclusive). [`#6497 <https://github.com/gammapy/gammapy/issues/6497>`_]
- Fix unit handling in RegionNDMap.plot_mask [`#6520 <https://github.com/gammapy/gammapy/issues/6520>`_]




Contributors
------------
- Arnau Aguasca-Cabot
- Atreyee Sinha
- Axel Donath
- Bruno Khélifi
- Daniel Morcuende
- Ebraam
- Fabio Acero
- Fabio PINTORE
- Gabriel Emery
- Kirsty Feijen
- Leander Schlegel
- Marie-Sophie Carrasco
- Mireia Nievas-Rosillo
- Natthan Pigoux
- Quentin Remy
- Régis Terrier
- Stefan Fröse
- Thomas Vuillaume
- Tomas Bylund
- Tora T. H. Arnesen
- rcervinoucm
- yaochengchen
