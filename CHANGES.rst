.. _gammapy_0p15_release:

0.15 (unreleased)
-----------------

- Planned for Nov 2019

.. _gammapy_0p14_release:

0.14 (Sep 30, 2019)
-------------------

Summary
+++++++

- Released Sep 30, 2019
- 8 contributors
- 101 pull requests (not all listed below)

**What's new**

Gammapy v0.14 features a new high level analysis interface. Starting from
a YAML configuration file, it supports the standard use-cases of joint
or stacked 3D as well as 1D reflected region analyses. It also supports
computation of flux points for all cases. The usage of this new ``Analysis``
class is demonstrated in the `hess.html <./notebooks/hess.html>`__ tutorial.

Following the proposal in :ref:`pig-016` the subpackages ``gammapy.background``
and ``gammapy.image`` were removed. Existing functionality was moved to the
``gammapy.cube`` and ``gammapy.spectrum`` subpackages.

A new subpackage ``gammapy.modeling`` subpackage as introduced. All spectral,
spatial, temporal and combined models were moved to the new namespace and
renamed following a consistent naming scheme. This provides a much clearer
structure of the model types and hierarchy for users.

The ``SkyEllipse`` model was removed. Instead the ``GaussianSpatialModel``
as well as the ``DiskSpatialModel`` now support parameters for
elongation. A bug that lead to an incorrect flux normalization of the
``PointSpatialModel`` at high latitudes was fixed. The default coordinate
frame for all spatial models was changed to ``icrs``. A new
``ConstantTemporalModel`` was introduced.

A new ``MapDataset.to_spectrum_dataset()`` method allows to reduce a map
dataset to a spectrum dataset in a specified analysis region. The
``SpectrumDatasetOnOffStacker`` was removed and placed by a ``SpectrumDatasetOnOff.stack()``
and ``Datasets.stack_reduce()`` method. A ``SpectrumDataset.stack()``
method was also added.

Following :ref:`pig-013` the support for Python 3.5 was dropped with Gammapy v0.14.
At the same time the versions of the required dependencies were updated to
Numpy 1.16, Scipy 1.2, Astropy 3.2, Regions 0.5, Pyyaml 5.1, Click 7.0 and
Jsonschema 3.0.

**Contributors:**

In alphabetical order by first name:

- Atreyee Sinha
- Axel Donath
- Christoph Deil
- Régis Terrier
- Fabio Pintore
- Quentin Remy
- José Enrique Ruiz
- Johannes King
- Luca Giunti
- Lea Jouvin

Pull Requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy v0.14 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=✓&q=is%3Apr+milestone%3A0.14>`__.

- [#2412] Remove model XML serialization (Quentin Remy)
- [#2404] Clean up spectral model names (Christoph Deil)
- [#2401] Clean up spatial model names (Christoph Deil)
- [#2400] Clean up temporal model names (Christoph Deil)
- [#2385] Change spatial model default frame to icrs (Christoph Deil)
- [#2381] Add ``MapDataset.stack()``  (Atreyee Sinha)
- [#2379] Cleanup ``WcsNDMap`` FITS convention handling (Axel Donath)
- [#2378] Add support for 3D analysis in the high-level interface (José Enrique Ruiz)
- [#2377] Implement ``WcsGeom`` coord caching (Axel Donath)
- [#2375] Adapt ``MapMakerObs`` to return a ``MapDataset`` (Atreyee Sinha)
- [#2368] Add ``MapDataset.create()`` method (Atreyee Sinha)
- [#2367] Fix SkyPointSource evaluation (Christoph Deil)
- [#2366] Remove lon wrapping in spatial models (Christoph Deil)
- [#2365] Remove gammapy/maps/measure.py (Christoph Deil)
- [#2360] Add ``SpectrumDatasetOnOff.stack()`` (Régis Terrier)
- [#2359] Remove ``BackgroundModels`` class (Axel Donath)
- [#2358] Adapt MapMakerObs to also compute an EDispMap and PSFMap (Atreyee Sinha)
- [#2356] Add ``SpectrumDataset.stack()`` (Régis Terrier)
- [#2354] Move gammapy.utils.fitting to gammapy.modeling (Christoph Deil)
- [#2351] Change OrderedDict to dict  (Christoph Deil)
- [#2347] Simplify ``EdispMap.stack()`` and ``PsfMap.stack()`` (Luca Giunti)
- [#2346] Add ``SpectrumDatasetOnOff.create()`` (Régis Terrier)
- [#2345] Add ``SpectrumDataset.create()`` (Régis Terrier)
- [#2344] Change return type of ``WcsGeom.get_coord()`` to quantities (Axel Donath)
- [#2343] Implement ``WcsNDMap.sample()`` and remove ``MapEventSampler`` (Fabio Pintore)
- [#2342] Add zero clipping in ``MapEvaluator.apply_psf`` (Luca Giunti)
- [#2338] Add model registries and ``Model.from_dict()`` method (Quentin Remy)
- [#2335] Remove ``SpectrumAnalysisIACT`` class (José Enrique Ruiz)
- [#2334] Simplify and extend background model handling (Axel Donath)
- [#2330] Migrate SpectrumAnalysisIACT to the high-level interface (José Enrique Ruiz)
- [#2326] Fix bug in the spectral gaussian model evaluate method (Lea Jouvin)
- [#2323] Add high-level Config and Analysis classes (José Enrique Ruiz)
- [#2321] Dissolve ``gammapy.image`` (Christoph Deil)
- [#2320] Dissolve ``gammapy.background`` (Christoph Deil)
- [#2314] Add datasets serialization (Quentin Remy)
- [#2313] Add elongated gaussian model (Luca Giunti)
- [#2308] Use parfive in gammapy download (José Enrique Ruiz)
- [#2292] Implement ``MapDataset.to_spectrum_dataset()`` method (Régis Terrier)
- [#2279] Update Gammapy packaging, removing astropy-helpers (Christoph Deil)
- [#2274] PIG 16 - Gammapy package structure (Christoph Deil)
- [#2219] PIG 12 - High-level interface (José Enrique Ruiz)
- [#2218] PIG 13 - Gammapy dependencies and distribution (Christoph Deil)
- [#2136] PIG 9 - Event sampling (Fabio Pintore)

.. _gammapy_0p13_release:

0.13 (Jul 26, 2019)
-------------------

Summary
+++++++

- Released Jul 26, 2019
- 15 contributors
- 2 months of work
- 72 pull requests (not all listed below)

**What's new**

The Gammapy v0.13 release includes many bug-fixes, a lot of clean-up work
and some new features.

Gammapy v0.13 implements a new ``SpectralGaussian`` and ``PLSuperExpCutoff4FGL``
model. To support binned simulation of counts data in a uniform
way ``MapDataset.fake()``, ``SpectrumDataset.fake()`` and ``SpectrumDatasetOnOff.fake()``
methods were implemented, which simulate binned counts maps and spectra from models.
In addition a nice string representations for all of the dataset classes was implemented
together with convenience functions to compute residuals using different methods on all
of them. The algorithm and API of the current ``LightCurveEstimator`` was changed to
use datasets. Now it is possible to compute lightcurves using spectral as well
as cube based analyses. The definition of the position angle of the ``SkyEllipse`` model
was changed to follow IAU conventions.

The handling of sky regions in Gammapy was unified as described in `PIG 10`_.
For convenience regions can now also be created from DS9 region strings. The clean-up
process of ``gammapy.spectrum`` was continued by removing the ``PHACountsSpectrum``
class, which is now fully replaced by the ``SpectrumDatasetOnOff`` class. The
``Energy`` and ``EnergyBounds`` classes were also removed. Grids of energies can be
created and handled directly using the ``MapAxis`` object now.

The algorithm to compute solid angles for maps was fixed, so that it gives correct
results for WCS projections even with high spatial distortions. Standard analyses
using TAN or CAR projections are only affected on a <1% level. Different units
for the energy axis of the counts and exposure map in a ``MapDataset`` are now
handled correctly.

The recommended conda environment for Gammapy v0.13 was updated. It now relies
on Python 3.7, Ipython 7.5, Scipy 1.3, Matplotlib 3.1, Astropy 3.1, and Healpy 1.12.
These updates should be backwards compatible. Sripts and notebooks should
run and give the same results.

**Contributors:**

In alphabetical order by first name:

- Atreyee Sinha
- Axel Donath
- Brigitta Sipocz
- Bruno Khelifi
- Christoph Deil
- Fabio Pintore
- Fabio Acero
- Kaori Nakashima
- José Enrique Ruiz
- Léa Jouvin
- Luca Giunti
- Quentin Remy
- Régis Terrier
- Silvia Manconi
- Yu Wun Wong

Pull Requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy v0.13 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=✓&q=is%3Apr+milestone%3A0.13+>`__.

- [#2296] Implement model YAML serialisation (Quentin Remy)
- [#2310] Remove old ``LightCurveEstimator`` class (Axel Donath)
- [#2305] Remove ``SpectrumSimulation`` class (Axel Donath)
- [#2300] Change to IAU convention for position angle in SkyEllipse model (Luca Giunti)
- [#2298] Implement ``.fake()`` methods on datasets (Léa Jouvin)
- [#2297] Implement Fermi 4FGL catalog spectral models and catalog (Kaori Nakashima & Yu Wun Wong)
- [#2294] Fix pulsar spin-down model bug (Silvia Manconi)
- [#2289] Add ``gammapy/utils/fitting/sampling.py`` (Fabio Acero)
- [#2287] Implement ``__str__`` methoda for dataset (Léa Jouvin)
- [#2278] Refactor class ``CrabSpectrum`` in a function (Léa Jouvin)
- [#2277] Implement GTI union (Régis Terrier)
- [#2276] Fix map pixel solid angle computation (Axel Donath)
- [#2272] Remove ``SpectrumStats`` class (Axel Donath)
- [#2264] Implement ``MapDataset`` FITS I/O (Axel Donath)
- [#2262] Clean up sky region select code (Christoph Deil)
- [#2259] Fix ``Fit.minos_contour`` method for frozen parameters  (Axel Donath)
- [#2257] Update astropy-helpers to v3.2.1 (Brigitta Sipocz)
- [#2254] Add select_region method for event lists (Régis Terrier)
- [#2250] Remove ``PHACountsSpectrum`` class (Axel Donath)
- [#2244] Implement ``SpectralGaussian`` model class (Léa Jouvin)
- [#2243] Speed up mcmc_sampling tutorial (Fabio Acero)
- [#2240] Remove use of NDDataArray from CountsSpectrum (Axel Donath)
- [#2239] Remove GeneralRandom class (Axel Donath)
- [#2238] Implement ``MapEventSampler`` class (Fabio Pintore)
- [#2237] Remove ``Energy`` and ``EnergyBounds`` classes (Axel Donath)
- [#2235] Remove unused functions in stats/data.py (Régis Terrier)
- [#2230] Improve spectrum/models.py coverage (Régis Terrier)
- [#2229] Implement ``InverseCDFSampler`` class (Fabio Pintore)
- [#2217] Refactor gammapy download (José Enrique Ruiz)
- [#2206] Remove unused map iter_by_pix and iter_by_coord methods (Christoph Deil)
- [#2204] Clean up ``gammapy.utils.random`` (Fabio Pintore)
- [#2200] Update astropy_helpers to v3.2 (Brigitta Sipocz)
- [#2192] Improve ``gammapy.astro`` code and tests (Christoph Deil)
- [#2129] PIG 10 - Regions (Christoph Deil)
- [#2089] Improve ``ReflectedRegionsFinder`` class (Bruno Khelifi)

.. _PIG 10: https://docs.gammapy.org/dev/development/pigs/pig-010.html

.. _gammapy_0p12_release:

0.12 (May 30, 2019)
-------------------

Summary
+++++++

- Released May 30, 2019
- 9 contributors
- 2 months of work
- 66 pull requests (not all listed below)

**What's new**

For Gammapy v0.12 we did our homework, cleaned up the basement and emptied the
trash bin. It is a maintenance release that does not introduce many new features,
but where we have put a lot of effort into integrating the ``gammapy.spectrum``
submodule into the datasets framework we introduced in the previous Gammapy version.
For this we replaced the former ``SpectrumObservation`` class by a new ``SpectrumDatasetOnOff``
class, which now works with the general ``Fit`` and ``Datasets`` objects in
``gammapy.utils.fitting``. This also enabled us to remove the ``SpectrumObservationList``
and ``SpectrumFit`` classes. We adapted the ``SpectrumExtraction`` class accordingly.
We also refactored the ``NDData`` class to use ``MapAxis`` to handle the data axes. This
affects the ``CountsSpectrum`` and the IRF classes in ``gammapy.irf``.

In addition we changed the ``FluxPointsEstimator`` to work with the new ``SpectrumDatasetOnOff``
as well as the ``MapDataset``. Now it is possible to compute flux points for 1D
as well 3D data with a uniform API. We added a new ``NaimaModel`` wrapper class (https://naima.readthedocs.io/),
which allows you to fit true physical, spectral models directly to counts based
gamma-ray data. To improve the fit convergence of the ``SkyDisk`` and ``SkyEllipse``
models we introduced a new parameter defining the slope of the edge of these models.

If you would like to know how to adapt your old spectral analysis scripts to Gammapy
v0.12, please checkout the updated tutorial notebooks (https://docs.gammapy.org/0.12/tutorials.html)
and `get in contact with us <https://gammapy.org/contact.html>`__ anytime if you need help.

**Contributors:**

In alphabetical order by first name:

- Atreyee Sinha
- Axel Donath
- Christoph Deil
- Dirk Lennarz
- Debanjan Bose (new)
- José Enrique Ruiz
- Lars Mohrmann
- Luca Giunti
- Régis Terrier

Pull Requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy v0.12 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=✓&q=is%3Apr+milestone%3A0.12+>`__.

- [#2171] Remove Poisson chi2 approximations (Christoph Deil)
- [#2169] Remove warning astropy_helpers.sphinx.conf is deprecated (José Enrique Ruiz)
- [#2166] Remove PHACountsSpectrumList class (Régis Terrier)
- [#2163] Fix integrate_spectrum for small integration ranges (Axel Donath)
- [#2160] Add default of "all" for DataStore.get_observations (Christoph Deil)
- [#2157] Rename SpectrumDataset.counts_on to SpectrumDataset.counts (Régis Terrier)
- [#2154] Implement DataStoreMaker for IACT DL3 indexing (Christoph Deil)
- [#2153] Remove SpectrumObservation and SpectrumObservationList classes (Régis Terrier)
- [#2152] Improve FluxPointEstimator for joint likelihood datasets (Axel Donath)
- [#2151] Add todo for improving wcs solid angle computation (Debanjan Bose)
- [#2146] Implement scipy confidence method (Axel Donath)
- [#2145] Make tests run without GAMMAPY_DATA (Christoph Deil)
- [#2142] Implement oversampling option for background model evaluation (Axel Donath)
- [#2141] Implement SkyDisk and SkyEllipse edge parameter (Axel Donath)
- [#2140] Clean up spectral tutorials (Atreyee Sinha)
- [#2139] Refactor SpectrumExtraction to use SpectrumDatasetOnOff (Régis Terrier)
- [#2133] Replace DataAxis and BinnedDataAxis classes by MapAxis (Axel Donath)
- [#2132] Change MapAxis.edges and MapAxis.center attributes to quantities (Atreyee Sinha)
- [#2131] Implement flux point estimation for MapDataset (Axel Donath)
- [#2130] Implement MapAxis.upsample() and MapAxis.downsample() methods (Axel Donath)
- [#2128] Fix Feldman-Cousins examples (Dirk Lennarz)
- [#2126] Fix sorting of node values in MapAxis (Atreyee Sinha)
- [#2124] Implement NaimaModel wrapper class (Luca Giunti)
- [#2123] Remove SpectrumFit class (Axel Donath)
- [#2121] Move plotting helper functions to SpectrumDatasetOnOff (Axel Donath)
- [#2119] Clean up Jupyter notebooks with PyCharm static code analysis (Christoph Deil)
- [#2118] Remove tutorials/astropy_introduction.ipynb (Christoph Deil)
- [#2115] Remove SpectrumResult object (Axel Donath)
- [#2114] Refactor energy grouping (Axel Donath)
- [#2112] Refactor FluxPointEstimator to use Datasets (Axel Donath)
- [#2111] Implement SpectrumDatasetOnOff class (Régis Terrier)
- [#2108] Fix frame attribute of SkyDiffuseCube and SkyDiffuseMap (Lars Mohrmann)
- [#2106] Add frame attribute for SkyDiffuseMap (Lars Mohrmann)
- [#2104] Implement sparse summed fit statistics in Cython (Axel Donath)

.. _gammapy_0p11_release:

0.11 (Mar 29, 2019)
-------------------

Summary
+++++++

- Released Mar 29, 2019
- 11 contributors
- 2 months of work
- 65 pull requests (not all listed below)

**What's new?**

Gammapy v0.11 implements a large part of the new joint-likelihood fitting
framework proposed in `PIG 8 - datasets`_ . This includes the introduction of the
``FluxPointsDataset``, ``MapDataset`` and ``Datasets`` classes, which now represent
the main interface to the ``Fit`` class and fitting backends in Gammapy. As a
first use-case of the new dataset classes we added a tutorial demonstrating a
joint-likelihood fit of a CTA 1DC Galactic center observations. We also
considerably improved the performance of the 3D likelihood evaluation by
evaluating the source model components on smaller cutouts of the map.
We also added a tutorial demonstrating the use of the ``MapDataset`` class for
MCMC sampling and show how to interface Gammapy to the widely used emcee package.
Gammapy v0.11 also includes a new pulsar analysis tutorial. It demonstrates
how to compute phase curves and phase resolved sky maps with Gammapy.
To better support classical analysis methods in our main API we implemented
a ``MapMakerRing`` class, that provides ring and adaptive ring background
estimation for map and image estimation.

Gammapy v0.11 improves the support for the scipy and sherpa fitting backends. It
now implements full support of parameter freezing and parameter limits for both
backends. We also added a ``reoptimize`` option to the ``Fit.likelihood_profile``
method to compute likelihood profiles with reoptimizing remaining free parameters.

For Gammapy v0.11 we added a ``SkyEllipse`` model to support fitting of elongated
sources and changed the parametrization of the ``SkyGaussian`` to integrate correctly
on the sphere. The spatial model classes now feature simple support for coordinate
frames, such that the position of the source can be defined and fitted independently
of the coordinate system of the data. Gammapy v0.11 now supports the evaluation
non-radially symmetric 3D background models and defining multiple background models
for a single ``MapDataset``.

Gammapy v0.11 drops support for Python 2.7, only Python 3.5 or newer is supported (see `PIG 3`_).
If you have any questions or need help to install Python 3, or to update your
scripts and notebooks to work in Python 3, please contact us any time on the
Gammapy mailing list or Slack. We apologise for the disruption and are happy to
help with this transition. Note that Gammapy v0.10 will remain available and is
Python 2 compatible forever, so sticking with that version might be an option
in some cases. pip and conda should handle this correctly, i.e. automatically
pick the last compatible version (Gammapy v0.10) on Python 2, or if you try
to force installation of a later version by explicitly giving a version number,
emit an error and exit without installing or updating.

For Gammapy v0.11 we removed the unmaintained ``gammapy.datasets`` sub-module.
Please use the ``gammapy download`` command to download datasets instead and
the ``$GAMMAPY_DATA`` environment variable to access the data directly from
your local gammapy-datasets folder.

**Contributors:**

In alphabetical order by first name:

- Atreyee Sinha
- Axel Donath
- Brigitta Sipocz
- Christoph Deil
- Fabio Acero
- hugovk
- Jason Watson (new)
- José Enrique Ruiz
- Lars Mohrmann
- Luca Giunti (new)
- Régis Terrier

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.11 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?q=is%3Apr+milestone%3A0.11+is%3Aclosed>`__.

- [#2098] Remove gammapy.datasets submodule (Axel Donath)
- [#2097] Clean up tutorial notebooks (Christoph Deil)
- [#2093] Clean up PSF3D / TablePSF interpolation unit handling (Axel Donath)
- [#2085] Improve EDispMap and PSFMap stacking (Régis Terrier)
- [#2077] Add MCMC tutorial using emcee (Fabio Acero)
- [#2076] Clean up maps/wcs.py (Axel Donath)
- [#2071] Implement MapDataset npred evaluation using cutouts (Axel Donath)
- [#2069] Improve support for scipy fitting backend (Axel Donath)
- [#2066] Add SkyModel.position and frame attribute (Axel Donath)
- [#2065] Add evaluation radius to SkyEllipse model (Luca Giunti)
- [#2064] Add simulate_dataset() convenience function (Fabio Acero)
- [#2054] Add likelihood profile reoptimize option (Axel Donath)
- [#2051] Add WcsGeom.cutout() method (Léa Jouvin)
- [#2050] Add notebook for 3D joint analysis (Léa Jouvin)
- [#2049] Add EventList.select_map_mask() method (Régis Terrier)
- [#2046] Add SkyEllipse model (Luca Giunti)
- [#2039] Simplify and move energy threshold computation (Axel Donath)
- [#2038] Add tutorial for pulsar analysis (Marion Spir-Jacob)
- [#2037] Add parameter freezing for sherpa backend (Axel Donath)
- [#2035] Fix symmetry issue in solid angle calculation for WcsGeom (Jason Watson)
- [#2034] Change SkyGaussian to spherical representation (Luca Giunti)
- [#2033] Add evaluation of asymmetric background models (Jason Watson)
- [#2031] Add EDispMap class (Régis Terrier)
- [#2030] Add Datasets class (Axel Donath)
- [#2028] Add hess notebook to gammapy download list (José Enrique Ruiz)
- [#2026] Refactor MapFit into MapDataset (Atreyee Sinha)
- [#2023] Add FluxPointsDataset class (Axel Donath)
- [#2022] Refactor TablePSF class (Axel Donath)
- [#2019] Simplify PSF stacking and containment radius computation (Axel Donath)
- [#2017] Updating astropy_helpers to 3.1 (Brigitta Sipocz)
- [#2016] Drop support for Python 2 (hugovk)
- [#2012] Drop Python 2 support (Christoph Deil)
- [#2009] Improve field-of-view coordinate transformations (Lars Mohrmann)

.. _gammapy_0p10_release:

0.10 (Jan 28, 2019)
-------------------

Summary
+++++++

- Released Jan 28, 2019
- 7 contributors
- 2 months of work
- 30 pull requests (not all listed below)

**What's new?**

Gammapy v0.10 is a small release. An option to have a background model with
parameters such as normalization and spectral tilt was added. The curated
example datasets were improved, the ``gammapy download`` script and access of
example data from the tutorials via the ``GAMMAPY_DATA`` environment variable
were improved. A notebook ``image_analysis`` showing how to use Gammapy to make
and model 2D images for a given given energy band, as a special case of the
existing 3D map-based analysis was added.

A lot of the work recently went into planning the work ahead for 2019. See the
`Gammapy 1.0 roadmap`_ and the `PIG 7 - models`_ as well as `PIG 8 - datasets`_
and get in touch if you want to contribute. We plan to ship a first version of
the new datasets API in Gammapy v0.11 in March 2019.

Gammapy v0.10 is the last Gammapy release that supports Python 2 (see `PIG 3`_).
If you have any questions or need help to install Python 3, or to update your
scripts and notebooks to work in Python 3, please contact us any time on the
Gammapy mailing list or Slack. We apologise for the disruption and are happy to
help with this transition.

pyyaml is now a core dependency of Gammapy, i.e. will always be automatically
installed as a dependency. Instructions for installing Gammapy on Windows, and
continuous testing on Windows were improved.

.. _PIG 7 - models: https://github.com/gammapy/gammapy/pull/1971
.. _PIG 8 - datasets: https://github.com/gammapy/gammapy/pull/1986

**Contributors:**

- Atreyee Sinha
- Axel Donath
- Christoph Deil
- David Fidalgo
- José Enrique Ruiz
- Lars Mohrmann
- Régis Terrier

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.10 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?q=is%3Apr+milestone%3A0.10+is%3Aclosed>`__.

- [#2001] Use GAMMAPY_DATA everywhere / remove GAMMAPY_EXTRA (José Enrique Ruiz)
- [#2000] Fix cta_simulation notebook, use CTA prod 3 IRFs (Régis Terrier)
- [#1998] Fix SensitivityEstimator after IRF API change (Régis Terrier)
- [#1995] Add pyyaml as core dependency (Christoph Deil)
- [#1994] Unify Fermi-LAT datasets used in Gammapy (Axel Donath)
- [#1991] Improve SourceCatalogObjectHGPS spatial model (Axel Donath)
- [#1990] Add background model for map fit (Atreyee Sinha)
- [#1989] Add tutorial notebook for 2D image analysis (Atreyee Sinha)
- [#1988] Improve gammapy download (José Enrique Ruiz)
- [#1979] Improve output units of spectral models (Axel Donath)
- [#1975] Improve EnergyDependentTablePSF evaluate methods (Axel Donath)
- [#1969] Improve ObservationStats (Lars Mohrmann)
- [#1966] Add ObservationFilter select methods (David Fidalgo)
- [#1962] Change data access in notebooks to GAMMAPY_DATA (José Enrique Ruiz)
- [#1951] Add keepdim option for maps (Atreyee Sinha)

.. _gammapy_0p9_release:

0.9 (Nov 29, 2018)
------------------

Summary
+++++++

- Released Nov 29, 2018
- 9 contributors (3 new)
- 2 months of work
- 88 pull requests (not all listed below)

**What's new?**

Gammapy v0.9 comes just two months after v0.8. This is following the `Gammapy
1.0 roadmap`_, Gammapy will from now on have bi-monthly releases, as we work
towards the Gammapy 1.0 release in fall 2019.

Gammapy v0.9 contains many fixes, and a few new features. Big new features
like observation event and time filters, background model classes, as well as
support for fitting joint datasets will come in spring 2019.

The ``FluxPointEstimator`` has been rewritten, and the option to compute
spectral likelihood profiles has been added. The background and diffuse model
interpolation in energy has been improved to be more accurate. The
``gammapy.utils.fitting`` backend is under heavy development, most of the
functionality of MINUIT (covariance, confidence intervals, profiles, contours)
can now be obtained from any ``Fit`` class (spectral or map analysis). Maps now
support arithmetic operators, so that you can e.g. write ``residual = counts -
model`` if ``counts`` and ``model`` are maps containing observed and model
counts.

Gammapy v0.9 now requires Astropy 2.0 or later, and Scipy was changed from
status of optional to required dependency, since currently it is required for
most analysis tasks (e.g. using interpolation when evaluating instrument
responses). Please also note that we have a `plan to drop Python 2.7 support`_
in Gammapy v0.11 in March 2019. If you have any questions or concerns about
moving your scripts and notebooks to Python 3, or need Python 2 support with
later Gammapy releases in 2019, please let us know!

.. _Gammapy 1.0 roadmap: https://github.com/gammapy/gammapy/pull/1841
.. _plan to drop Python 2.7 support: https://github.com/gammapy/gammapy/pull/1278

**Contributors:**

- Atreyee Sinha
- Axel Donath
- Brigitta Sipocz
- Christoph Deil
- Daniel Morcuende (new)
- David Fidalgo
- Ignacio Minaya (new)
- José Enrique Ruiz
- José Luis Contreras (new)
- Régis Terrier

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.9 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?q=is%3Apr+milestone%3A0.9+is%3Aclosed>`__.

- [#1949] Add fit minos_contour method (Christoph Deil)
- [#1937] No copy of input and result model in fit (Christoph Deil)
- [#1934] Improve FluxPointEstimator test and docs (Axel Donath)
- [#1933] Add likelihood profiles to FluxPointEstimator (Axel Donath)
- [#1930] Add sections in documentation navigation bar (José Enrique Ruiz)
- [#1929] Rewrite FluxPointEstimator (Axel Donath)
- [#1927] Improve Fit class, add confidence method (Christoph Deil)
- [#1926] Fix MapAxis interpolation FITS serialisation (Atreyee Sinha)
- [#1922] Add Fit.covar method (Christoph Deil)
- [#1921] Use and improve ScaledRegularGridInterpolator (Axel Donath)
- [#1919] Add Scipy as core dependency (Axel Donath)
- [#1918] Add parameters correlation matrix property (Christoph Deil)
- [#1912] Add ObservationFilter class (David Fidalgo)
- [#1909] Clean up irf/io.py and add load_cta_irf function (Régis Terrier)
- [#1908] Take observation time from GTI table (David Fidalgo)
- [#1904] Fix parameter limit handling in fitting (Christoph Deil)
- [#1903] Improve flux points class (Axel Donath)
- [#1898] Review and unify quantity handling (Axel Donath)
- [#1895] Rename obs_list to observations (David Fidalgo)
- [#1894] Improve Background3D energy axis integration (Axel Donath)
- [#1893] Add MapGeom equality operator (Régis Terrier)
- [#1891] Add arithmetic operators for maps (Régis Terrier)
- [#1890] Change map quantity to view instead of copy (Régis Terrier)
- [#1888] Change ObservationList class to Observations (David Fidalgo)
- [#1884] Improve analysis3d tutorial notebook (Ignacio Minaya)
- [#1883] Fix fit parameter bug for very large numbers (Christoph Deil)
- [#1871] Fix TableModel and ConstantModel output dimension (Régis Terrier)
- [#1862] Move make_psf, make_mean_psf and make_mean_edisp (David Fidalgo)
- [#1861] Change from live to on time in background computation (Christoph Deil)
- [#1859] Fix in MapFit energy dispersion apply (Régis Terrier)
- [#1857] Modify image_fitting_with_sherpa to use DC1 runs (Atreyee Sinha)
- [#1855] Add ScaledRegularGridInterpolator (Axel Donath)
- [#1854] Add FluxPointProfiles class (Christoph Deil)
- [#1846] Allow different true and reco energy in map analysis (Atreyee Sinha)
- [#1845] Improve first steps with Gammapy tutorial (Daniel Morcuende)
- [#1837] Add method to compute energy-weighted 2D PSF kernel (Atreyee Sinha)
- [#1836] Fix gammapy download for Python 2 (José Enrique Ruiz)
- [#1807] Change map smooth widths to match Astropy (Atreyee Sinha)
- [#1849] Improve gammapy.stats documentation page (José Luis Contreras)
- [#1766] Add gammapy jupyter CLI for developers (José Enrique Ruiz)
- [#1763] Improve gammapy download (José Enrique Ruiz)
- [#1710] Clean up TableModel implementation (Axel Donath)
- [#1419] PIG 4 - Setup for tutorial notebooks and data (José Enrique Ruiz and Christoph Deil)

.. _gammapy_0p8_release:

0.8 (Sep 23, 2018)
------------------

Summary
+++++++

- Released Sep 23, 2018
- 24 contributors (6 new)
- 7 months of work
- 314 pull requests (not all listed below)

**What's new?**

Gammapy v0.8 features major updates to maps and modeling, as well as
installation and how to get started with tutorial notebooks. It also contains
many smaller additions, as well as many fixes and improvements.

The new ``gammapy.maps`` is now used for all map-based analysis (2D images and
3D cubes with an energy axis). The old SkyImage and SkyCube classes have been
removed. All code and documentation has been updated to use ``gammapy.maps``. To
learn about the new maps classes, see the ``intro_maps`` tutorial at
:ref:`tutorials` and the :ref:`gammapy.maps <maps>` documentation page.

The new ``gammapy.utils.fitting`` contains a simple modeling and fitting
framework, that allows the use of ``iminuit`` and ``sherpa`` optimisers as
"backends" for any fit in Gammapy. The classes in ``gammapy.spectrum.models`` (1D
spectrum models) are updated, and ``gammapy.image.models`` (2D spatial models) and
``gammapy.cube.models`` (3D cube models) was added. The ``SpectrumFit`` class was
updated and a ``MapFit`` to fit models to maps was added. This part of Gammapy
remains work in progress, some changes and major improvements are planned for
the coming months.

With Gammapy v0.8, we introduce the ``gammapy download`` command to download
tutorial notebooks and example datasets. A step by step guide is here:
:ref:`getting-started`. Previously tutorial notebooks were maintained in a
separate ``gammapy-extra`` repository, which was inconvenient for users to clone
and use, and more importantly wasn't version-coupled with the Gammapy code
repository, causing major issues in this phase where Gammapy is still under
heavy development.

The recommended way to install Gammapy (described at :ref:`getting-started`) is
now to use conda and to create an environment with dependencies pinned to fixed
versions to get a consistent and reproducible environment. E.g. the Gammapy v0.8
environment uses Python 3.6, Numpy 1.15 and Astropy 3.0. As before, Gammapy is
compatible with a wide range of versions of Numpy and Astropy from the past
years and many installation options are available for Gammapy (e.g. pip or
Macports) in addition to conda. But we wanted to offer this new "stable
recommended environment" option for Gammapy as a default.

The new ``analysis_3d`` notebook shows how to run a 3D analysis for IACT data
using the ``MapMaker`` and ``MapFit`` classes. The ``simulate_3d`` shows how to
simulate and fit a source using CTA instrument response functions. The
simulation is done on a binned 3D cube, not via unbinned event sampling. The
``fermi_lat`` tutorial shows how to analyse high-energy Fermi-LAT data with
events, exposure and PSF pre-computed using the Fermi science tools. The
``hess`` and ``light_curve`` tutorial show how to analyse data from the recent
first H.E.S.S. test data release. You can find these tutorials and more at
:ref:`tutorials`.

Another addition in Gammapy v0.8 is :ref:`gammapy.astro.darkmatter
<astro-darkmatter>`, which contains spatial and spectral models commonly used in
dark matter searches using gamma-ray data.

The number of optional dependencies used in Gammapy has been reduced. Sherpa is
now an optional fitting backend, modeling is built-in in Gammapy. The following
packages are no longer used in Gammapy: scikit-image, photutils, pandas, aplpy.
The code quality and test coverage in Gammapy has been improved a lot.

This release also contains a large number of small improvements and bug fixes to
the existing code, listed below in the changelog.

We are continuing to develop Gammapy at high speed, significant improvements on
maps and modeling, but also on the data and IRF classes are planned for the
coming months and the v0.9 release in fall 2019. We apologise if you are already
using Gammapy for science studies and papers and have to update your scripts and
notebooks to work with the new Gammapy version. If possible, stick with a given
stable version of Gammapy. If you update to a newer version, let us know if you
have any issues or questions. We're happy to help!

Gammapy v0.8 works on Linux, MacOS and Windows, with Python 3.5, 3.6 as well as
legacy Python 2.7.

**Contributors:**

- Andrew Chen (new)
- Atreyee Sinha
- Axel Donath
- Brigitta Sipocz
- Bruno Khelifi
- Christoph Deil
- Cosimo Nigro
- David Fidalgo (new)
- Fabio Acero
- Gabriel Emery (new)
- Hubert Siejkowski (new)
- Jean-Philippe Lenain
- Johannes King
- José Enrique Ruiz
- Kai Brügge
- Lars Mohrmann
- Laura Vega Garcia (new)
- Léa Jouvin
- Marion Spir-Jacob (new)
- Matthew Wood
- Matthias Wegen
- Oscar Blanch
- Régis Terrier
- Roberta Zanin

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.8 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=%E2%9C%93&q=is%3Apr+milestone%3A0.8+is%3Amerged+>`__.

- [#1822] Use GAMMAPY_DATA in Gammapy codebase (José Enrique Ruiz)
- [#1821] Improve analysis 3D tutorial (Axel Donath)
- [#1818] Add HESS and background modeling tutorial (Christoph Deil)
- [#1812] Add Fit likelihood profile method (Axel Donath)
- [#1808] Rewrite getting started, improve tutorials and install pages (Christoph Deil)
- [#1800] Add ObservationTableChecker and improve EVENTS checker (Christoph Deil)
- [#1799] Fix EnergyDispersion write and to_sherpa (Régis Terrier)
- [#1791] Move tutorial notebooks to the Gammapy repository (José Enrique Ruiz)
- [#1785] Unify API of Gammapy Fit classes (Axel Donath)
- [#1764] Format all code in Gammapy black (Christoph Deil)
- [#1761] Add black notebooks functionality (José Enrique Ruiz)
- [#1760] Add conda env file for release v0.8 (José Enrique Ruiz)
- [#1759] Add find_peaks for images (Christoph Deil)
- [#1755] Change map FITS unit header key to standard "BUNIT" (Christoph Deil)
- [#1751] Improve EventList and data checkers (Christoph Deil)
- [#1750] Remove EventListDataset class (Christoph Deil)
- [#1748] Add DataStoreChecker and ObservationChecker (Christoph Deil)
- [#1746] Unify and fix testing of plot methods (Axel Donath)
- [#1731] Fix and unify Map.iter_by_image (Axel Donath)
- [#1711] Clean up map reprojection code (Axel Donath)
- [#1702] Add mask filter option to MapFit (Axel Donath)
- [#1697] Improve convolution code and tests (Axel Donath)
- [#1696] Add parameter auto scale (Johannes Kind and Christoph Deil)
- [#1695] Add WcsNDMap convolve method (Axel Donath)
- [#1685] Add quantity support to map coordinates (Axel Donath)
- [#1681] Add make_images method in MapMaker (Axel Donath)
- [#1675] Add gammapy.stats.excess_matching_significance (Christoph Deil)
- [#1660] Fix spectrum energy grouping, use nearest neighbor method (Johannes King)
- [#1658] Bundle skimage block_reduce in gammapy.extern (Christoph Deil)
- [#1634] Add SkyDiffuseCube model for 3D maps (Roberta Zanin and Christoph Deil)
- [#1630] Add new observation container class (David Fidalgo)
- [#1616] Improve reflected background region finder (Régis Terrier)
- [#1606] Change FluxPointFitter to use minuit (Axel Donath)
- [#1605] Remove old sherpa backend from SpectrumFit (Johannes King)
- [#1594] Remove SkyImage and SkyCube (Christoph Deil)
- [#1582] Migrate ring background to use gammapy.maps (Régis Terrier)
- [#1576] Migrate detect.cwt to use gammapy.maps (Hubert Siejkowski)
- [#1573] Migrate image measure and profile to use gammapy.maps (Axel Donath)
- [#1568] Remove IACT and Fermi-LAT basic image estimators (Christoph Deil)
- [#1564] Migrate gammapy.detect to use gammapy.maps (Axel Donath)
- [#1562] Add MapMaker run method (Atreyee Sinha)
- [#1558] Integrate background spectrum in MapMaker (Léa Jouvin)
- [#1556] Sync sky model parameters with components (Christoph Deil)
- [#1554] Introduce map copy method (Axel Donath)
- [#1543] Add plot_interactive method for 3D maps (Fabio Acero)
- [#1527] Migrate ASmooth to use gammapy.maps (Christoph Deil)
- [#1517] Remove cta_utils and CTASpectrumObservation (Christoph Deil)
- [#1515] Remove old background model code (Christoph Deil)
- [#1505] Remove old Sherpa 3D map analysis code (Christoph Deil)
- [#1495] Change MapMaker to allow partially contained observations (Atreyee Sinha)
- [#1492] Add robust periodogram to gammapy.time (Matthias Wegen)
- [#1489] Add + operator for SkyModel (Johannes King)
- [#1476] Add evaluate method Background3D IRF (Léa Jouvin)
- [#1475] Add field-of-view coordinate transformations (Lars Mohrmann)
- [#1474] Add more models to the xml model registry (Fabio Acero)
- [#1470] Add background to map model evaluator (Atreyee Sinha)
- [#1456] Add light curve upper limits (Bruno Khelifi)
- [#1447] Add a PSFKernel to perform PSF convolution on Maps (Régis Terrier)
- [#1446] Add WCS map cutout method (Atreyee Sinha)
- [#1444] Add map smooth method (Atreyee Sinha)
- [#1443] Add slice_by_idx methods to gammapy.maps (Axel Donath)
- [#1435] Add __repr__ methods to Maps and related classes (Axel Donath)
- [#1433] Fix map write for custom axis name (Christoph Deil)
- [#1432] Add PSFMap class (Régis Terrier)
- [#1426] Add background estimation for phase-resolved spectra (Marion Spir-Jacob)
- [#1421] Add map region mask (Régis Terrier)
- [#1412] Change to default overwrite=False in gammapy.maps (Christoph Deil)
- [#1408] Fix 1D spectrum joint fit (Johannes King)
- [#1406] Add adaptive lightcurve time binning method (Gabriel Emery)
- [#1401] Remove old spatial models and CatalogImageEstimator (Christoph Deil)
- [#1397] Add XML SkyModel serialization (Johannes King)
- [#1395] Change Map.get_coord to return a MapCoord object (Régis Terrier)
- [#1387] Update catalog to new model classes (Christoph Deil)
- [#1381] Add 3D fit example using gammapy.maps (Johannes King)
- [#1386] Improve spatial models and add diffuse models (Johannes King)
- [#1378] Change 3D model evaluation from SkyCube to Map (Christoph Deil)
- [#1377] Add more SkySpatialModel subclasses (Johannes King)
- [#1376] Add new SpatialModel base class (Johannes King)
- [#1374] Add units to gammapy.maps (Régis Terrier)
- [#1373] Improve 3D analysis code using gammapy.maps (Christoph Deil)
- [#1372] Add 3D analysis functions using gammapy.maps (Régis Terrier)
- [#1369] Add gammapy download command (José Enrique Ruiz)
- [#1367] Add first draft of LightCurve model class (Christoph Deil)
- [#1362] Fix map sum_over_axes (Christoph Deil)
- [#1360] Sphinx RTD responsive theme for documentation (José Enrique Ruiz)
- [#1357] Add map geom pixel solid angle computation (Régis Terrier)
- [#1354] Apply FOV mask to all maps in ring background estimator (Lars Mohrmann)
- [#1347] Fix bug in LightCurveEstimator (Lars Mohrmann)
- [#1346] Fix bug in map .fits.gz write (change map data transpose) (Christoph Deil)
- [#1345] Improve docs for SpectrumFit (Johannes King)
- [#1343] Apply containment correction in true energy (Johannes King)
- [#1341] Remove u.ct from gammapy.spectrum (Johannes King)
- [#1339] Add create fixed time interval method for light curves (Gabriel Emery)
- [#1337] Enable rate models in SpectrumSimulation (Johannes King)
- [#1334] Fix AREASCAL read for PHA count spectrum (Régis Terrier)
- [#1331] Fix background image estimate (Régis Terrier)
- [#1317] Add function to compute counts maps (Régis Terrier)
- [#1231] Improve HESS HGPS catalog source class (Christoph Deil)

.. _gammapy_0p7_release:

0.7 (Feb 28, 2018)
------------------

Summary
+++++++

- Released Feb 28, 2018
- 25 contributors (16 new)
- 10 months of work
- 178 pull requests (not all listed below)

**What's new?**

Installation:

- Gammapy 0.7 supports legacy Python 2.7, as well as Python 3.5 and 3.6.
  If you are still using Python 2.7 with Gammapy, please update to Python 3. Let
  us know if you need any help with the update, or are blocked from updating for
  some reason, by filling out the 1-minute `Gammapy installation questionnaire`_
  form. This will help us make a plan how to finish the Python 2 -> 3 transition
  and to set a timeline (`PIG 3`_).
- The Gammapy conda packages are now distributed via the ``conda-forge`` channel,
  i.e. to install or update Gammapy use the command ``conda install gammapy -c
  conda-forge``. Most other packages have also moved to ``conda-forge`` in the
  past years, the previously used ``astropy`` and ``openastronomy`` channels are
  no longer needed.
- We now have a conda ``environment.yml`` file that contains all packages used
  in the tutorials. See instructions here: :ref:`tutorials`.

Documentation:

- We have created a separate project webpage at https://gammapy.org .
  The https://docs.gammapy.org page is not just for the Gammapy documentation.
- A lot of new tutorials were added in the form of Jupyter notebooks. To make the content of the
  notebooks easier to navigate and search, a rendered static version of the notebooks was integrated
  in the Sphinx-based documentation (the one you are looking at) at :ref:`tutorials`.
- Most of the Gammapy tutorials can be executed directly in the browser via the https://mybinder.org/
  service. There is a "launch in binder" link at the top of each tutorial in the docs,
  see e.g. here: `CTA data analysis with Gammapy <notebooks/cta_data_analysis.html>`__
- A page was created to collect the information for CTA members how to get started with Gammapy
  and with contact / support channels: https://gammapy.org/cta.html

Gammapy Python package:

- This release contains many bug fixes and improvements to the existing code,
  ranging from IRF interpolation to spectrum and lightcurve computation. Most of
  the improvements (see the list of pull requests below) were driven by user
  reports and feedback from CTA, HESS, MAGIC and Fermi-LAT analysis. Please
  update to the new version and keep filing bug reports and feature requests!
- A new sub-package `gammapy.maps` was added that features WCS and HEALPix based maps,
  arbitrary extra axes in addition to the two spatial dimensions (e.g. energy,
  time or event type). Support for multi-resolution and sparse maps is work in
  progress. These new maps classes were implemented based on the experience
  gained from the existing ``SkyImage`` and ``SkyCube`` classes as well as the
  Fermi science tools, Fermipy and pointlike. Work on new analysis code based on
  ``gammapy.maps`` within Gammapy is starting now (see `PIG 2`_). Users are
  encouraged to start using ``gammapy.maps`` in their scripts. The plan is to
  keep the existing ``SkyImage`` and ``SkyCube`` and image / cube analysis code
  that we have now mostly unchanged (only apply bugfixes), and to remove them at
  some future date after the transition to the use of ``gammapy.maps`` within
  Gammapy (including all tests and documentation and tutorials) is complete and
  users had some time to update their code. If you have any questions or need
  help with ``gammapy.maps`` or find an issue or missing feature, let us know!

Command line interface:

- The Gammapy command-line interface was changed to use a single command
  ``gammapy`` multiple sub-commands (like ``gammapy info`` or ``gammapy image
  bin``). Discussions on developing the high-level interface for Gammapy (e.g.
  as a set of command line tools, or a config file driven analysis) are starting
  now.

Organisation:

- A webpage at https://gammapy.org/ was set up, separate from the Gammapy
  documentation page https://docs.gammapy.org/ .
- The Gammapy project and team organisation was set up with clear roles and
  responsibilities, in a way to help the Gammapy project grow, and to support
  astronomers and projects like CTA using Gammapy better. This is described at
  https://gammapy.org/team.html .
- To improve the quality of Gammapy, we have set up a proposal-driven process
  for major improvements for Gammapy, described in :ref:`pig-001`. We are now
  starting to use this to design a better low-level analysis code (`PIG 2`_) and
  to define a plan to finish the Python 2-> 3 transition (`PIG 3`_).

.. _PIG 2: https://github.com/gammapy/gammapy/pull/1277
.. _PIG 3: https://github.com/gammapy/gammapy/pull/1278
.. _Gammapy installation questionnaire: https://goo.gl/forms/0QuYYyyPCbKnFJJI3

**Contributors:**

- Anne Lemière (new)
- Arjun Voruganti
- Atreyee Sinha (new)
- Axel Donath
- Brigitta Sipocz
- Bruno Khelifi (new)
- Christoph Deil
- Cosimo Nigro (new)
- Jean-Philippe Lenain (new)
- Johannes King
- José Enrique Ruiz (new)
- Julien Lefaucheur
- Kai Brügge (new)
- Lab Saha (new)
- Lars Mohrmann
- Léa Jouvin
- Matthew Wood
- Matthias Wegen (new)
- Oscar Blanch (new)
- Peter Deiml (new)
- Régis Terrier
- Roberta Zanin (new)
- Rubén López-Coto (new)
- Thomas Armstrong (new)
- Thomas Vuillaume (new)
- Yves Gallant (new)

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.7 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=%E2%9C%93&q=is%3Apr+milestone%3A0.7+is%3Amerged+>`__.

- [#1319] Fix a bug in SpectrumStacker (Anne Lemière)
- [#1318] Improve MapCoord interface (Matthew Wood)
- [#1316] Add flux point estimation for multiple observations (Lars Mohrmann)
- [#1312] Add Background 2D class (Léa Jouvin)
- [#1305] Fix exposure and flux units in IACTBasicImageEstimator (Yves Gallant)
- [#1300] Add PhaseCurve class for periodic systems (Lab Saha)
- [#1294] Fix IACTBasicImageEstimator psf method (Yves Gallant)
- [#1291] Add meta attribute to maps (Léa Jouvin)
- [#1290] Change image_pipe and fov to include a minimum offset cut (Atreyee Sinha)
- [#1289] Fix excess for given significance computation (Oscar Blanch)
- [#1287] Fix time in LightCurveEstimator result table (Jean-Philippe Lenain)
- [#1281] Add methods for WCS maps (Matthew Wood)
- [#1266] No pytest import from non-test code (Christoph Deil)
- [#1268] Fix PSF3D.to_energy_dependent_table_psf (Christoph Deil)
- [#1246] Improve map read method (Matthew Wood)
- [#1240] Finish change to Click in gammapy.scripts (Christoph Deil)
- [#1238] Clean up catalog image code (Axel Donath)
- [#1235] Introduce main ``gammapy`` command line tool (Axel Donath and Christoph Deil)
- [#1227] Remove gammapy-data-show and gammapy-cube-bin (Christoph Deil)
- [#1226] Make DataStoreObservation properties less lazy (Christoph Deil)
- [#1220] Fix flux point computation for non-power-law models (Axel Donath)
- [#1215] Finish integration of Jupyter notebooks with Sphinx docs (Jose Enrique Ruiz)
- [#1211] Add IRF write methods (Thomas Armstrong)
- [#1210] Fix min energy handling in SpectrumEnergyGrouper (Julien Lefaucheur and Christoph Deil)
- [#1207] Add theta2 distribution plot to EventList class (Thomas Vuillaume)
- [#1204] Consistently use mode='constant' in convolutions of RingBackgroundEstimator (Lars Mohrmann)
- [#1195] Change IRF extrapolation behaviour (Christoph Deil)
- [#1190] Refactor gammapy.maps methods for calculating index and coordinate arrays (Matthew Wood)
- [#1183] Add function to compute background cube (Roberta Zanin and Christoph Deil)
- [#1179] Fix two bugs in LightCurveEstimator, and improve speed considerably (Lars Mohrmann)
- [#1176] Integrate tutorial notebooks in Sphinx documentation (Jose Enrique Ruiz)
- [#1170] Add sparse map prototype (Matthew Wood)
- [#1169] Remove old HEALPix image and cube classes (Christoph Deil)
- [#1166] Fix ring background estimation (Axel Donath)
- [#1162] Add ``gammapy.irf.Background3D`` (Roberta Zanin and Christoph Deil)
- [#1150] Fix PSF evaluate error at low energy and high offset (Bruno Khelifi)
- [#1134] Add MAGIC Crab reference spectrum (Cosimo Nigro)
- [#1133] Fix energy_resolution method in EnergyDispersion class (Lars Mohrmann)
- [#1127] Fix 3FHL spectral indexes for PowerLaw model (Julien Lefaucheur)
- [#1115] Fix energy bias computation (Cosimo Nigro)
- [#1110] Remove ATNF catalog class and Green catalog load function (Christoph Deil)
- [#1108] Add HAWC 2HWC catalog (Peter Deiml)
- [#1107] Rewrite GaussianBand2D model (Axel Donath)
- [#1105] Emit warning when HDU loading from index is ambiguous (Lars Mohrmann)
- [#1104] Change conda install instructions to conda-forge channel (Christoph Deil)
- [#1103] Remove catalog and data browser Flask web apps (Christoph Deil)
- [#1102] Add 3FGL spatial models (Axel Donath)
- [#1100] Add energy reference for exposure map (Léa Jouvin)
- [#1098] Improve flux point fitter (Axel Donath)
- [#1093] Implement I/O methods for ``gammapy.maps`` (Matthew Wood)
- [#1092] Add random seed argument for CTA simulations (Julien Lefaucheur)
- [#1090] Add default parameters for spectral models (Axel Donath)
- [#1089] Fix Fermi-LAT catalog flux points property (Axel Donath)
- [#1088] Update Gammapy to match Astropy region changes (Johannes King)
- [#1087] Add peak energy property to some spectral models (Axel Donath)
- [#1085] Update astropy-helpers to v2.0 (Brigitta Sipocz)
- [#1084] Add flux points upper limit estimation (Axel Donath)
- [#1083] Add JSON-serialisable source catalog object dict (Arjun Voruganti)
- [#1082] Add observation sanity check method to DataStore (Lars Mohrmann)
- [#1078] Add printout for 3FHL and gamma-cat sources (Arjun Voruganti)
- [#1076] Development in ``gammapy.maps`` (Matthew Wood)
- [#1073] Fix spectrum fit for case of no EDISP (Johannes King)
- [#1070] Add Lomb-Scargle detection function (Matthias Wegen)
- [#1069] Add easy access to parameter errors (Johannes King)
- [#1067] Add flux upper limit computation to TSImageEstimator (Axel Donath)
- [#1065] Add skip_missing option to ``DataStore.obs_list`` (Johannes King)
- [#1057] Use system pytest rather than astropy (Brigitta Sipocz)
- [#1054] Development in ``gammapy.maps`` (Matthew Wood)
- [#1053] Add sensitivity computation (Bruno Khelifi)
- [#1051] Improve 3D simulation / analysis example (Roberta Zanin)
- [#1045] Fix energy dispersion apply and to_sherpa (Johannes King)
- [#1043] Make ``gammapy.spectrum.powerlaw`` private (Christoph Deil)
- [#1040] Add combined 3D model and simple npred function (Christoph Deil)
- [#1038] Remove ``gammapy.utils.mpl_style`` (Christoph Deil)
- [#1136] Improve CTA sensitivity estimator (Axel Donath and Kai Brügge)
- [#1035] Some cleanup of FluxPoints code and tests (Christoph Deil)
- [#1032] Improve table unit standardisation and flux points (Christoph Deil)
- [#1031] Add HGPS catalog spatial models (Axel Donath)
- [#1029] Add 3D model simulation example (Roberta Zanin)
- [#1027] Add gamma-cat resource and resource index classes (Christoph Deil)
- [#1026] Fix Fermi catalog flux points upper limits (Axel Donath)
- [#1025] Remove spectrum butterfly class (Christoph Deil)
- [#1021] Fix spiralarm=False case in make_base_catalog_galactic (Ruben Lopez-Coto)
- [#1014] Introduce TSImageEstimator class (Axel Donath)
- [#1013] Add Fermi-LAT 3FHL spatial models (Axel Donath)
- [#845] Add background model component to SpectrumFit (Johannes King)
- [#111] Include module-level variables in API docs (Christoph Deil)

.. _gammapy_0p6_release:

0.6 (Apr 28, 2017)
------------------

Summary
+++++++

- Released Apr 28, 2017
- 14 contributors (5 new)
- 5 months of work
- 147 pull requests (not all listed below)

**What's new?**

- Release and installation
    - Until now, we had a roughly bi-yearly release cycle for Gammapy.
      Starting now, we will make stable releases more often, to ship features and fixes to Gammapy users more quickly.
    - Gammapy 0.6 requires Python 2.7 or 3.4+, Numpy 1.8+, Scipy 0.15+, Astropy 1.3+, Sherpa 4.9.0+ .
      Most things will still work with older Astropy and Sherpa, but we dropped testing
      for older versions from our continuous integration.
    - Gammapy is now available via Macports, a package manager for Mac OS (``port install py35-gammapy``)
- Documentation
    - Added many tutorials as Jupyter notebooks (linked to from the docs front-page)
    - Misc docs improvements and new getting started notebooks
- For CTA
    - Better support for CTA IRFs
    - A notebook showing how to analyse some simulated CTA data (preliminary files from first data challenge)
    - Better support and documentation for CTA will be the focus of the next release (0.7).
- For Fermi-LAT
    - Introduced a reference dataset: https://github.com/gammapy/gammapy-fermi-lat-data
    - Added convenience class to work with Fermi-LAT datasets
- gammapy.catalog
    - Add support for gamma-cat, an open data collection and source catalog for gamma-ray astronomy
      (https://github.com/gammapy/gamma-cat)
    - Access to more Fermi-LAT catalogs (1FHL, 2FHL, 3FHL)
- gammapy.spectrum
    - Better flux point class
    - Add flux point SED fitter
    - EBL-absorbed spectral models
    - Improved spectrum simulation class
- gammapy.image
    - Add image radial and box profiles
    - Add adaptive ring background estimation
    - Add adaptive image smooth algorithm
- gammapy.cube
    - Add prototype for 3D analysis of IACT data (work in progress)
- gammapy.time
    - Add prototype lightcurve estimator for IACT data (work in progress)
- gammapy.irf
    - Many IRF classes now rewritten to use the generic ``NDDataArray`` and axis classes
    - Better handling of energy dispersion
- gammapy.utils
    - Add gammapy.utils.modeling (work in progress)
    - Add gammapy.utils.sherpa (generic interface to sherpa for fitting, with models
      and likelihood function defined in Gammapy) (work in progress)
- Many small bugfixes and improvements throughout the codebase and documentation

**Contributors:**

- Arjun Voruganti (new)
- Arpit Gogia (new)
- Axel Donath
- Brigitta Sipocz
- Bruno Khelifi (new)
- Christoph Deil
- Dirk Lennarz
- Fabio Acero (new)
- Johannes King
- Julien Lefaucheur
- Lars Mohrmann (new)
- Léa Jouvin
- Nachiketa Chakraborty
- Régis Terrier
- Zé Vinícius (new)

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.6 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=%E2%9C%93&q=is%3Apr+milestone%3A0.6+is%3Amerged+>`__.

- [#1006] Add possibilty to skip runs based on alpha in SpectrumExtraction (Johannes King)
- [#1002] Containment correction in SpectrumObservation via AREASCAL (Johannes King)
- [#1001] Add SpectrumAnalysisIACT (Johannes King)
- [#997] Add compute_chisq method to lightcurve class (Nachiketa Chakraborty)
- [#994] Improve Gammapy installation docs (Christoph Deil)
- [#988] Add spectral model absorbed by EBL that can be fit (Julien Lefaucheur)
- [#985] Improve error methods on spectral models (Axel Donath)
- [#979] Add flux point fitter class (Axel Donath)
- [#976] Fixes to Galactic population simulation (Christoph Deil)
- [#975] Add PLSuperExpCutoff3FGL spectral model (Axel Donath)
- [#966] Remove SkyMask (merge with SkyImage) (Christoph Deil)
- [#950] Add light curve computation (Julien Lefaucheur)
- [#933] Change IRF plotting from imshow to pcolormesh (Axel Donath)
- [#932] Change NDDataArray default_interp_kwargs to extrapolate (Johannes King)
- [#919] Fix Double plot issue in notebooks and improve events.peek() (Fabio Acero)
- [#911] Improve EnergyDispersion2D get_response and tests (Régis Terrier)
- [#906] Fix catalog getitem to work with numpy int index (Zé Vinícius)
- [#898] Add printout for 3FGL catalog objects (Arjun Voruganti)
- [#893] Add Fermi-LAT 3FGL catalog object lightcurve property (Arpit Gogia)
- [#888] Improve CTA IRF and simulation classes (point-like analysis) (Julien Lefaucheur)
- [#885] Improve spectral model uncertainty handling (Axel Donath)
- [#884] Improve BinnedDataAxis handling of lo / hi binning (Johannes King)
- [#883] Improve spectrum docs page (Johannes King)
- [#881] Add support for observations with different energy binning in SpectrumFit (Lars Mohrmann)
- [#875] Add CTA spectrum simulation example (Julien Lefaucheur)
- [#872] Add SED type e2dnde to FluxPoints (Johannes King)
- [#871] Add Parameter class to SpectralModel (Johannes King)
- [#870] Clean up docstrings in background sub-package (Arpit Gogia)
- [#868] Add Fermi-LAT 3FHL catalogue (Julien Lefaucheur)
- [#865] Add Fermi basic image estimator (Axel Donath)
- [#864] Improve edisp.apply to support different true energy axes (Johannes King)
- [#859] Remove old image_profile function (Axel Donath)
- [#858] Fix Fermi catalog flux point upper limits (Axel Donath)
- [#855] Add Fermi-LAT 1FHL catalogue (Julien Lefaucheur)
- [#854] Add Fermi-LAT dataset class (Axel Donath)
- [#851] Write Macports install docs (Christoph Deil)
- [#847] Fix Sherpa spectrum OGIP file issue (Régis Terrier and Johannes King)
- [#842] Add AbsorbedSpectralModel and improve CTA IRF class (Julien Lefaucheur)
- [#840] Fix energy binning issue in cube pipe (Léa Jouvin)
- [#837] Fix containment fraction issue for table PSF (Léa Jouvin)
- [#836] Fix spectrum observation write issue (Léa Jouvin)
- [#835] Add image profile estimator class (Axel Donath)
- [#834] Bump to require Astropy v1.3 (Christoph Deil)
- [#833] Add image profile class (Axel Donath)
- [#832] Improve NDDataArray (use composition, not inheritance) (Johannes King)
- [#831] Add CTA Sensitivity class and plot improvements (Julien Lefaucheur)
- [#830] Add gammapy.utils.modeling and GammaCat to XML (Christoph Deil)
- [#827] Add energy dispersion for 3D spectral analysis (Léa Jouvin)
- [#826] Add sky cube computation for IACT data (Léa Jouvin)
- [#825] Update astropy-helpers to v1.3 (Brigitta Sipocz)
- [#824] Add XSPEC table absorption model to spectral table model (Julien Lefaucheur)
- [#820] Add morphology models for gamma-cat sources (Axel Donath)
- [#816] Add class to access CTA point-like responses (Julien Lefaucheur)
- [#814] Remove old flux point classes (Axel Donath)
- [#813] Improve Feldman Cousins code (Dirk Lennarz)
- [#812] Improve differential flux point computation code (Axel Donath)
- [#811] Adapt catalogs to new flux point class (Axel Donath)
- [#810] Add new flux point class (Axel Donath)
- [#798] Add Fvar variability measure for light curves (Nachiketa Chakraborty)
- [#796] Improve LogEnergyAxis object (Axel Donath)
- [#797] Improve WStat implementation (Johannes King)
- [#793] Add GammaCat source catalog (Axel Donath)
- [#791] Misc fixes to spectrum fitting code (Johannes King)
- [#784] Improve SkyCube exposure computation (Léa Jouvin)

.. _gammapy_0p5_release:

0.5 (Nov 22, 2016)
------------------

Summary
+++++++

- Released Nov 22, 2016
- 12 contributors (5 new)
- 7 months of work
- 184 pull requests (not all listed below)
- Requires Python 2.7 or 3.4+, Numpy 1.8+, Scipy 0.15+, Astropy 1.2+, Sherpa 4.8.2+

**What's new?**

- Tutorial-style getting started documentation as Jupyter notebooks
- Removed ``gammapy.regions`` and have switched to the move complete
  and powerful `regions <http://astropy-regions.readthedocs.io/>`__ package
  (planned to be added to the Astropy core within the next year).
- ``gammapy.spectrum`` - Many 1-dimensional spectrum analysis improvements (e.g. spectral point computation)
- ``gammapy.image`` - Many ``SkyImage`` improvements, adaptive ring background estimation, asmooth algorithm
- ``gammapy.detect`` - CWT and TS map improvements
- ``gammapy.time`` - A lightcurve class and variability test
- ``gammapy.irf`` - Many improvements to IRF classes, especially the PSF classes.
- Many improved tests and test coverage

**Contributors:**

- Axel Donath
- Brigitta Sipocz
- Christoph Deil
- Domenico Tiziani (new)
- Helen Poon (new)
- Johannes King
- Julien Lefaucheur (new)
- Léa Jouvin
- Matthew Wood (new)
- Nachiketa Chakraborty (new)
- Olga Vorokh
- Régis Terrier

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.5 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=%E2%9C%93&q=is%3Apr+milestone%3A0.5+is%3Amerged+>`__.

- [#790] Add powerlaw energy flux integral for ``gamma=2`` (Axel Donath)
- [#789] Fix Wstat (Johannes King)
- [#783] Add PHA type II file I/O to SpectrumObservationList (Johannes King)
- [#778] Fix Gauss PSF energy bin issue (Léa Jouvin)
- [#777] Rewrite crab spectrum as class (Axel Donath)
- [#774] Add skyimage smooth method (Axel Donath)
- [#772] Stack EDISP for a set of observations (Léa Jouvin)
- [#767] Improve PSF checker and add a test (Christoph Deil)
- [#766] Improve SkyCube convolution and npred computation (Axel Donath)
- [#763] Add TablePSFChecker (Domenico Tiziani)
- [#762] Add IRFStacker class (Léa Jouvin)
- [#759] Improve SkyCube energy axes (Axel Donath)
- [#754] Change EventList from Table subclass to attribute (Christoph Deil)
- [#753] Improve SkyCube class (Axel Donath)
- [#746] Add image asmooth algorithm (Axel Donath)
- [#740] Add SpectrumObservationStacker (Johannes King)
- [#739] Improve kernel background estimator (Axel Donath)
- [#738] Fix reflected region pixel origin issue (Léa Jouvin)
- [#733] Add spectral table model (Julien Lefaucheur)
- [#731] Add energy dispersion RMF integration (Léa Jouvin)
- [#719] Add adaptive ring background estimation (Axel Donath)
- [#713] Improve ring background estimation (Axel Donath)
- [#710] Misc image and cube cleanup (Christoph Deil)
- [#709] Spectrum energy grouping (Christoph Deil)
- [#679] Add flux point computation method (Johannes King)
- [#677] Fermi 3FGL and 2FHL spectrum plotting (Axel Donath)
- [#661] Improve continuous wavelet transform (Olga Vorokh)
- [#660] Add Fermipy sky image code to Gammapy (Matthew Wood)
- [#653] Add up- and downsampling to SkyImage (Axel Donath)
- [#649] Change to astropy regions package (Christoph Deil)
- [#648] Add class to load CTA IRFs (Julien Lefaucheur)
- [#647] Add SpectrumSimulation class (Johannes King)
- [#641] Add ECPL model, energy flux and integration methods (Axel Donath)
- [#640] Remove pyfact (Christoph Deil)
- [#635] Fix TS maps low stats handling (Axel Donath)
- [#631] Fix ExclusionMask.distance (Olga Vorokh)
- [#628] Add flux points computation methods (Johannes King)
- [#622] Make gammapy.time great again (Christoph Deil)
- [#599] Move powerlaw utility functions to separate namespace (Christoph Deil)
- [#594] Fix setup.py and docs/conf.py configparser import (Christoph Deil)
- [#593] Remove gammapy/hspec (Christoph Deil)
- [#591] Add spectrum energy flux computation (Axel Donath)
- [#582] Add SkyImageList (Olga Vorokh)
- [#558] Finish change to use gammapy.extern.regions (Johannes King and Christoph Deil)
- [#569] Add detection utilities à la BgStats (Julien Lefaucheur)
- [#565] Add exptest time variability test (Helen Poon)
- [#564] Add LightCurve class (Nachiketa Chakraborty)
- [#559] Add paste, cutout and look_up methods to SkyMap class (Axel Donath)
- [#557] Add spectrum point source containment correction option (Régis Terrier)
- [#556] Add offset-dependent table PSF class (Domenico Tiziani)
- [#549] Add mean PSF computation (Léa Jouvin)
- [#547] Add astropy.regions to gammapy.extern (Johannes King)
- [#546] Add Target class (Johannes King)
- [#545] Add PointingInfo class (Christoph Deil)
- [#544] Improve SkyMap.coordinates (Olga Vorokh)
- [#541] Refactor effective area IRFs to use NDDataArray (Johannes King)
- [#535] Add spectrum and flux points to HGPS catalog (Axel Donath)
- [#531] Add ObservationTableSummary class (Julien Lefaucheur)
- [#530] Update readthedocs links from .org to .io (Brigitta Sipocz)
- [#529] Add data_summary method to DataStore (Johannes King)
- [#527] Add n-dim data base class for gammapy.irf (Johannes King)
- [#526] Add King PSF evaluate and to_table_psf methods (Léa Jouvin)
- [#524] Improve image pipe class (Léa Jouvin)
- [#523] Add Gauss PSF to_table_psf method (Axel Donath)
- [#521] Fix image pipe class (Léa Jouvin)

.. _gammapy_0p4_release:

0.4 (Apr 20, 2016)
------------------

Summary
+++++++

- Released Apr 20, 2016
- 10 contributors (5 new)
- 8 months of work
- 108 pull requests (not all listed below)
- Requires Python 2.7 or 3.4+, Numpy 1.8+, Scipy 0.15+, Astropy 1.0+, Sherpa 4.8+

**What's new?**

- Women are hacking on Gammapy!
- IACT data access via DataStore and HDU index tables
- Radially-symmetric background modeling
- Improved 2-dim image analysis
- 1-dim spectral analysis
- Add sub-package ``gammapy.cube`` and start working on 3-dim cube analysis
- Continuous integration testing for Windows on Appveyor added
  (Windows support for Gammapy is preliminary and incomplete)

**Contributors:**

- Axel Donath
- Brigitta Sipocz (new)
- Christoph Deil
- Dirk Lennarz (new)
- Johannes King
- Jonathan Harris
- Léa Jouvin (new)
- Luigi Tibaldo (new)
- Manuel Paz Arribas
- Olga Vorokh (new)

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.4 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=%E2%9C%93&q=is%3Apr+milestone%3A0.4+is%3Amerged+>`__.

- [#518] Fixes and cleanup for SkyMap (Axel Donath)
- [#511] Add exposure image computation (Léa Jouvin)
- [#510] Add acceptance curve smoothing method (Léa Jouvin)
- [#507] Add Fermi catalog spectrum evaluation and plotting (Johannes King)
- [#506] Improve TS map computation performance (Axel Donath)
- [#503] Add FOV background image modeling (Léa Jouvin)
- [#502] Add DataStore subset method (Johannes King)
- [#487] Add SkyMap class (Axel Donath)
- [#485] Add OffDataBackgroundMaker (Léa Jouvin)
- [#484] Add Sherpa cube analysis prototype (Axel Donath)
- [#481] Add new gammapy.cube sub-package (Axel Donath)
- [#478] Add observation stacking method for spectra (Léa Jouvin and Johannes King)
- [#475] Add tests for TS map image computation (Olga Vorokh)
- [#474] Improve significance image analysis (Axel Donath)
- [#473] Improve tests for HESS data (Johannes King)
- [#462] Misc cleanup (Christoph Deil)
- [#461] Pacman (Léa Jouvin)
- [#459] Add radially symmetric FOV background model (Léa Jouvin)
- [#457] Improve data and observation handling (Christoph Deil)
- [#456] Fix and improvements to TS map tool (Olga Vorokh)
- [#455] Improve IRF interpolation and extrapolation (Christoph Deil)
- [#447] Add King profile PSF class (Christoph Deil)
- [#436] Restructure spectrum package and command line tool (Johannes King)
- [#435] Add info about Gammapy contact points and gammapy-extra (Christoph Deil)
- [#421] Add spectrum fit serialisation code (Johannes King)
- [#403] Improve spectrum analysis (Johannes King)
- [#415] Add EventList plots (Jonathan Harris)
- [#414] Add Windows tests on Appveyor (Christoph Deil)
- [#398] Add function to compute exposure cubes (Luigi Tibaldo)
- [#396] Rewrite spectrum analysis (Johannes King)
- [#395] Fix misc issues with IRF classes (Johannes King)
- [#394] Move some data specs to gamma-astro-data-formats (Christoph Deil)
- [#392] Use external ci-helpers (Brigitta Sipocz)
- [#387] Improve Gammapy catalog query and browser (Christoph Deil)
- [#383] Add EnergyOffsetArray (Léa Jouvin)
- [#379] Add gammapy.region and reflected region computation (Johannes King)
- [#375] Misc cleanup of scripts and docs (Christoph Deil)
- [#371] Improve catalog utils (Christoph Deil)
- [#369] Improve the data management toolbox (Christoph Deil)
- [#367] Add Feldman Cousins algorithm (Dirk Lennarz)
- [#364] Improve catalog classes and gammapy-extra data handling (Jonathan Harris, Christoph Deil)
- [#361] Add gammapy-spectrum-pipe (Johannes King)
- [#359] Add 1D spectrum analysis tool based on gammapy.hspec (Johannes King)
- [#353] Add some scripts and examples (Christoph Deil)
- [#352] Add data management tools (Christoph Deil)
- [#351] Rewrite EnergyDispersion class (Johannes King)
- [#348] Misc code cleanup (Christoph Deil)
- [#347] Add background cube model comparison plot script (Manuel Paz Arribas)
- [#342] Add gammapy-bin-image test (Christoph Deil)
- [#339] Remove PoissonLikelihoodFitter (Christoph Deil)
- [#338] Add example script for cube background models (Manuel Paz Arribas)
- [#337] Fix sherpa morphology fitting script (Axel Donath)
- [#335] Improve background model simulation (Manuel Paz Arribas)
- [#332] Fix TS map boundary handling (Axel Donath)
- [#330] Add EnergyDispersion and CountsSpectrum (Johannes King)
- [#319] Make background cube models (Manuel Paz Arribas)
- [#290] Improve energy handling (Johannes King)

.. _gammapy_0p3_release:

0.3 (Aug 13, 2015)
------------------

Summary
+++++++

- Released Aug 13, 2015
- 9 contributors (5 new)
- 4 months of work
- 24 pull requests
- Requires Astropy version 1.0 or later.
- On-off likelihood spectral analysis was added in gammapy.hspec,
  contributed by Régis Terrier and Ignasi Reichardt.
  It will be refactored and is thus not part of the public API.
- The Gammapy 0.3 release is the basis for an `ICRC 2015 poster contribution <https://indico.cern.ch/event/344485/session/142/contribution/695>`__

**Contributors:**

- Manuel Paz Arribas
- Christoph Deil
- Axel Donath
- Jonathan Harris (new)
- Johannes King (new)
- Stefan Klepser (new)
- Ignasi Reichardt (new)
- Régis Terrier
- Victor Zabalza (new)

Pull requests
+++++++++++++

- [#326] Fix Debian install instructions (Victor Zabalza)
- [#318] Set up and document logging for Gammapy (Christoph Deil)
- [#317] Using consistent plotting style in docs (Axel Donath)
- [#312] Add an "About Gammapy" page to the docs (Christoph Deil)
- [#306] Use assert_quantity_allclose from Astropy (Manuel Paz Arribas)
- [#301] Simplified attribute docstrings (Manuel Paz Arribas)
- [#299] Add cube background model class (Manuel Paz Arribas)
- [#296] Add interface to HESS FitSpectrum JSON output (Christoph Deil)
- [#295] Observation table subset selection (Manuel Paz Arribas)
- [#291] Remove gammapy.shower package (Christoph Deil)
- [#289] Add a simple Makefile for Gammapy. (Manuel Paz Arribas)
- [#286] Function to plot Fermi 3FGL light curves (Jonathan Harris)
- [#285] Add infos how to handle times in Gammapy (Christoph Deil)
- [#283] Consistent random number handling and improve sample_sphere (Manuel Paz Arribas)
- [#280] Add new subpackage: gammapy.time (Christoph Deil)
- [#279] Improve SNRcat dataset (Christoph Deil)
- [#278] Document observation tables and improve gammapy.obs (Manuel Paz Arribas)
- [#276] Add EffectiveAreaTable exporter to EffectiveAreaTable2D (Johannes King)
- [#273] Fix TS map header writing and temp file handling (Axel Donath)
- [#264] Add hspec - spectral analysis using Sherpa (Régis Terrier, Ignasi Reichardt, Christoph Deil)
- [#262] Add SNRCat dataset access function (Christoph Deil)
- [#261] Fix spiral arm model bar radius (Stefan Klepser)
- [#260] Add offset-dependent effective area IRF class (Johannes King)
- [#256] EventList class fixes and new features (Christoph Deil)

.. _gammapy_0p2_release:

0.2 (Apr 13, 2015)
------------------

Summary
+++++++

- Released Apr 13, 2015
- 4 contributors (1 new)
- 8 months of work
- 40 pull requests
- Requires Astropy version 1.0 or later.
- Gammapy now uses `Cython <http://cython.org/>`__,
  i.e. requires a C compiler for end-users and in addition Cython for developers.

**Contributors:**

- Manuel Paz Arribas (new)
- Christoph Deil
- Axel Donath
- Ellis Owen

Pull requests
+++++++++++++

- [#254] Add changelog for Gammapy (Christoph Deil)
- [#252] Implement TS map computation in Cython (Axel Donath)
- [#249] Add data store and observation table classes, improve event list classes (Christoph Deil)
- [#248] Add function to fill acceptance image from curve (Manuel Paz Arribas)
- [#247] Various fixes to image utils docstrings (Manuel Paz Arribas)
- [#246] Add catalog and plotting utils (Axel Donath)
- [#245] Add colormap and PSF inset plotting functions (Axel Donath)
- [#244] Add 3FGL to dataset fetch functions (Manuel Paz Arribas)
- [#236] Add likelihood converter function (Christoph Deil)
- [#235] Add some catalog utilities (Christoph Deil)
- [#234] Add multi-scale TS image computation (Axel Donath)
- [#231] Add observatory and data classes (Christoph Deil)
- [#230] Use setuptools entry_points for scripts (Christoph Deil)
- [#225] Misc cleanup (Christoph Deil)
- [#221] TS map calculation update and docs (Axel Donath)
- [#215] Restructure TS map computation (Axel Donath)
- [#212] Bundle xmltodict.py in gammapy/extern (Christoph Deil)
- [#210] Restructure image measurement functions (Axel Donath)
- [#205] Remove healpix_to_image function (moved to reproject repo) (Christoph Deil)
- [#200] Fix quantity errors from astro source models (Christoph Deil)
- [#194] Bundle TeVCat in gammapy.datasets (Christoph Deil)
- [#191] Add Fermi PSF dataset and example (Ellis Owen)
- [#188] Add tests for spectral_cube.integral_flux_image (Ellis Owen)
- [#187] Fix bugs in spectral cube class (Ellis Owen)
- [#186] Add iterative kernel background estimator (Ellis Owen)

.. _gammapy_0p1_release:

0.1 (Aug 25, 2014)
------------------

Summary
+++++++

- Released Aug 25, 2014
- 5 contributors
- 15 months of work
- 82 pull requests
- Requires Astropy version 0.4 or later.

**Contributors:**

- Rolf Bühler
- Christoph Deil
- Axel Donath
- Ellis Owen
- Régis Terrier

Pull requests
+++++++++++++

Note that Gammapy development started out directly in the master branch,
i.e. for some things there is no pull request we can list here.

- [#180] Clean up datasets code and docs (Christoph Deil)
- [#177] Misc code and docs cleanup (Christoph Deil)
- [#176] Add new gammapy.data sub-package (Christoph Deil)
- [#167] Add image profile function (Ellis Owen)
- [#166] Add SED from Cube function (Ellis Owen)
- [#160] Add code to make model images from a source catalog (Ellis Owen)
- [#157] Re-write Galaxy modeling code (Axel Donath)
- [#156] Add Fermi Vela dataset (Ellis Owen)
- [#155] Add PSF convolve function (Ellis Owen)
- [#154] Add Fermi PSF convolution method (Ellis Owen)
- [#151] Improve npred cube functionality (Ellis Owen)
- [#150] Add npred cube computation (Christoph Deil and Ellis Owen)
- [#142] Add EffectiveAreaTable and EnergyDependentMultiGaussPSF classes (Axel Donath)
- [#138] Add Crab flux point dataset (Rolf Bühler)
- [#128] Add flux point computation using Lafferty & Wyatt (1995) (Ellis Owen)
- [#122] Add morphology models as Astropy models (Axel Donath)
- [#117] Improve synthetic Milky Way modeling (Christoph Deil)
- [#116] Add Galactic source catalog simulation methods (Christoph Deil)
- [#109] Python 2 / 3 compatibility with a single codebase (Christoph Deil)
- [#103] Add datasets functions to fetch Fermi catalogs (Ellis Owen)
- [#100] Add image plotting routines (Christoph Deil)
- [#96] Add wstat likelihood function for spectra and images (Christoph Deil)
- [#88] Add block reduce function for HDUs (Ellis Owen)
- [#84] Add TablePSF and Fermi PSF (Christoph Deil)
- [#68] Integrate PyFACT functionality in Gammapy (Christoph Deil)
- [#67] Add image measure methods (Christoph Deil)
- [#66] Add plotting module and HESS colormap (Axel Donath)
- [#65] Add model image and image measurement functionality (Axel Donath)
- [#64] Add coordinate string IAU designation format (Christoph Deil)
- [#58] Add per-pixel solid angle function in image utils (Ellis Owen)
- [#48] Add sphere and power-law sampling functions (Christoph Deil)
- [#34] Rename tevpy to gammapy (Christoph Deil)
- [#25] Add continuous wavelet transform class (Régis Terrier)
- [#12] Add coverage reports to continuous integration on coveralls (Christoph Deil)
- [#11] Add blob detection (Axel Donath)
- Rename tevpy to gammapy in `commit 7e955f <https://github.com/cdeil/gammapy/commit/7e955ffae71353f7b10c9de4a69b977e7c036c6d>`__ on Aug 19, 2013 (Christoph Deil)
- Start tevpy repo with `commit 11af4c <https://github.com/gammapy/gammapy/commit/11af4c7436bb79f8e2cae8d0441693232eebe1ba>`__ on May 15, 2013 (Christoph Deil)
