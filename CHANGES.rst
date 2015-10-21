.. _gammapy_1p0_release:

1.0 (Fall 2015 or spring 2016, unreleased)
------------------------------------------

Summary
+++++++

Gammapy 1.0 will be released in fall 2015 or spring 2016.

Gammapy 1.0 will depend on the Astropy 1.1 and Sherpa 4.8.

For plans and progress see https://github.com/gammapy/gammapy/milestones/1.0

.. _gammapy_0p5_release:

0.5 (November 2, 2015, unreleased)
----------------------------------

Summary
+++++++

Gammapy 0.5 will be released on November 2, 2015.

For plans and progress see https://github.com/gammapy/gammapy/milestones/0.5

.. _gammapy_0p4_release:

0.4 (October 5, 2015, unreleased)
---------------------------------

Summary
+++++++

Gammapy 0.4 will be released on October 5, 2015.

For plans and progress see https://github.com/gammapy/gammapy/milestones/0.4

Contributors
++++++++++++

- Axel Donath
- Christoph Deil
- Dirk Lennarz (new)
- Johannes King
- Manuel Paz Arribas

Pull requests
+++++++++++++

- [#367] Add Feldman Cousins algorithm (Dirk Lennarz)
- [#359] Add 1D spectrum analysis tool based on gammapy.hspec (Johannes King)
- [#351] Rewrite most parts of EnergyDispersion class (Johannes King)
- [#353] Add some scripts and examples (Christoph Deil)
- [#352] Add data management tools (Christoph Deil)
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

0.3 (August 13, 2015)
---------------------

Summary
+++++++

- Released on August 13, 2015 (`Gammapy 0.3 on PyPI <https://pypi.python.org/pypi/gammapy/0.3>`__)
- Requires Astropy version 1.0 or later.
- 9 contributors (5 new)
- 4 months of work (from April 13, 2014 to August 13, 2015)
- 24 pull requests
- On-off likelihood spectral analysis was added in ``gammapy.hspec``,
  contributed by Regis Terrier and Ignasi Reichardt.
  It will be refactored and is thus not part of the public API.
- The Gammapy 0.3 release is the basis for an `ICRC 2015 poster contribution <https://indico.cern.ch/event/344485/session/142/contribution/695>`__

Contributors
++++++++++++

- Manuel Paz Arribas
- Christoph Deil
- Axel Donath
- Jonathan Harris (new)
- Johannes King (new)
- Stefan Klepser (new)
- Ignasi Reichardt (new)
- Regis Terrier
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
- [#264] Add hspec - spectral analysis using Sherpa (Regis Terrier, Ignasi Reichardt, Christoph Deil)
- [#262] Add SNRCat dataset access function (Christoph Deil)
- [#261] Fix spiral arm model bar radius (Stefan Klepser)
- [#260] Add offset-dependent effective area IRF class (Johannes King)
- [#256] EventList class fixes and new features (Christoph Deil)

.. _gammapy_0p2_release:

0.2 (April 13, 2015)
--------------------

Summary
+++++++

- Released on April 13, 2015 (`Gammapy 0.2 on PyPI <https://pypi.python.org/pypi/gammapy/0.2>`__)
- Requires Astropy version 1.0 or later.
- Gammapy now uses `Cython <http://cython.org/>`__,
  i.e. requires a C compiler for end-users and in addition Cython for developers.
- 4 contributors (1 new)
- 8 months of work (from August 25, 2014 to April 13, 2015)
- 40 pull requests

Contributors
++++++++++++

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

0.1 (August 25, 2014)
---------------------

Summary
+++++++

- Released on August 25, 2014 (`Gammapy 0.1 on PyPI <https://pypi.python.org/pypi/gammapy/0.1>`__)
- Requires Astropy version 0.4 or later.
- 5 contributors
- 15 months of work (from May 15, 2013 to August 25, 2014)
- 82 pull requests

Contributors
++++++++++++

- Rolf Bühler
- Christoph Deil
- Axel Donath
- Ellis Owen
- Regis Terrier

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
- [#157] Re-write Galaxy modelling code (Axel Donath)
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
- [#25] Add continuous wavelet transform class (Regis Terrier)
- [#12] Add coverage reports to continuous integration on coveralls (Christoph Deil)
- [#11] Add blob detection (Axel Donath)
- Rename tevpy to gammapy in `commit 7e955f <https://github.com/cdeil/gammapy/commit/7e955ffae71353f7b10c9de4a69b977e7c036c6d>`__ on Aug 19, 2013 (Christoph Deil)
- Start tevpy repo with `commit 11af4c <https://github.com/gammapy/gammapy/commit/11af4c7436bb79f8e2cae8d0441693232eebe1ba>`__ on May 15, 2013 (Christoph Deil)
