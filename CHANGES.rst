
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

- Manuel Paz Arribas
- Christoph Deil
- Axel Donath
- Johannes King

Pull requests
+++++++++++++

- Make background cube models [#319] (Manuel Paz Arribas)
- Production of true/reco bg cube models should use the same model [#335] (Manuel Paz Arribas)
- Fix TS map boundary handling [#332] (Axel Donath)
- Fix sherpa morphology fitting script [#337] (Axel Donath)
- Add example script to produce true and reco bg cube model [#338] (Manuel Paz Arribas)
- Remove PoissonLikelihoodFitter [#339] (Christoph Deil)
- Add EnergyDispersion and CountsSpectrum [#330] (Johannes King)
- Add nice true-reco bg cube model comparison plot script to the high-level docs [#347] (Manuel Paz Arribas)

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

- EventList class fixes and new features [#256] (Christoph Deil)
- Add offset-dependent effective area IRF class [#260] (Johannes King)
- Fix spiral arm model bar radius [#261] (Stefan Klepser)
- Add SNRCat dataset access function [#262] (Christoph Deil)
- Add hspec - spectral analysis using Sherpa [#264] (Regis Terrier, Ignasi Reichardt, Christoph Deil)
- Improve SNRcat dataset [#279] (Christoph Deil)
- Add new subpackage: gammapy.time [#280] (Christoph Deil)
- Document observation tables and improve ``gammapy.obs`` [#278] (Manuel Paz Arribas)
- Add infos how to handle times in Gammapy [#285] (Christoph Deil)
- Function to plot Fermi 3FGL light curves [#286] (Jonathan Harris)
- Add EffectiveAreaTable exporter to EffectiveAreaTable2D [#276] (Johannes King)
- Add interface to HESS FitSpectrum JSON output [#296] (Christoph Deil)
- Remove gammapy.shower package [#291] (Christoph Deil)
- Add cube background model class [#299] (Manuel Paz Arribas)
- Use assert_quantity_allclose from Astropy [#306] (Manuel Paz Arribas)
- Consistent random number handling and improve sample_sphere [#283] (Manuel Paz Arribas)
- Simplified attribute docstrings [#301] (Manuel Paz Arribas)
- Add a simple Makefile for Gammapy. [#289] (Manuel Paz Arribas)
- Observation table subset selection [#295] (Manuel Paz Arribas)
- Set up and document logging for Gammapy [#318] (Christoph Deil)
- Using consistent plotting style in docs [#317] (Axel Donath) 
- Add an "About Gammapy" page to the docs [#312] (Christoph Deil)
- Fix Debian install instructions [#326] (Victor Zabalza)
- Fixed writing TS map headers [#273] (Axel Donath)
- Changed temporary file handling in tests [#273] (Axel Donath)

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

- Add iterative kernel background estimator [#186] (Ellis Owen)
- Fix bugs in spectral cube class [#187] (Ellis Owen)
- Add tests for spectral_cube.integral_flux_image [#188] (Ellis Owen)
- Add Fermi PSF dataset and example [#191] (Ellis Owen)
- Bundle TeVCat in gammapy.datasets [#194] (Christoph Deil)
- Fix quantity errors from astro source models [#200] (Christoph Deil)
- Remove healpix_to_image function (moved to reproject repo) [#205] (Christoph Deil)
- Restructure image measurement functions [#210] (Axel Donath)
- Bundle xmltodict.py in gammapy/extern [#212] (Christoph Deil)
- Restructure TS map computation [#215] (Axel Donath)
- TS map calculation update and docs [#221] (Axel Donath)
- Misc cleanup [#225] (Christoph Deil)
- Use setuptools entry_points for scripts [#230] (Christoph Deil)
- Add observatory and data classes [#231] (Christoph Deil)
- Add multi-scale TS image computation [#234] (Axel Donath)
- Add some catalog utilities [#235] (Christoph Deil)
- Add likelihood converter function [#236] (Christoph Deil)
- Add 3FGL to dataset fetch functions [#244] (Manuel Paz Arribas)
- Add colormap and PSF inset plotting functions [#245] (Axel Donath)
- Add catalog and plotting utils [#246] (Axel Donath)
- Various fixes to image utils docstrings [#247] (Manuel Paz Arribas)
- Add function to fill acceptance image from curve [#248] (Manuel Paz Arribas)
- Add data store and observation table classes, improve event list classes [#249] (Christoph Deil)
- Implement TS map computation in Cython [#252] (Axel Donath)
- Add changelog for Gammapy [#254] (Christoph Deil)

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

- Start tevpy repo with `commit 11af4c <https://github.com/gammapy/gammapy/commit/11af4c7436bb79f8e2cae8d0441693232eebe1ba>`__ (Christoph Deil)
- Rename tevpy to Gammapy in `commit 7e955f <https://github.com/cdeil/gammapy/commit/7e955ffae71353f7b10c9de4a69b977e7c036c6d>`__ on Aug 19, 2013 (Christoph Deil)
- Add blob detection [#11] (Axel Donath)
- Add coverage reports to continuous integration on coveralls [#12] (Christoph Deil)
- Add continuous wavelet transform class [#25] (Regis Terrier)
- Rename tevpy to gammapy [#34] (Christoph Deil)
- Add sphere and power-law sampling functions [#48] (Christoph Deil)
- Add per-pixel solid angle function in image utils [#58] (Ellis Owen)
- Add coordinate string IAU designation format [#64] (Christoph Deil)
- Add model image and image measurement functionality [#65] (Axel Donath)
- Add plotting module and HESS colormap [#66] (Axel Donath)
- Add image measure methods [#67] (Christoph Deil)
- Integrate PyFACT functionality in Gammapy [#68] (Christoph Deil)
- Add TablePSF and Fermi PSF [#84] (Christoph Deil)
- Add block reduce function for HDUs [#88] (Ellis Owen)
- Add wstat likelihood function for spectra and images [#96] (Christoph Deil)
- Add image plotting routines [#100] (Christoph Deil)
- Add datasets functions to fetch Fermi catalogs [#103] (Ellis Owen)
- Python 2 / 3 compatibility with a single codebase [#109] (Christoph Deil)
- Add Galactic source catalog simulation methods [#116] (Christoph Deil)
- Improve synthetic Milky Way modeling [#117] (Christoph Deil)
- Add morphology models as Astropy models [#122] (Axel Donath)
- Add flux point computation using Lafferty & Wyatt (1995) [#128] (Ellis Owen)
- Add Crab flux point dataset [#138] (Rolf Bühler)
- Add EffectiveAreaTable and EnergyDependentMultiGaussPSF classes [#142] (Axel Donath)
- Add npred cube computation [#150] (Christoph Deil and Ellis Owen)
- Improve npred cube functionality [#151] (Ellis Owen)
- Add Fermi PSF convolution method [#154] (Ellis Owen)
- Add PSF convolve function [#155] (Ellis Owen)
- Add Fermi Vela dataset [#156] (Ellis Owen)
- Re-write Galaxy modelling code [#157] (Axel Donath)
- Add code to make model images from a source catalog [#160] (Ellis Owen)
- Add SED from Cube function [#166] (Ellis Owen)
- Add image profile function [#167] (Ellis Owen)
- Add new gammapy.data sub-package [#176] (Christoph Deil)
- Misc code and docs cleanup [#177] (Christoph Deil)
- Clean up datasets code and docs [#180] (Christoph Deil)
