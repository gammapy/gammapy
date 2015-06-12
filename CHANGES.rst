1.0 (Fall, 2015)
----------------

Gammapy 1.0 will be released in fall 2015.

We aim for October 2015, but are willing to delay a bit if important features are missing
or the documentation and tests need further work.

Gammapy 1.0 will depend on the Astropy 1.1 and Sherpa 4.8.

For plans and progress see https://github.com/gammapy/gammapy/milestones/1.0

Pull requests
+++++++++++++

- No changes yet

0.4 (August 24, 2015)
---------------------

Gammapy 0.4 will be released on August 24, 2015.

For plans and progress see https://github.com/gammapy/gammapy/milestones/0.4

Pull requests
+++++++++++++

- No changes yet

.. _gammapy_0p3_release:

0.3 (July 9 , 2015)
-------------------

Gammapy 0.3 will be released on July 9, 2015.

For plans and progress see https://github.com/gammapy/gammapy/milestones/0.3

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

.. _gammapy_0p2_release:

0.2 (April 13, 2015)
--------------------

Release notes
+++++++++++++

- Released on April 13, 2015 (`Gammapy 0.2 on PyPI <https://pypi.python.org/pypi/gammapy/0.2>`__)
- Contributors: Manuel Paz Arribas (new), Axel Donath, Ellis Owen, Christoph Deil
- 8 months of work (from August 25, 2014 to April 13, 2015)
- 40 pull requests, 4 authors
- Requires Astropy version 1.0 or later.
- Gammapy now uses `Cython <http://cython.org/>`__,
  i.e. requires a C compiler for end-users and in addition Cython for developers.

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

Release notes
+++++++++++++

- Released on August 25, 2014 (`Gammapy 0.1 on PyPI <https://pypi.python.org/pypi/gammapy/0.1>`__)
- Contributors: Axel Donath, Ellis Owen, Regis Terrier, Rolf Bühler, Christoph Deil
- 15 months of work (from May 15, 2013 to August 25, 2014)
- 82 pull requests, 5 authors
- Requires Astropy version 0.4 or later.

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
