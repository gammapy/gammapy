.. _gammapy_1p0_release:

1.0 (unreleased)
----------------

Summary
+++++++

For plans and progress for Gammapy 1.0, see https://github.com/gammapy/gammapy/milestones/1.0

.. _gammapy_0p7_release:

0.7 (unreleased)
----------------

Summary
+++++++

For plans and progress for Gammapy 0.7, see https://github.com/gammapy/gammapy/milestones/0.7


.. _gammapy_0p6_release:

0.6 (Feb 13, 2017)
------------------

Summary
+++++++

- Not released yet. For plans and progress for Gammapy 0.6, see https://github.com/gammapy/gammapy/milestones/0.6
- XXX contributors (XXX new)
- 3 months of work (from November 22, 2016 to February 13, 2017)
- XXX pull requests (not all listed below)

**What's new?**

- Release and installation
    - Until now, we had a roughly bi-yearly release cycle for Gammapy.
      Starting now, we will make stable releases more often, to ship features and fixes to Gammapy users more quickly.
    - Gammapy 0.6 requires Python 2.7 or 3.4+, Numpy 1.8+, Scipy 0.15+, Astropy 1.3+, Sherpa 4.9.0+
      (most things will still work with older Astropy and Sherpa, but we dropped testing
      for older versions from our continuous integration)
    - Gammapy now available via Macports (``port install py35-gammapy``)
- Documentation
    - Many tutorial Jupyter notebooks added and existing ones improved
    - Misc docs improvements and new getting started notebooks
    - Separate section added for developer documentation, and extended developer documentation
- **CTA**
    - tbd: point out docs and examples
    - better support for CTA IRFs
- **Fermi-LAT**
    - reference dataset: https://github.com/gammapy/gammapy-fermi-lat-data
    - data access and first analysis example (tbd)
- ``gammapy.catalog``
    - Access to ``gamma-cat``
    - Access to more Fermi-LAT catalogs (1FHL, 2FHL, 3FHL)
- ``gammapy.spectrum``
    - much improved flux point class
    - EBL-absorbed spectral models
    - Improved spectrum simulation class
    - Misc fixes and improvements (e.g. for energy dispersion)
- ``gammapy.image``
    - image radial and box profiles
- ``gammapy.time``
    - tbd: improved documentation using Fermi-LAT and gamma-cat lightcurves
- ``gammapy.irf``:
    - Many IRF classes now rewritten to use the generic ``NDDataArray`` and axis classes
- ``gammapy.utils``:
    - Add ``gammapy.utils.modeling`` with support for XML models (work in progress)

**Contributors:**

- Arpit Gogia (new)
- Axel Donath
- Brigitta Sipocz
- Christoph Deil
- Dirk Lennarz
- Johannes King
- Julien Lefaucheur
- Lars Mohrmann (new)
- Léa Jouvin
- Nachiketa Chakraborty
- Régis Terrier

Pull requests
+++++++++++++

This list is incomplete. Small improvements and bug fixes are not listed here.

See the complete `Gammapy 0.6 merged pull requests list on Github <https://github.com/gammapy/gammapy/pulls?utf8=%E2%9C%93&q=is%3Apr+milestone%3A0.6+is%3Amerged+>`__.

- [#906] Enable catalog getitem to work with numpy int index (Zé Vinícius)
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

0.5 (November 22, 2016)
-----------------------

Summary
+++++++

- Released on November 22, 2016 (`Gammapy 0.5 on PyPI <https://pypi.python.org/pypi/gammapy/0.5>`__)
- 12 contributors (5 new)
- 7 months of work (from April 20, 2016 to November 22, 2016)
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

0.4 (April 20, 2016)
--------------------

Summary
+++++++

- Released on April 20, 2016 (`Gammapy 0.4 on PyPI <https://pypi.python.org/pypi/gammapy/0.4>`__)
- 10 contributors (5 new)
- 8 months of work (from August 13, 2015 to April 20, 2016)
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

0.3 (August 13, 2015)
---------------------

Summary
+++++++

- Released on August 13, 2015 (`Gammapy 0.3 on PyPI <https://pypi.python.org/pypi/gammapy/0.3>`__)
- 9 contributors (5 new)
- 4 months of work (from April 13, 2014 to August 13, 2015)
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

0.2 (April 13, 2015)
--------------------

Summary
+++++++

- Released on April 13, 2015 (`Gammapy 0.2 on PyPI <https://pypi.python.org/pypi/gammapy/0.2>`__)
- 4 contributors (1 new)
- 8 months of work (from August 25, 2014 to April 13, 2015)
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

0.1 (August 25, 2014)
---------------------

Summary
+++++++

- Released on August 25, 2014 (`Gammapy 0.1 on PyPI <https://pypi.python.org/pypi/gammapy/0.1>`__)
- 5 contributors
- 15 months of work (from May 15, 2013 to August 25, 2014)
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
- [#25] Add continuous wavelet transform class (Régis Terrier)
- [#12] Add coverage reports to continuous integration on coveralls (Christoph Deil)
- [#11] Add blob detection (Axel Donath)
- Rename tevpy to gammapy in `commit 7e955f <https://github.com/cdeil/gammapy/commit/7e955ffae71353f7b10c9de4a69b977e7c036c6d>`__ on Aug 19, 2013 (Christoph Deil)
- Start tevpy repo with `commit 11af4c <https://github.com/gammapy/gammapy/commit/11af4c7436bb79f8e2cae8d0441693232eebe1ba>`__ on May 15, 2013 (Christoph Deil)
