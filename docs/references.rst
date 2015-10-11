.. _references:

References
==========

.. _publications:

Publications
------------

This is the bibliography containing the literature references for the implemented methods
referenced from the Gammapy docs.

The best reference to TeV data analysis is Chapter 7 of Mathieu de Naurois's habilitation thesis.

.. [Albert2007] `Albert et al. (2007) <http://adsabs.harvard.edu/abs/2007NIMPA.583..494A>`_,
   "Unfolding of differential energy spectra in the MAGIC experiment",

.. [Berge2007] `Berge et al. (2007) <http://adsabs.harvard.edu/abs/2007A%26A...466.1219B>`_,
   "Background modelling in very-high-energy gamma-ray astronomy"

.. [Cash1979] `Cash (1979) <http://adsabs.harvard.edu/abs/1983ApJ...272..317L>`_,
   "Parameter estimation in astronomy through application of the likelihood ratio"

.. [Cousins2007] `Cousins et al. (2007) <http://adsabs.harvard.edu/abs/2007physics...2156C>`_,
   "Evaluation of three methods for calculating statistical significance when incorporating a
   systematic uncertainty into a test of the background-only hypothesis for a Poisson process"

.. [Feldman1998] `Feldman & Cousins (1998) <http://adsabs.harvard.edu/abs/1998PhRvD..57.3873F>`_,
   "Unified approach to the classical statistical analysis of small signals"
   
.. [Lafferty1994] `Lafferty & Wyatt (1994) <http://adsabs.harvard.edu/abs/1995NIMPA.355..541L>`_,
   "Where to stick your data points: The treatment of measurements within wide bins"

.. [LiMa1983] `Li & Ma (1983) <http://adsabs.harvard.edu/abs/1983ApJ...272..317L>`_,
   "Analysis methods for results in gamma-ray astronomy"

.. [Meyer2010] `Meyer et al. (2010) <http://adsabs.harvard.edu/abs/2010A%26A...523A...2M>`_,
   "The Crab Nebula as a standard candle in very high-energy astrophysics"

.. [Naurois2012] `de Naurois (2012) <http://inspirehep.net/record/1122589>`_,
   "Very High Energy astronomy from H.E.S.S. to CTA. Opening of a new astronomical window on the non-thermal Universe",

.. [Piron2001] `Piron et al. (2001) <http://adsabs.harvard.edu/abs/2001A%26A...374..895P>`_,
   "Temporal and spectral gamma-ray properties of Mkn 421 above 250 GeV from CAT observations between 1996 and 2000",

.. [Rolke2005] `Rolke et al. (2005) <http://adsabs.harvard.edu/abs/2005NIMPA.551..493R>`_,
   "Limits and confidence intervals in the presence of nuisance parameters",

.. [Stewart2009] `Stewart (2009) <http://adsabs.harvard.edu/abs/2009A%26A...495..989S>`_,
   "Maximum-likelihood detection of sources among Poissonian noise"

Software references:

.. [Raue2012] `Raue (2012) <http://adsabs.harvard.edu/abs/2012AIPC.1505..789R>`_,
   "PyFACT: Python and FITS analysis for Cherenkov telescopes"

.. [Robitaille2013] `Robitaille et al. (2013) <http://adsabs.harvard.edu/abs/2013A%26A...558A..33A>`_
   "Astropy: A community Python package for astronomy"

.. [Knoedlseder2012] `Knödlseder et at. (2012) <http://adsabs.harvard.edu/abs/2012ASPC..461...65K>`_
   "GammaLib: A New Framework for the Analysis of Astronomical Gamma-Ray Data"
   
.. [FSSC2013] `Fermi LAT Collaboration (2013) <http://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/overview.html>`_
   "Science Tools: LAT Data Analysis Tools"

.. [Mayer2015] `Michael Mayer (2015) <https://publishup.uni-potsdam.de/frontdoor/index/index/docId/7150>`_
   "Pulsar wind nebulae at high energies"


.. _glossary:

Glossary
--------

.. [CSV] comma-separated values; see also :ref:`CSV_files`.
.. [MET] mission elapsed time; see also :ref:`MET_definition` in :ref:`time_handling`.
.. [RST] restructured text; the markup format used for documentation and docstrings.
         See `here <http://en.wikipedia.org/wiki/ReStructuredText>`__ and `here <http://sphinx-doc.org/rest.html>`__.

Other related packages
----------------------

Besides Gammapy, there are two other major open-source analysis packages (that we are aware of)
for multi-mission gamma-ray likelihood analysis:

- `Gammalib`_/`ctools`_ is a C++ package with Python wrapper, similar to the Fermi-LAT ScienceTools,
  that uses the same input file formats as Gammapy.
- 3ML is a Python package that uses existing packages (e.g. the Fermi-LAT ScienceTools or the HAWC software)
  to deal with the data and IRFs and compute the likelihood for a given model.

So there is some overlap of Gammapy with other efforts, but as mentioned in  :ref:`about-overview` the scope
of Gammapy is larger. And also all packages are new and the implementations radically different, so that's
a good thing when it comes to flexibility and the ability to cross-check results and methods.

Make sure to also check out the following packages that contain very useful functionality for gamma-ray astronomy:

* `Sherpa`_ --- X-ray modeling and fitting package by the Chandra X-ray Center
* `gammalib`_ and `ctools`_ --- Gamma-ray data analysis library and tools by Jürgen Knödlseder
* `ctapipe`_ --- CTA Python pipeline experimental version
* `threeml`_ --- the multi-missing maximum likelihood framework by Giacomo Vianello and others from Stanford
  (`code <https://github.com/giacomov/3ML>`__,
  `example notebook <http://nbviewer.ipython.org/github/giacomov/3ML/blob/master/examples/090217206.ipynb>`__)
* `gammatools`_ --- Python tools for Fermi-LAT gamma-ray data analysis by Matthew Wood
* `naima`_ --- an SED modeling and fitting package by Victor Zabalza
* `GamERa`_ --- a C++ gamma-ray source modeling package (SED, SNR model, Galactic population model) by Joachim Hahn
* `Enrico <https://github.com/gammapy/enrico/>`__ --- helps you with your Fermi data analysis
* http://voparis-cta-client.obspm.fr/ --- prototype web app for CTA data access / analysis, not open source.


.. _ctapipe: https://github.com/cta-observatory/ctapipe
.. _Sherpa: http://cxc.cfa.harvard.edu/sherpa/
.. _GammaLib: http://gammalib.sourceforge.net
.. _ctools: http://cta.irap.omp.eu/ctools/
.. _naima: https://github.com/zblz/naima
.. _GamERa: https://github.com/JoachimHahn/GamERa
.. _gammatools: https://github.com/woodmd/gammatools
.. _threeml: http://threeml.stanford.edu/

Some other projects:

* `act-analysis`_ --- Python scripts and Makefiles for some common gamma-ray data analysis tasks by Karl Kosack
* `VHEObserverTools`_ --- tools to predict detectability at VHE by Jeremy Perkins
* `photon_simulator`_ --- Python code to simulate X-ray observations

.. _act-analysis: https://bitbucket.org/kosack/act-analysis
.. _VHEObserverTools: https://github.com/kialio/VHEObserverTools
.. _photon_simulator: http://yt-project.org/doc/analyzing/analysis_modules/photon_simulator.html

Other useful packages
---------------------

In addition to the packages mentioned in the last section and at :ref:`install-dependencies`,
here's a few other Python packages you might find useful / interesting:

* See the list here: http://www.astropy.org/affiliated/
* Pulsar timing package `PINT <https://github.com/nanograv/PINT>`__
* `iminuit <https://github.com/iminuit/iminuit>`__ fitter and
  `probfit <https://github.com/iminuit/probfit>`__ likelihood function builder.
* Andy strong has C++ codes (GALPROP and Galplot) for Galactic cosmic rays and emission
  and source population synthesis at http://www.mpe.mpg.de/~aws/propagate.html .
