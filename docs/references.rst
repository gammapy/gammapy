.. include:: references.txt

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

.. [FSSC2013] `Fermi LAT Collaboration (2013) <https://fermi.gsfc.nasa.gov/ssc/data/analysis/scitools/overview.html>`_
   "Science Tools: LAT Data Analysis Tools"

.. [Mayer2015] `Michael Mayer (2015) <https://publishup.uni-potsdam.de/frontdoor/index/index/docId/7150>`_
   "Pulsar wind nebulae at high energies"

.. _glossary:

Glossary
--------

.. [MET] mission elapsed time; see also :ref:`MET_definition` in :ref:`time_handling`.

Other gamma-ray packages
------------------------

Here are some other software packages for gamma-ray astronomy:

* `Gammalib`_ /`ctools`_ is a C++ package with Python wrapper, similar to the Fermi-LAT ScienceTools,
  that to a large degree uses the same input data formats as Gammapy.
* `3ML`_ is a Python package that uses existing packages (e.g. the Fermi-LAT ScienceTools or the HAWC software)
  to deal with the data and IRFs and compute the likelihood for a given model.
* `Sherpa`_ --- X-ray modeling and fitting package by the Chandra X-ray Center
* `ctapipe`_ --- CTA Python pipeline experimental version
* `FermiPy`_ --- Fermi-LAT science tools high-level Python interface by Matthew Wood
* `gammatools`_ --- Python tools for Fermi-LAT gamma-ray data analysis by Matthew Wood
* `pointlike`_ -- Fermi-LAT science tools alternative by Toby Burnett
* `naima`_ --- an SED modeling and fitting package by Victor Zabalza
* `Gamera`_ --- a C++ gamma-ray source modeling package (SED, SNR model, Galactic population model) with a Python wrapper called Gappa by Joachim Hahn
* `FLaapLUC`_ --- Fermi/LAT automatic aperture photometry Light C<->Urve pipeline by Jean-Philippe Lenain
* http://voparis-cta-client.obspm.fr/ --- prototype web app for CTA data access / analysis, not open source.
* `act-analysis`_ --- Python scripts and Makefiles for some common gamma-ray data analysis tasks by Karl Kosack
* `VHEObserverTools`_ --- tools to predict detectability at VHE by Jeremy Perkins
* `photon_simulator`_ --- Python code to simulate X-ray observations
* `pycrflux`_ --- Python module to plot cosmic-ray flux
* Andy strong has C++ codes (GALPROP and Galplot) for Galactic cosmic rays and emission
  and source population synthesis at http://www.mpe.mpg.de/~aws/propagate.html .

Other useful packages
---------------------

In addition to the packages mentioned in the last section and at :ref:`install-dependencies`,
here's a few other Python packages you might find useful / interesting:

* See the list here: http://www.astropy.org/affiliated/
* Pulsar timing package `PINT`_
* `iminuit`_ fitter and `probfit`_ likelihood function builder.
