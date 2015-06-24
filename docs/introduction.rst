.. _introduction:

Introduction
============

Show me some code!
------------------

Gammapy gives you easy access to some frequently used methods in TeV gamma-ray astronomy from Python.

What's the statistical significance when 10 events have been observed with a known background level of 4.2
according to [LiMa1983]_?

Call the `~gammapy.stats.significance` function:

.. code-block:: python

   >>> from gammapy.stats import significance
   >>> significance(n_observed=10, mu_background=4.2, method='lima')
   2.3979181291475453

What's the differential gamma-ray flux and spectral index of the Crab nebula at 3 TeV
according to [Meyer2010]_?

Call the `~gammapy.spectrum.crab_flux` and `~gammapy.spectrum.crab_spectral_index` functions:

.. code-block:: python

   >>> from gammapy.spectrum import crab_flux, crab_spectral_index
   >>> energy = 3
   >>> crab_flux(energy, reference='meyer')
   1.8993523278650278e-12
   >>> crab_spectral_index(energy, reference='meyer')
   2.6763224503600429

All functionality is in subpackages (e.g. `gammapy.stats` or `gammapy.spectrum`) ...
browse their docs to see if it contains the methods you want.

But I don't know how to code in Python!
---------------------------------------

Hmm ... OK.

Some of the Gammapy functionality can be called from command line tools.

But, to be honest, if you're an astronomer, you should learn to code in Python.
Start at http://python4astronomers.github.io or the `astropy` docs.

For example, if you have a counts and background model image and would like to compute
a significance image with a correlation radius of 0.1 deg:

.. code-block:: bash

   $ gammapy-make-derived-maps --in counts.fits background.fits \
                               --meaning counts background \
                               --out significance.fits \
                               --correlation_radius 0.1

Say you have an image that contains the
`Crab nebula <http://en.wikipedia.org/wiki/Crab_Nebula>`_
and want to look up the map value at the Crab position
(name lookup is done with `SIMBAD <http://simbad.u-strasbg.fr/simbad/>`_):

.. code-block:: bash

   $ gammapy-lookup-map-values crab_image.fits --object "Crab"

You can call ``gammapy-tool-name --help`` or ``gammapy-tool-name -h`` for any tool.

A full list of available command line tools can be found in TODO: ``tools``.

Other related packages
----------------------

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