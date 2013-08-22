.. _introduction:

Introduction
============

Show me some code!
------------------

`gammapy` gives you easy access to some frequently used methods in TeV gamma-ray astronomy from Python.

What's the statistical significance when 10 events have been observed with a known background level of 4.2
according to [LiMa1983]_?
`gammapy.stats` knows::

   >>> from gammapy.stats import significance
   >>> significance(n_observed=10, mu_background=4.2, method='lima')
   2.3979181291475453

What's the differential gamma-ray flux and spectral index of the Crab nebula at 3 TeV
according to [Meyer2010]_?
`gammapy.spec` knows::

   >>> from gammapy.spec import crab
   >>> energy = 3
   >>> crab.diff_flux(energy, ref='meyer')
   1.8993523278650278e-12
   >>> crab.spectral_index(energy, ref='meyer')
   2.6763224503600429

All functionality is in subpackages (e.g. `gammapy.stats` or `gammapy.spec`) ...
browse their docs (see list below) to see if it contains the methods you want.

But I don't know how to code in Python!
---------------------------------------

Hmm ... OK.

Some of the `gammapy` functionality can be called from command line tools.

But, to be honest, if you're an astronomer, you should learn to code in Python.
Start at http://python4astronomers.github.io or `here <http://www.astropy.org>`_  

For example, if you have a counts and background model image and would like to compute
a significance image with a correlation radius of 0.1 deg::

   $ gp-make-derived-maps --in counts.fits background.fits \
                           --meaning counts background
                           --out significance.fits
                           --correlation_radius 0.1

Say you have an image that contains the
`Crab nebula <http://en.wikipedia.org/wiki/Crab_Nebula>`_
and want to look up the map value at the Crab position 
(name lookup is done with `SIMBAD <http://simbad.u-strasbg.fr/simbad/>`_)::

   $ gp-lookup-map-values crab_image.fits --object "Crab"

You can call `tool --help` for any tool.

A full list of available command line tools can be found in `tools`.

Other related packages
----------------------

There are several other great open source packages for gamma-ray data analysis (alphabetical order):

* `act-analysis`_ --- a similar package as ``gammapy`` by Karl Kosack
* `gammafits`_ --- an SED modeling and fitting package by Victor Zabalza
* `gammalib`_ and `ctools`_ --- Gamma-ray data analysis library and tools by Jürgen Knödlseder
* `gamma-speed`_ --- benchmarking of TeV data analysis tools by Andrei Ignat
* `PyFACT`_ --- a similar package as ``gammapy`` by Martin Raue

.. _act-analysis: https://bitbucket.org/kosack/act-analysis
.. _PyFACT: http://pyfact.readthedocs.org
.. _gammafits: https://github.com/zblz/gammafits
.. _GammaLib: http://gammalib.sourceforge.net
.. _ctools: http://cta.irap.omp.eu/ctools/
.. _gamma-speed: https://github.com/gammapy/gamma-speed
