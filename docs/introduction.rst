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
Start at http://python4astronomers.github.io or the Astropy docs.

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
