.. include:: references.txt

.. _getting-started:

Getting Started
===============

Introduction
------------

New to Gammapy? You came to the right place!

Gammapy is a Python package, consisting of functions and classes, that you can use
as a flexible and extensible toolbox to implement and execute exactly the analysis you want.

In addition, Gammapy provides some command line tools
(and we're starting to add web apps with a graphical user interface)
that make it easy to perform common tasks simply by specifying a command and some parameters as arguments,
no Python programming needed.

This is a 5 minute tutorial that shows you how get started using Gammapy as a package or command line tool.

If you'd like to follow along, make sure you have Gammapy installed (see :ref:`installation`).

If you're new to Python for gamma-ray astronomy and would like to learn the basics, we recommend
you go to the `Scipy Lecture Notes`_ or the `Practical Python for Astronomers Tutorial`_.

Using Gammapy as a Python package
---------------------------------

Here's a few very simple examples how to use Gammapy as a Python package.

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
   >>> crab_flux(energy=3, reference='meyer')
   1.8993523278650278e-12
   >>> crab_spectral_index(energy=3, reference='meyer')
   2.6763224503600429

All functionality is in subpackages (e.g. `gammapy.stats` or `gammapy.spectrum`) ...
browse their docs to see if it contains the methods you want.

Using Gammapy as a command line tool
------------------------------------

This section contains a few examples how to use Gammapy as a command line tool.

Say you have a counts and background model image and would like to compute
a significance image with a correlation radius of 0.1 deg:

.. code-block:: bash

   $ gammapy-make-derived-maps --in counts.fits background.fits \
                               --meaning counts background \
                               --out significance.fits \
                               --correlation_radius 0.1

Say you have an image that contains the `Crab nebula`_
and want to look up the map value at the Crab position
(name lookup is done with `SIMBAD`_):

.. code-block:: bash

   $ gammapy-lookup-map-values crab_image.fits --object "Crab"

You can call ``gammapy-tool-name --help`` or ``gammapy-tool-name -h`` for any tool.

More information on tools and a list of all available tools can be found in :ref:`scripts`.

What next?
----------

If you'd like to continue with tutorials to learn Gammapy, go to :ref:`tutorials`.

To learn about some specific functionality that could be useful for your work,
start browsing the "Getting Started" section of Gammapy sub-package that
might be of interest to you (e.g. `gammapy.spectrum`, `gammapy.obs`, `gammapy.catalog`, ...).

Not sure if Gammapy has the feature you want or how to do what you want?
Ask for help on the `Gammapy mailing list`_.

.. _Crab nebula: http://en.wikipedia.org/wiki/Crab_Nebula
.. _SIMBAD: http://simbad.u-strasbg.fr/simbad
