.. include:: references.txt

.. _getting-started:

Getting Started
===============

If you'd like to get started using Gammapy, you've come to the right place!

Reading through this page will just take you a few minutes.

But we hope that you'll get curious and start executing the examples yourself,
using Gammapy to analyse (simulated) H.E.S.S. and real Fermi-LAT data.

If you're new to Python for gamma-ray astronomy and would like to learn the basics, we recommend
you go to the `Scipy Lecture Notes`_ or the `Practical Python for Astronomers Tutorial`_.

Gammapy as a Python package and set of science tools
----------------------------------------------------

Gammapy is a Python package, consisting of functions and classes, that you can use
as a flexible and extensible toolbox to implement and execute exactly the analysis you want.

On top of that, Gammapy provides some command line tools (sometimes driven by a config file),
and in the future we plan on adding web apps with a graphical user interface.
To use those no Python programming skills are required, you'll just have to specify which
data to analyse, with which method and parameters.

Getting set up
--------------

First, make sure you have Gammapy installed (see :ref:`installation`).

You can use this command to make sure the Python package is available::

   $ python -c 'import gammapy'

To check if the Gammapy command line tools have been installed and are available on your PATH, use this command::

    $ gammapy-info --version

The Gammapy tutorials use some example datasets that are stored in the ``gammapy-extra`` repository on Github.
So please go follow the instructions at :ref:`gammapy-extra` to fetch those, then come back here.

To check if ``gammapy-extra`` is available and the ``GAMMAPY_EXTRA`` shell environment variable set, use this command::

    $ echo $GAMMAPY_EXTRA
    $ ls $GAMMAPY_EXTRA/logo/gammapy_banner.png

Stuck already?
Ask for help on the `Gammapy mailing list`_!

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

All functionality is in subpackages (e.g. `gammapy.stats` or `gammapy.spectrum`).
Just browse this documentation to see if the functionality you are looking for is available.
You can try for example to find a suitable data structure to represent a counts vector,
i.e. list of events binned in energy.

Using Gammapy from the Jupyter notebooks
----------------------------------------

Next we encourage you to have a look at the
`Gammapy Jupyter notebooks <http://nbviewer.jupyter.org/github/gammapy/gammapy-extra/blob/master/notebooks/Index.ipynb>`__.

Jupyter notebooks are documents that combine code input and text and graphical output,
and are wonderful tools to learn and explore (both programming and the data),
and finally to share results with your colleagues.

Using Gammapy via command line tools
------------------------------------

At the moment we are mostly focused on developing the Gammapy Python package.
But we already have a few command line tools to execute common analysis tasks.

At the terminal (the shell, not the Python prompt), you can type ``gammapy<TAB><TAB>`` to get
a list of available command line tools.

.. code-block:: bash

    $ gammapy<TAB><TAB>

You can use the ``--help`` option on each one to get some help what it's arguments and options are.

.. code-block:: bash

    $ gammapy-image-bin --help
    usage: gammapy-image-bin [-h] [--overwrite] event_file reference_file out_file

    Bin events into an image.

    positional arguments:
      event_file      Input FITS event file name
      reference_file  Input FITS reference image file name
      out_file        Output FITS counts cube file name

    optional arguments:
      -h, --help      show this help message and exit
      --overwrite     Overwrite existing output file? (default: False)

Again ... the command line tools don't expose much functionality yet.
We will improve this (and add examples and documentation) for the command line tools soon!

TODO: add example here, maybe for ``gammapy-image-bin`` or something more interesting?


What next?
----------

If you'd like to continue with tutorials to learn Gammapy, go to :ref:`tutorials`.

To learn about some specific functionality that could be useful for your work,
start browsing the "Getting Started" section of Gammapy sub-package that
might be of interest to you (e.g. `gammapy.data`, `gammapy.catalog`, `gammapy.spectrum`, ...).

Not sure if Gammapy has the feature you want or how to do what you want?
Ask for help on the `Gammapy mailing list`_.

.. _Crab nebula: http://en.wikipedia.org/wiki/Crab_Nebula
.. _SIMBAD: http://simbad.u-strasbg.fr/simbad
