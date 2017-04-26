.. _install-other:

Other package managers
======================

Besides conda, Gammapy and some of the optional dependencies (Sherpa, Astropy-affiliated packages)
as not yet available in other package managers (such as
e.g. `apt-get <https://en.wikipedia.org/wiki/Advanced_Packaging_Tool>`__
or `yum <https://en.wikipedia.org/wiki/Yellowdog_Updater,_Modified>`__ on Linux
or `Macports <https://www.macports.org/>`__
or `homebrew <http://brew.sh/>`__ on Mac.

So installing Gammapy this way is not recommended at this time.
(The recommended method is conda as mentioned above).

Still, it's possible and common on systems where users have root access
to install some of the dependencies using those package managers, and then
to use `pip`_ to do the rest of the installation.

So as a convenience, here we show the commands to install those packages that are available,
so that you don't have to look up the package names.

We do hope this situation will improve in the future as more astronomy packages become
available in those distributions and versions are updated.

apt-get
-------

`apt-get <https://en.wikipedia.org/wiki/Advanced_Packaging_Tool>`__ is a popular package manager on Linux,
e.g. on Debian or Ubuntu.

The following packages are available:

.. code-block:: bash

    sudo apt-get install \
        python3-pip python3-scipy python3-matplotlib python3-skimage python3-sklearn \
        python3-pandas python3-h5py python3-yaml ipython3-notebook python3-uncertainties \
        python3-astropy python3-click

The following packages have to be installed with pip:

.. code-block:: bash

    pip3 install --user \
        gammapy naima photutils reproject gwcs astroplan \
        iminuit emcee healpy

Sherpa currently doesn't work on Python 3.
You could try to use Python 2 and pip-installing Sherpa (don't know if that works).

A Debian package for Sherpa is in preparation: https://github.com/sherpa/sherpa/issues/75

A Debian package for Gammapy is in preparation: https://github.com/gammapy/gammapy/issues/324

As far as I can see there's no HEALPIX or healpy package.

yum
---

`yum <https://en.wikipedia.org/wiki/Yellowdog_Updater,_Modified>`__ is a popular package manager on Linux,
e.g. on Scientific linux or Red Hat Linux.

If you are a ``yum`` user, please contribute the equivalent commands
(see e.g. the Macports section below).

Homebrew
--------

`Homebrew <http://brew.sh/>`_ is a popular package manager on Mac.

If you are a ``brew`` user, please contribute the equivalent commands
(see e.g. the Macports section above).


Fermi ScienceTools
------------------

The `Fermi ScienceTools <http://fermi.gsfc.nasa.gov/ssc/data/analysis/software/>`__
ships with it's own Python 2.7 interpreter.

If you want to use Astropy or Gammapy with that Python, you have to install it using
that Python interpreter, other existing Python interpreters or installed packages
can't be used (when they have C extensions, like Astropy does).

Fermi ScienceTools version ``v10r0p5`` (released Jun 24, 2015) includes
Python 2.7.8, Numpy 1.9.1, Scipy 0.14.0, matplotlib 1.1.1, PyFITS 3.1.2.
Unfortunately pip, ipython or Astropy are not included.

So first in stall `pip`_ (see
`instructions <https://pip.pypa.io/en/latest/installing.html#install-pip>`__),
and then

.. code-block:: bash

   $ pip install ipython astropy gammapy

If this doesn't work (which is not uncommon, this is known to fail to compile the C
extensions of Astropy on some platforms), ask your Python-installation-savvy co-worker
or on the Astropy or Gammapy mailing list.
