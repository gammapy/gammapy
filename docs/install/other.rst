.. include:: ../references.txt

.. _install-other:

Other package managers
======================

Besides conda, Gammapy and some of the optional dependencies (Sherpa, Astropy-affiliated packages)
are not yet available in other package managers, such as e.g. `apt-get`_ or `yum`_ on Linux
or `Macports`_ or `Homebrew`_ on Mac.

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

On Ubuntu or Debian Linux, you can use `apt-get`_ and `pip`_ to install Gammapy and it's dependencies.

The following packages are available:

.. code-block:: bash

    sudo apt-get install \
        python3-pip python3-scipy python3-matplotlib python3-skimage python3-sklearn \
        python3-pandas python3-h5py python3-yaml ipython3-notebook python3-uncertainties \
        python3-astropy python3-click

The following packages have to be installed with pip:

.. code-block:: bash

    python3 -m pip install --user \
        gammapy naima photutils reproject \
        iminuit emcee healpy sherpa

Another option to install software on Debian (and any system) is to use conda.

yum
---

`yum`_ is a popular package manager on Linux, e.g. on Scientific linux or Red Hat Linux.

If you are a ``yum`` user, please contribute the equivalent commands
(see e.g. the Macports section below).

Homebrew
--------

`Homebrew`_ is a popular package manager on Mac.

Gammapy currently isn't packaged with Homebrew. It should be possible to install
Python / pip / Numpy / Astropy with ``brew`` and then to install Gammapy with ``pip``.

If you're a ``brew`` user, please let us know if it works and what the exact commands are.

Note that we have some Gammapy developers and users on Mac that use Macports.
For this you can find detailed instructions here: :ref:`install-macports`

Fermi ScienceTools
------------------

The `Fermi ScienceTools`_ ships with it's own Python 2.7 interpreter.

If you want to use Astropy or Gammapy with that Python, you have to install it using
that Python interpreter, other existing Python interpreters or installed packages
can't be used (when they have C extensions, like Astropy does).

Fermi ScienceTools version ``v10r0p5`` (released Jun 24, 2015) includes
Python 2.7.8, Numpy 1.9.1, Scipy 0.14.0, matplotlib 1.1.1, PyFITS 3.1.2.
Unfortunately pip, ipython or Astropy are not included.

So first in stall `pip`_ (see `pip install instructions`_), and then

.. code-block:: bash

   $ python -m pip install ipython astropy gammapy

If this doesn't work (which is not uncommon, this is known to fail to compile the C
extensions of Astropy on some platforms), ask your Python-installation-savvy co-worker
or on the Astropy or Gammapy mailing list.
