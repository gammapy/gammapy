.. include:: references.txt

.. _install:

Installation
============

Here we provide short installation instructions for Gammapy and its dependencies.

Gammapy works (and is continually tested) with Python 2 and 3 on Linux, Mac OS X and Windows.

More specifically, in the Python 2 series we only support (i.e. test in continuous integration)
Python 2.7, and in the Python 3 series we support version 3.4 or later.
Gammapy will (probably) not work with older Python versions,
such as 2.6 or 3.2 (see :ref:`development-python2and3` if you care why).

Due to the large variety of systems, package managers and setups in us it's not
possible to give a detailed description for every option.

Using `conda`_ is a good option to get everything installed from scratch within minutes.
It works on any Linux, Mac or Windows machine and doesn't require root access.

If you get stuck, have a look at the extensive installation instructions for Astropy
at http://www.astropy.org/ or ask on the `Gammapy mailing list`_.

The main way to improve the instructions is via your feedback!

Conda
-----

To install the latest Gammapy **stable** version as well as the most common
optional dependencies for Gammapy, first install `Anaconda <http://continuum.io/downloads>`__
and then run these commands:

.. code-block:: bash

    conda config --add channels astropy --add channels sherpa
    conda install gammapy naima \
        scipy matplotlib ipython-notebook \
        cython click

We strongly recommend that you install the optional dependencies of Gammapy to have the full
functionality available:

.. code-block:: bash

    conda install \
        scikit-image scikit-learn h5py pandas \
        aplpy photutils reproject

    pip install iminuit

Sherpa is the only Gammapy dependency that's not yet available on Python 3, so if you want
to use Sherpa for modeling / fitting, install Anaconda Python 2 and

.. code-block:: bash

    conda install sherpa

For a quick (depending on your download and disk speed, usually a few minutes),
non-interactive install of `Miniconda <http://conda.pydata.org/miniconda.html>`__
and Gammapy from scratch, use the commands from this script:
`gammapy-conda-install.sh <https://raw.githubusercontent.com/gammapy/gammapy/master/gammapy-conda-install.sh>`__.

Executing it like this should also work:

.. code-block:: bash

    bash "$(curl -fsSL https://raw.githubusercontent.com/gammapy/gammapy/master/gammapy-conda-install.sh)"

To update to the latest version:

.. code-block:: bash

    conda update --all
    conda update gammapy

Overall ``conda`` is a great cross-platform package manager, you can quickly learn how to use
it by reading the docs `here <http://conda.pydata.org/docs/>`__.

pip
---

To install the latest Gammapy **stable** version (see `Gammapy page on PyPI`_) using `pip`_:

.. code-block:: bash

   $ pip install gammapy

To install the current Gammapy **development** version using `pip`_:

.. code-block:: bash

   $ pip install git+https://github.com/gammapy/gammapy.git#egg=gammapy

If that doesn't work because the download from PyPI or Github is blocked by your network,
but you have some other means of copying files onto that machine,
you can get the tarball (``.tar.gz`` file) from PyPI or ``.zip`` file from Github, and then
``pip install <filename>``.

setup.py
--------

To download the latest development version of Gammapy:

.. code-block:: bash

   $ git clone https://github.com/gammapy/gammapy.git
   $ cd gammapy

Now you install, run tests or build the documentation:

.. code-block:: bash

   $ python setup.py install
   $ python setup.py test
   $ python setup.py build_docs

Also you have easy access to the Python scripts from the tutorials and examples: 

.. code-block:: bash

   $ cd docs/tutorials
   $ cd examples

If you want to contribute to Gammapy, but are not familiar with Python or
git or Astropy yet, please have a look at the  
`Astropy developer documentation <http://docs.astropy.org/en/latest/#developer-documentation>`__.

Other package managers
----------------------

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
+++++++

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
+++

`yum <https://en.wikipedia.org/wiki/Yellowdog_Updater,_Modified>`__ is a popular package manager on Linux,
e.g. on Scientific linux or Red Hat Linux.

If you are a ``yum`` user, please contribute the equivalent commands
(see e.g. the Macports section below).

Macports
++++++++

`Macports <https://www.macports.org/>`__ is a popular package manager on Mac.
Gammapy is available via Macports.

To install Gammapy and it's core dependencies:

.. code-block:: bash

    sudo port install py35-gammapy

The commands to update Gammapy and it's dependencies to the latest stable versions are:

.. code-block:: bash

    sudo port selfupdate
    sudo port upgrade outdated

The rest of this section is a quick crash course about Macports, to explain the most common
commands and how to set up and check things. There's not really anything Gammapy-specific
here, but we thought it might be useful to summarise this information for Macports users here.

To check that Gammapy is installed, and which version you have:

.. code-block:: bash

    port installed '*gammapy'
    /opt/local/bin/python3.5 -c 'import gammapy; print(gammapy.__version__)'

Macports supports several versions of Python, so you can choose the one you want.
Parallel installation of multiple Python versions works well, but is only really useful for developers.
So if you want Python 2.7 or Python 3.6, you would have to adapt the commands given in this section
to use that version number instead. If you're not sure which version to use, at this time (January 2017)
we recommend you choose Python 3.5 (because Python 3 is the future, and 3.6 was just released and there
are still a few minor issues being ironed out).

Usually if you're using Macports, you will add this line to your ``~/.profile`` file:

.. code-block:: bash

    export PATH="/opt/local/bin:/opt/local/sbin:$PATH"

This means that you can just execute Python via ``python3.5`` and will get the Macports Python
(and not some other Python, like e.g. the system Python in ``/usr/bin`` or an Anaconda Python in ``$HOME``).

Macports also has a convenience command ``port select`` built in to select a given Python version:

.. code-block:: bash

    sudo port select python python35

This will create a symbolic link ``/opt/local/bin/python -> /opt/local/bin/python3.5`` and means that
now if you execute ``python``, you will get the Macports Python 3.5.
If you're not sure what your configuration is, you can use these commands to find out:

.. code-block:: bash

    port select --summary # show selection and list other things where one can select a default version
    which python
    ls -l `which python`
    python --version

From here on out, we assume that you've done this setup and ``python`` is the correct Python you want to use.

Many other software, including several optional dependencies of Gammapy, is available via Macports.
Here's some examples for some scientific computing and astronomy packages:

.. code-block:: bash

    sudo port install \
        py35-pip py35-pytest \
        py35-scipy py35-matplotlib py35-scikit-image py35-scikit-learn \
        py35-pandas py35-emcee py35-h5py py35-ipython py35-uncertainties \
        py35-healpy py35-cython

To search which software is available in Macports (searches package name and description):

.. code-block:: bash

    port search <name>

There are about 100,000 Python packages on `PyPI`_. Many of those aren't re-packaged and available in Macports,
and some are outdated (although usually Macports packages are updated within days or weeks of the release
of new package versions).

Using the Macports Python as the basis, you can use the Macports pip to install more Python packages.
The default should be to use Macports and to only pip install what's not available there,
because then updates usually just work (see commands above), whereas with pip it's usually a more manual process.

.. code-block:: bash

    python -m pip install --no-deps --user \
        naima photutils reproject astroplan iminuit


There's a few things worth pointing out about how we execute ``pip`` to install packages:

* Instead of using the command line tool ``pip``, we're executing via ``python -m pip``.
  This is because users frequently accidentally execute the wrong pip (e.g. from system Python or Anaconda)
  that happens to be on their ``$PATH`` and then either the install fails, or it succeeds but then
  trying to import the package fails because it's in a ``site-packages`` folder that's unrelated
  to the ``python`` they are using.
* The ``--no-deps`` option instructs ``pip`` to not recursively fetch and install all dependencies.
  Of course, auto-installing all dependencies can be convenient, but it also often happens that
  this leads to the installation of many packages (e.g. Numpy, Scipy, ....) and is not what you want.
  So being explicit about which packages to install is the safer thing to do here.
* We're not using ``sudo`` here and we are using the ``--user`` option. Using ``sudo python -m pip install``
  would result in the installation of packages in
  ``opt/local/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages``,
  the ``site-packages`` folder where Macports installs packages.
  This will usually work, but can then cause problems later on when you try to upgrade or add packages
  via ``sudo port install``. Macports updates work so well because it is very well organised and e.g. keeps
  manifests of all files installed (you can list them with ``port contents py35-gammapy``). So basically,
  to not mess with this, you should never touch files in ``/opt/local`` except through ``port`` commands.
  The ``--user`` option of ``pip`` means "install in my user site-packages folder", which at this time
  on macOS is ``/Users/<username>/Library/Python/3.5/lib/python/site-packages`` and is by default on the
  list of folders searched by Python to find packages to import.

To uninstall Python packages:

.. code-block:: bash

    sudo port uninstall <packagename>
    pip uninstall <packagename>

To check where a given package you're using is installed:

.. code-block:: bash

    python -c 'import numpy; print(numpy.__file__)'
    python -c 'import gammapy; print(gammapy.__file__)'

Homebrew
++++++++

`Homebrew <http://brew.sh/>`_ is a popular package manager on Mac.

If you are a ``brew`` user, please contribute the equivalent commands
(see e.g. the Macports section above).


Check Gammapy installation
--------------------------

To check if Gammapy is correctly installed, start up python or ipython,
import Gammapy and run the unit tests:

.. code-block:: bash

   $ python -c 'import gammapy; gammapy.test()'

To check if the Gammapy command line tools are on your ``$PATH`` try this:

.. code-block:: bash

   $ gammapy-info --tools

To check which dependencies of Gammapy you have installed:

.. code-block:: bash

   $ gammapy-info --dependencies

.. _install-issues:

Common issues
-------------

If you have an issue with Gammapy installation or usage, please check
this list. If your issue is not adressed, please send an email to the
mailing list.

- Q: I get an error mentioning something (e.g. Astropy) isn't available,
  but I did install it.

  A: Check that you're using the right ``python`` and that your
  ``PYTHONPATH`` isn't pointing to places that aren't appropriate
  for this Python (usually it's best to not set it at all)
  using these commands:

  .. code-block:: bash

      which python
      echo $PYTHONPATH
      python -c 'import astropy'

.. _install-dependencies:

Dependencies
------------

.. note:: The philosophy of Gammapy is to build on the existing scientific Python stack.
   This means that you need to install those dependencies to use Gammapy.

   We are aware that too many dependencies is an issue for deployment and maintenance.
   That's why currently Gammapy only has two core dependencies --- Numpy and Astropy.
   We are considering making Sherpa, Scipy, scikit-image, photutils, reproject and naima
   core dependencies.

   In addition there are about a dozen optional dependencies that are OK to import
   from Gammapy because they are potentially useful (not all of those are
   actually currently imported).

   Before the Gammapy 1.0 release we will re-evaluate and clarify the Gammapy dependencies.

The required core dependencies of Gammapy are:

* `Numpy`_ - the fundamental package for scientific computing with Python
* `Astropy`_ - the core package for Astronomy in Python
* `regions`_ - Astropy regions package. Planned for inclusion in Astropy core as `astropy.regions`.
* `click`_ for making command line tools

We're currently using

* `PyYAML`_ for `YAML <http://en.wikipedia.org/wiki/YAML>`__ data handling (config and results files)
* `flask`_ and some Flask plugins for Gammapy web apps

Currently optional dependencies that are being considered as core dependencies:

* `Sherpa`_ for modeling / fitting (doesn't work with Python 3 yet)
* `scipy library`_ for numerical methods
* `scikit-image`_ for some image processing tasks
* `photutils`_ for image photometry
* `reproject`_ for image reprojection
* `naima`_ for SED modeling

Allowed optional dependencies:

* `matplotlib`_ for plotting
* `aplpy`_ for sky image plotting (provides a high-level API)
* `pandas`_ CSV read / write; DataFrame
* `scikit-learn`_ for some data analysis tasks
* `GammaLib`_ and `ctools`_ for simulating data and likelihood fitting
* `ROOT`_ and `rootpy`_ conversion helper functions (still has some Python 3 issues)
* `uncertainties`_ for linear error propagation
* `gwcs`_ for generalised world coordinate transformations
* `astroplan`_ for observation planning and scheduling
* `iminuit`_ for fitting by optimization
* `emcee`_ for fitting by MCMC sampling
* `h5py`_ for `HDF5 <http://en.wikipedia.org/wiki/Hierarchical_Data_Format>`__ data handling
* `healpy`_ for `HEALPIX <http://healpix.jpl.nasa.gov/>`__ data handling


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
