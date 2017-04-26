.. _install-macports:

Installation with Macports
==========================

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
