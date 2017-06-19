.. _install-conda:

Installation with conda
=======================

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
