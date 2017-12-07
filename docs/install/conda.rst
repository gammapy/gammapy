.. _install-conda:

Installation with conda
=======================

To install the latest Gammapy **stable** version as well as the most common
optional dependencies for Gammapy, first install `Anaconda <http://continuum.io/downloads>`__
and then run these commands:

.. code-block:: bash

    conda config --add channels conda-forge --add channels sherpa
    conda install gammapy naima \
        scipy matplotlib ipython-notebook \
        cython click

We strongly recommend that you install the optional dependencies of Gammapy to have the full
functionality available:

.. code-block:: bash

    conda install \
        scikit-image scikit-learn h5py pandas \
        aplpy photutils reproject

    python -m pip install iminuit

Sherpa is the only Gammapy dependency that's not yet available on Python 3, so if you want
to use Sherpa for modeling / fitting, install Anaconda Python 2 and

.. code-block:: bash

    conda install sherpa

To update to the latest version:

.. code-block:: bash

    conda update --all
    conda update gammapy

Overall ``conda`` is a great cross-platform package manager, you can quickly learn how to use
it by reading the docs `here <http://conda.pydata.org/docs/>`__.
