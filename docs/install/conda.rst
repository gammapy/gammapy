.. _install-conda:

Installation with conda
=======================

To install the latest Gammapy **stable** version as well as the most common
optional dependencies for Gammapy, first install `Anaconda
<http://continuum.io/downloads>`__ and then run this commands:

.. code-block:: bash

    conda config --add channels conda-forge --add channels sherpa
    conda install gammapy naima sherpa \
        scipy matplotlib ipython-notebook \
        cython click reproject iminuit

To update to the latest version:

.. code-block:: bash

    conda update --all
    conda update gammapy

Overall ``conda`` is a great cross-platform package manager, you can quickly
learn how to use it by reading the `conda docs
<http://conda.pydata.org/docs/>`__.
