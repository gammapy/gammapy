.. include:: ../references.txt

.. _install:

************
Installation
************

The recommended way to install Gammapy is to install the Anaconda
distribution from https://www.anaconda.com/download/ and then to install the
Gammapy and it's dependencies by executing these commands in a terminal:

.. code-block:: bash

    curl -O https://gammapy.org/download/install/gammapy-0.16-environment.yml
    conda env create -f gammapy-0.16-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.

This creates a conda environment called ``gammapy-0.16`` which you can activate via:

.. code-block:: bash

    conda activate gammapy-0.16

Note that you have to execute that activation command (but not the environment
cration command) every time you open a new terminal window, because the default
with conda is the base environment, which doesn't have Gammapy installed.

To check your Gammapy installation, you can use this command:

.. code-block:: bash

    gammapy info

Congratulations! You are all set to start using Gammapy!

If you're new to Python, ipython and Jupyter, read the :ref:`getting-started`
guide. To learn how to use Gammapy, go to :ref:`tutorials`.

The following pages contain detailed information about Gammapy dependencies in
various installation options:

.. toctree::
    :maxdepth: 1

    dependencies
    pip
    other
