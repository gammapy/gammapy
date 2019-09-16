.. include:: ../references.txt

.. _install:

************
Installation
************

The recommended way to install Gammapy version 0.13 is to install the Anaconda
distribution from https://www.anaconda.com/download/ and then to install the
Gammapy and it's dependencies by executing these commands in a terminal:

.. code-block:: bash

    curl -O https://gammapy.org/download/install/gammapy-0.13-environment.yml
    conda env create -f gammapy-0.13-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and uncomment the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.

This creates a conda environment called ``gammapy-0.13``. To activate it:

.. code-block:: bash

    conda activate gammapy-0.13

Note that you have to execute that activation command every time you open a new
terminal window, because the default with conda is the base environment, which
doesn't have Gammapy installed.

To check your Gammapy installation, you can use this command:

.. code-block:: bash

    gammapy info

Congratulations! You are all set to start using Gammapy!

If you're new to Gammapy, go to :ref:`getting-started` and :ref:`tutorials` to
start learning how to use it. To learn more about installing Gammapy with conda,
see :ref:`install-conda` and :ref:`install-dependencies`.

Note that there are other ways to install Gammapy, either with ``conda``, or
with ``pip`` or other package managers (see :ref:`install-other`). Experts and
developers can also install the latest non-stable development version of Gammapy
(see :ref:`install-check`).

The following pages contain detailed installation information:

.. toctree::
    :maxdepth: 1

    dependencies
    conda
    check
    pip
    other
