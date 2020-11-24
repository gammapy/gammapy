.. include:: ../references.txt

.. _install:

************
Installation
************


Using Anaconda
--------------

The recommended way to install Gammapy is to install the Anaconda
distribution from https://www.anaconda.com/download/ and then to install the
Gammapy and it's dependencies by executing these commands in a terminal:

.. code-block:: bash

    curl -O https://gammapy.org/download/install/gammapy-0.18.2-environment.yml
    conda env create -f gammapy-0.18.2-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.

This creates a conda environment called ``gammapy-0.18.2`` which you can activate via:

.. code-block:: bash

    conda activate gammapy-0.18.2

Note that you have to execute that activation command (but not the environment
creation command) every time you open a new terminal window, because the default
with conda is the base environment, which doesn't have Gammapy installed.

To check your Gammapy installation, you can use this command:

.. code-block:: bash

    gammapy info


Using other package managers
----------------------------

If you don't want to use Anaconda, you can use other package managers. To do so,the following pages contain detailed
information about Gammapy dependencies and propose various installation options:

* See gammapy install dependencies :ref:`install-dependencies`
* If you want to install gammapy with pip see :ref:`install-pip`
* If you want to use other package manager see :ref:`install-other`


Download tutorials
------------------

You can now proceed to download the Gammapy tutorial notebooks and the example
datasets used there (at the moment from CTA, H.E.S.S. and Fermi-LAT). The total
size to download is about 180 MB. Select the location where you want to install
the datasets and proceed with the following commands:

.. code-block:: bash

    gammapy download tutorials --release 0.18.2
    cd gammapy-tutorials
    export GAMMAPY_DATA=$PWD/datasets

You might want to put the definition of the ``$GAMMAPY_DATA`` environment
variable in your shell profile setup file that is executed when you open a new
terminal (for example ``$HOME/.bash_profile``).

If you are not using the ``bash`` shell, handling of shell environment variables
might be different, e.g. in some shells the command to use is ``set`` or something
else instead of ``export``, and also the profile setup file will be different.

On Windows, you should set the ``GAMMAPY_DATA`` environment variable in the
"Environment Variables" settings dialog, as explained e.g.
`here <https://docs.python.org/3/using/windows.html#excursus-setting-environment-variables>`__

The datasets are curated and stable, the notebooks are still under development
just like Gammapy itself, and thus stored in a sub-folder that contains the
Gammapy version number.

If there are issues, note that you can just delete the folder any time using ``rm
-r gammapy-tutorials`` and start over.

What next?
----------

Congratulations! You are all set to start using Gammapy!

* If you're new to conda, Python, ipython and Jupyter, read the :ref:`getting-started` guide.
* To learn how to use Gammapy, go to :ref:`tutorials`.

.. Include toc hidden to avoid warnings in doc building

.. toctree::
    :hidden:

    dependencies
    other
    pip
