.. include:: ../references.txt

.. _quickstart:


Quickstart
==========

Installation using Anaconda
---------------------------

The easiest way to install Gammapy is to install the Anaconda
distribution from https://www.anaconda.com/download/ and then to install
Gammapy and its dependencies by executing this command in a terminal:

.. code-block:: bash

    conda install -c conda-forge gammapy

Though this is one line command is the standard way to install a software package using Anaconda, **we recommend to
make use of an environment definition file** that we provide, so you can get additional useful packages together with
Gammapy in a virtual isolated environment. If you want to learn about using virtual environments see
:ref:`virtual-envs`. In order to proceed in this way, just copy and paste in your terminal the two lines below:

.. code-block:: bash

    curl -O https://gammapy.org/download/install/gammapy-0.19-environment.yml
    conda env create -f gammapy-0.19-environment.yml

.. note::

    On Windows, you have to open up the conda environment file and delete the
    lines with ``sherpa`` and ``healpy``. Those are optional dependencies that
    currently aren't available on Windows.


This creates a conda environment called ``gammapy-0.19`` which you can activate via:

.. code-block:: bash

    conda activate gammapy-0.19

Note that you have to execute that activation command (but not the environment
creation command) every time you open a new terminal window, because the default
with conda is the base environment, which might not have Gammapy installed.

To check your Gammapy installation, you can use this command:

.. code-block:: bash

    gammapy info

To leave the environment, you may activate another one or just type:

.. code-block:: bash

    conda deactivate

.. _download-tutorials:

Download tutorials
------------------

You can now proceed to download the Gammapy tutorial notebooks and the example
datasets used there (at the moment from `CTA`_, `H.E.S.S.`_. and `Fermi-LAT`_).
The total size to download is ~180 MB. Select the location where you want
to install the datasets and proceed with the following commands:

.. code-block:: bash

    gammapy download notebooks --release 0.19
    gammapy download datasets
    export GAMMAPY_DATA=$PWD/gammapy-datasets

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

* If you're new to conda, Python, ipython and Jupyter, read the :ref:`using-gammapy` guide.
* To learn how to use Gammapy, go to :ref:`tutorials`.
