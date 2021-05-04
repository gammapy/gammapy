.. include:: ../references.txt

.. _quickstart:


Quickstart
==========

Installation using Anaconda
---------------------------

The easiest and recommended way to install Gammapy is to install the Anaconda
distribution from https://www.anaconda.com/download/ and then to install
Gammapy and its dependencies by executing this command in a terminal:

.. code-block:: bash

    conda install -c conda-forge gammapy

To check your Gammapy installation, you can use this command:

.. code-block:: bash

    gammapy info


Download tutorials
------------------

You can now proceed to download the Gammapy tutorial notebooks and the example
datasets used there (at the moment from `CTA`_, `H.E.S.S.`_. and `Fermi-LAT`_).
The total size to download is ~180 MB. Select the location where you want
to install the datasets and proceed with the following commands:

.. code-block:: bash

    gammapy download notebooks --release 0.18.2
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
