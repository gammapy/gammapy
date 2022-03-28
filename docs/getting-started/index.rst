.. include:: ../references.txt

.. _getting-started:

Getting started
===============

.. toctree::
    :hidden:

    install
    usage
    troubleshooting

Installation
------------

.. panels::
    :column: col-lg-6 col-md-6 col-sm-12 col-xs-12 p-3

    Working with conda?
    ^^^^^^^^^^^^^^^^^^^

    Gammapy is part of the `Anaconda <https://docs.continuum.io/anaconda/>`__
    distribution and can be installed with Anaconda or Miniconda:

    .. code-block:: bash

        conda install gammapy

    ---

    Prefer pip?
    ^^^^^^^^^^^

    Gammapy can be installed via pip from `PyPI <https://pypi.org/project/gammapy/>`__.


    .. code-block:: bash

        pip install gammapy

    ---
    :column: col-12 p-3

    In-depth instructions?
    ^^^^^^^^^^^^^^^^^^^^^^

    Working with virtual environments? Installing a specific version? Installing from source?  Check the advanced
    installation page.

    .. link-button:: install
        :type: ref
        :text: Learn more
        :classes: btn-secondary stretched-link


.. _download-tutorials:

Download Tutorial Datasets
--------------------------

You can now proceed to download the Gammapy tutorial notebooks and the example
datasets. The total size to download is ~180 MB. Select the location where you
want to install the datasets and proceed with the following commands:

.. code-block:: bash

    gammapy download notebooks --release 0.19
    gammapy download datasets
    export GAMMAPY_DATA=$PWD/gammapy-datasets

You might want to put the definition of the ``$GAMMAPY_DATA`` environment
variable in your shell profile setup file that is executed when you open a new
terminal (for example ``$HOME/.bash_profile``).

.. note::

    If you are not using the ``bash`` shell, handling of shell environment variables
    might be different, e.g. in some shells the command to use is ``set`` or something
    else instead of ``export``, and also the profile setup file will be different.
    On Windows, you should set the ``GAMMAPY_DATA`` environment variable in the
    "Environment Variables" settings dialog, as explained e.g.
    `here <https://docs.python.org/3/using/windows.html#excursus-setting-environment-variables>`__

The datasets are curated and stable, the notebooks are still under development
just like Gammapy itself, and thus stored in a sub-folder that contains the
Gammapy version number.

What next?
==========

Congratulations! You are all set to start using Gammapy!

* To learn how to use Gammapy, go to :ref:`tutorials`.
* If you're new to conda, Python and Jupyter, read the :ref:`using-gammapy` guide.
* If you encountered any issues you can check the :ref:`troubleshoot` guide.
