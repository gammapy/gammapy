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

    Gammapy can be installed with `Anaconda <https://docs.continuum.io/anaconda/>`__ or Miniconda:

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


Introduction
------------

.. accordion-header::
    :id: collapseTwo
    :title: How to access gamma-ray data
    :link: ../tutorials/data/hess.html#DL3-DR1

To access IACT data in the DL3 format, use the `~gammapy.data.DataStore`. It allows
easy access to observations stored in the DL3 data library.
It is also internally used by the high level interface `~gammapy.analysis.Analysis`.

.. accordion-footer::

.. accordion-header::
    :id: collapseSeven
    :title: How to compute a 1D spectrum
    :link: ../tutorials/analysis/1D/spectral_analysis.html

The `~gammapy.analysis.Analysis` class can perform spectral extraction. The
`~gammapy.analysis.AnalysisConfig` must be defined to produce '1d' datasets.
Alternatively, you can follow the spectral analysis tutorial.

.. accordion-footer::

.. accordion-header::
    :id: collapseSix
    :title: How to compute a 2D image
    :link: ../tutorials/index.html#d-image

Gammapy treats 2D maps as 3D cubes with one bin in energy. Sometimes, you might want to use previously
obtained images lacking an energy axis (eg: reduced using traditional IACT tools) for modeling and fitting
inside Gammapy. In this case, it is necessary to attach an `energy` axis on as it is showm in the tutorials.

.. accordion-footer::


.. accordion-header::
    :id: collapseEight
    :title: How to compute a lightcurve
    :link: ../tutorials/analysis/time/light_curve.html

The light curve estimation tutorial shows how to extract a run-wise lightcurve.

To perform an analysis in a time range smaller than that of an observation, it
is necessary to filter the latter with its `~gammapy.data.Observations.select_time` method. This produces
an new observation containing events in the specified time range. With the new
`~gammapy.data.Observations` it is then possible to perform the usual data
reduction which will produce datasets in the correct time range. The light curve
extraction can then be performed as usual with the
`~gammapy.estimators.LightCurveEstimator`. This is demonstrated in the light curve flare tutorial.

.. accordion-footer::

.. accordion-header::
    :id: collapseTwelve
    :title: How to detect sources in an image
    :link: ../tutorials/analysis/2D/detect.html

Gammapy provides methods to perform source detection in a 2D map. First step is
to produce a significance map, i.e. a map giving the probability that the flux
measured at each position is a background fluctuation. For a
`~gammapy.datasets.MapDataset`, the class `~gammapy.estimators.TSMapEstimator` can be
used. A simple correlated Li & Ma significance can be used, in particular for
ON-OFF datasets. The second step consists in applying a peak finer algorithm,
such as `~gammapy.estimators.utils.find_peaks`.

.. accordion-footer::

.. accordion-header::
    :id: collapseTwenty
    :title: How to combine data from multiple instruments
    :link: ../tutorials/analysis/3D/analysis_mwl.html

Gammapy offers the possibility to combine data from multiple instruments
in a "joint-likelihood" fit.

.. accordion-footer::


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


What next?
==========

Congratulations! You are all set to start using Gammapy!

* To learn how to use Gammapy, go to :ref:`tutorials`.
* If you're new to conda, Python and Jupyter, read the :ref:`using-gammapy` guide.
* If you encountered any issues you can check the :ref:`troubleshoot` guide.
