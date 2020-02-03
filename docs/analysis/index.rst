.. include:: ../references.txt

.. _analysis:

*******************************
analysis - High-level interface
*******************************

.. currentmodule:: gammapy.analysis

.. _analysis_intro:

Introduction
============

The high-level interface for Gammapy provides a high-level Python API for the
most common use cases identified in the analysis process. The classes and
methods included may be used in Python scripts, notebooks or as commands within
IPython sessions. The high-level user interface could also be used to automatise
processes driven by parameters declared in a configuration file in YAML format
that addresses the most common analysis use cases identified.

.. _analysis_start:

Getting started
===============

The easiest way to get started with the high-level interface is using it within
an IPython console or a notebook.

.. code-block:: python

    >>> from gammapy.analysis import Analysis, AnalysisConfig
    >>> config = AnalysisConfig()
    >>> analysis = Analysis(config)

Configuration and methods
=========================

You can have a look at the configuration settings provided by default, and also dump
them into a file that you can edit to start a new analysis from the modified config file.

.. code-block:: python

    >>> print(config)
    >>> config.write("config.yaml")
    >>> config = AnalysisConfig.read("config.yaml")

You can also start with the built-in default analysis configuration and update it by
passing values for just the parameters you want to set, using the
``AnalysisConfig.from_yaml`` method:

.. code-block:: python

    config = AnalysisConfig.from_yaml("""
    general:
        log:
            level: warning
    """)

Once you have your configuration defined you may start an `analysis` instance:

.. code-block:: python

    analysis = Analysis(config)

The hierarchical structure of the tens of parameters needed may be hard to follow. You can
print your analysis config as a mean to display its format and syntax, the parameters
and units allowed, as well as the different sections where they belong in the config structure.

.. code-block:: python

    >>> print(analysis.config)

At any moment you may add or change the value of one specific parameter needed in your analysis.

.. code-block:: python

    >>> analysis.config.datasets.geom.wcs.skydir.frame = "galactic"

General settings
----------------

In the following you may find more detailed information on the different sections which
compose the YAML formatted nested configuration settings hierarchy. The different
high-level analysis commands exposed may be reproduced within the
`First analysis <../notebooks/analysis_1.html>`__ tutorial.

The ``general`` section comprises information related with the ``log`` configuration,
as well as the output folder where all file outputs and datasets will be stored, declared
as value of the ``outdir`` parameter.

.. gp-howto-hli:: general

Observations selection
----------------------

The observations used in the analysis may be selected from a ``datastore`` declared in the
``observations`` section of the settings, using also different parameters and values to
create a composed filter.

.. gp-howto-hli:: observations

You may use the `get_observations()` method to proceed to make the observation filtering.
The observations are stored as a list of `~gammapy.data.Observation` objects.

.. code-block:: python

    >>> analysis.get_observations()
    >>> analysis.observations.ids
    ['23592', '23523', '23526', '23559']

Data reduction and datasets
---------------------------

The data reduction process needs a choice of a dataset type, declared as ``1d`` or ``3d``
in the ``type`` parameter of ``datasets`` section of the settings. For the estimation of the background in a ``1d``
use case, a background ``method`` is needed, other parameters related like the ``on_region``
and ``exclusion`` FITS file may be also present. Parameters for geometry are also needed and
declared in this section, as well as a boolean flag ``stack``.

.. gp-howto-hli:: datasets

You may use the `get_datasets()` method to proceed to the data reduction process.
The final reduced datasets are stored in the ``datasets`` attribute.
For spectrum datasets reduction the information related with the background estimation is
stored in the ``background`` property.

.. code-block:: python

    >>> analysis.get_datasets()
    >>> print(analysis.datasets)

Model
-----

For now we simply declare the model as a reference to a separate YAML file, passing
the filename into the `~gammapy.analysis.Analysis.read_model` method to fetch the model and attach it to your
datasets.

.. code-block:: python

    >>> analysis.read_models("model.yaml")

If you have a `~gammapy.modeling.models.Models` object, or a YAML string representing
one, you can use the `~gammapy.analysis.Analysis.set_models` method:

.. code-block:: python

    >>> models = Models(...)
    >>> analysis.set_models(models)

Fitting
-------

The parameters used in the fitting process are declared in the ``fit`` section.

.. gp-howto-hli:: fit

You may use the `run_fit()` method to proceed to the model fitting process. The result
is stored in the ``fit_result`` property.

.. code-block:: python

    >>> analysis.run_fit()

Flux points
-----------

For spectral analysis where we aim to calculate flux points in a range of energies, we
may declare the parameters needed in the ``flux_points`` section.

.. gp-howto-hli:: flux_points

You may use the `get_flux_points()` method to calculate the flux points. The result
is stored in the ``flux_points`` property as a `~gammapy.spectrum.FluxPoints` object.

.. code-block:: python

    >>> analysis.config.flux_points.source="crab"
    >>> analysis.get_flux_points()
    INFO:gammapy.analysis.analysis:Calculating flux points.
    INFO:gammapy.analysis.analysis:
          e_ref               ref_flux        ...        dnde_err        is_ul
           TeV              1 / (cm2 s)       ...    1 / (cm2 s TeV)
    ------------------ ---------------------- ... ---------------------- -----
    1.4125375446227544  1.928877387452331e-11 ... 1.2505519776748809e-12 False
    3.1622776601683795  7.426613493860134e-12 ...  2.106743519478604e-13 False
      7.07945784384138 1.4907957189689605e-12 ...   4.74857915062012e-14 False
    >>> analysis.flux_points.peek()

You may set fine-grained optional parameters for the `~gammapy.spectrum.FluxPointsEstimator` in the
``flux_points.params`` settings.

.. code-block:: python

    >>>  analysis.config.flux_points.params["reoptimize"]=True


Residuals
---------

For 3D analysis we can compute a residual image to check how good are the models
for the source and/or the background.

.. code-block:: python

    >>> analysis.datasets[0].plot_residuals()

Using the high-level interface
------------------------------

Gammapy tutorial notebooks that show examples using the high-level interface:

* `First analysis <../notebooks/analysis_1.html>`__

Reference/API
=============

.. automodapi:: gammapy.analysis
    :no-inheritance-diagram:
    :include-all-objects: