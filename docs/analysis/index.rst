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

You can start with the built-in default analysis configuration and update it by
passing values for just the parameters you want to set, using the
``AnalysisConfig.from_yaml`` method:

.. code-block:: python

    config = AnalysisConfig.from_yaml("""
    general:
        log:
            level: warning
    """)

The hierarchical structure of the tens of parameters needed may be hard to follow. You can
print as a *how-to* documentation a helping sample config file with example values for all
the sections and parameters or only for one specific section or group of parameters.

.. code-block:: python

    >>> print(config)

At any moment you can change the value of one specific parameter needed in the analysis.

.. code-block:: python

    >>> config.datasets.geom.wcs.skydir.frame = "galactic"

It is also possible to add new configuration parameters and values or overwrite the ones already
defined in your session analysis. In this case you may use the `analysis.update_config()` method
using a custom nested dictionary or custom YAML file (i.e. re-use a config file for specific
sections and/or from a previous analysis).:

.. code-block:: python

    >>> config_dict = {"data": {"datastore": "$GAMMAPY_DATA/hess-dl3-dr1"}}
    >>> analysis.update_config(config_dict)
    >>> analysis.update_config(filename="fit.yaml")

In the following you may find more detailed information on the different sections which
compose the YAML formatted nested configuration settings hierarchy.

General settings
----------------

The ``general`` section comprises information related with the ``log`` configuration,
as well as the output folder where all file outputs and datasets will be stored, declared
as value of the ``outdir`` parameter.

.. gp-howto-hli:: general

Observations selection
----------------------

The observations used in the analysis may be selected from a ``datastore`` declared in the
``data`` section of the settings, using also different parameters and values to
create a composed filter.

.. gp-howto-hli:: observations

You may use the `get_observations()` method to proceed to make the observation filtering.
The observations are stored as a list of `~gammapy.data.DataStoreObservation` objects.

.. code-block:: python

    >>> analysis.get_observations()
    >>> list(analysis.observations)
    [<gammapy.data.observations.DataStoreObservation at 0x11e040320>,
     <gammapy.data.observations.DataStoreObservation at 0x11153d550>,
     <gammapy.data.observations.DataStoreObservation at 0x110a84160>,
     <gammapy.data.observations.DataStoreObservation at 0x110a84b38>]

Data reduction and datasets
---------------------------

The data reduction process needs a choice of a dataset type, declared as ``1d`` or ``3d``
in the ``datasets`` section of the settings. For the estimation of the background in a ``1d``
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
    >>> analysis.datasets.info_table()

Model
-----

For now we simply declare the model as a reference to a separate YAML file, passing
the filename into the `~gammapy.analysis.Analysis.read_model` method to fetch the model and attach it to your
datasets.

.. code-block:: python

    >>> analysis.read_model("model.yaml")

If you have a `~gammapy.modeling.models.SkyModels` object, or a YAML string representing
one, you can use the `~gammapy.analysis.Analysis.set_models` method:

.. code-block:: python

    >>> models = SkyModels(...)
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

    >>> analysis.get_flux_points()
    INFO:gammapy.analysis.analysis:Calculating flux points.
    INFO:gammapy.analysis.analysis:
          e_ref               ref_flux                 dnde                 dnde_ul                dnde_err        is_ul
           TeV              1 / (cm2 s)          1 / (cm2 s TeV)        1 / (cm2 s TeV)        1 / (cm2 s TeV)
    ------------------ ---------------------- ---------------------- ---------------------- ---------------------- -----
    1.1364636663857248   5.82540193791155e-12 1.6945571729283257e-11 2.0092001005968464e-11  1.491004091925887e-12 False
    1.3768571648527583 2.0986802770569557e-12 1.1137098968561381e-11 1.4371773951168255e-11  1.483696107656724e-12 False
    1.6681005372000581 3.0592927032553813e-12  8.330762241576842e-12   9.97704078861513e-12  7.761855010963746e-13 False
    2.1544346900318834  1.991366151205521e-12  3.749504881244244e-12  4.655825384923802e-12  4.218641798406146e-13 False
    2.6101572156825363  7.174167397335237e-13 2.3532638339895766e-12 3.2547227459669707e-12   4.05804720903438e-13 False
    3.1622776601683777 1.0457942646403696e-12 1.5707172671966065e-12 2.0110274930777325e-12 2.0291499028818014e-13 False
     3.831186849557287 3.7676160725948056e-13  6.988070884720634e-13 1.0900735920193252e-12 1.6898704308171627e-13 False
    4.6415888336127775  5.492137361542478e-13 4.2471136559991427e-13  6.095655421226728e-13  8.225678668637978e-14 False
     5.994842503189405 3.5749624179174077e-13 2.2261366353081893e-13  3.350617464903039e-13  4.898878805758816e-14 False
      7.26291750173621 1.2879288326657447e-13 2.5317668601400673e-13 4.0803852787540073e-13  6.601201499048379e-14 False
      8.79922543569107  1.877442373267013e-13  7.097738087032472e-14  1.254638299336029e-13 2.2705519890120373e-14 False
    >>> analysis.flux_points.peek()

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