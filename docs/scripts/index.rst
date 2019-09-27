.. include:: ../references.txt

.. _HLI:

******************************
scripts - High-level interface
******************************

.. currentmodule:: gammapy.scripts

.. _HLI_intro:

Introduction
============
The high-level interface for Gammapy follows the recommendations written in
:ref:`pig-012`. It provides a high-level Python API for the most common use cases
identified in the analysis process. The classes and methods included may be used in
Python scripts, notebooks or as commands within IPython sessions. The high-level user
interface could also be used to automatise processes driven by parameters declared
in a configuration file in YAML format. Hence, it also provides you with different
configuration templates to address the most common analysis use cases identified.

.. _HLI_start:

Getting started
===============
The easiest way to get started with the high-level interface is using it within
an IPython console or a notebook.

.. code-block:: python

    >>> from gammapy.scripts import Analysis
    >>> analysis = Analysis()
        INFO:gammapy.scripts.analysis:Setting logging config: {'level': 'INFO'}

Configuration and methods
=========================
You can have a look at the configuration settings provided by default, and also dump
them into a file that you can edit to start a new analysis from the modified config file.

.. code-block:: python

    >>> print(analysis.config)
    >>> analysis.config.to_yaml("config.yaml")
    INFO:gammapy.scripts.analysis:Configuration settings saved into config.yaml
    >>> analysis = Analysis.from_yaml("config.yaml")

You may choose to start an analysis using a predefined **settings template**. If no
value for the settings template is provided, the `basic` template will be used by default.
You may dump these settings into a file, edit the file and re-initialize your settings
from the modified file.

.. code-block:: python

    >>> analysis = Analysis.from_template("1d")
    >>> analysis.config.to_yaml("config.yaml")
    >>> analysis = Analysis.from_yaml("config.yaml")

You could also have started with a built-in analysis configuration and extend it with
with your custom settings declared in a Python nested dictionary. Note how the nested
dictionary must follow the hierarchical structure of the parameters. Declaring the
configuration settings of the analysis in this way may be tedious and prone to errors
if you have several parameters to set, so we suggest you to proceed using a configuration
file.

.. code-block:: python

    >>> config_dict = {"general": {"logging": {"level": "WARNING"}}}
    >>> analysis = Analysis("3d")
    >>> analysis.config.update_settings(config_dict)

The hierarchical structure of the tens of parameters needed may be hard to follow. You can
print at any moment a *how-to* documentation with example values for all the sections and
parameters or only for one specific section or group of parameters.

.. code-block:: python

    >>> analysis.config.print_help()
    >>> analysis.config.print_help("flux")

At any moment you can change the value of one specific parameter needed in the analysis. Note
that it is a good practice to validate your settings when you modify the value of parameters.

.. code-block:: python

    >>> analysis.settings["reduction"]["background"]["on_region"]["frame"] = "galactic"
    >>> analysis.config.validate()

It is also possible to add new configuration parameters and values or overwrite the ones already
defined in your session analysis. In this case you may use the `config.update_settings()` method
using a custom nested dictionary or custom YAML file (i.e. re-use a config file for specific
sections and/or from a previous analysis).:

.. code-block:: python

    >>> config_dict = {"observations": {"datastore": "$GAMMAPY_DATA/hess-dl3-dr1"}}
    >>> analysis.config.update_settings(config=config_dict)
    >>> analysis.config.update_settings(configfile="fit.yaml")

In the following you may find more detailed information on the different sections which
compose the YAML formatted nested configuration settings hierarchy.

General settings
----------------
The `general` section comprises information related with the `logging` configuration,
as well as the output folder where all file outputs and datasets will be stored, declared
as value of the `outdir` parameter.

.. gp-howto-hli:: general

Observations selection
----------------------
The observations used in the analysis may be selected from a `datastore` declared in the
`observations` section of the settings, using also different parameters and values to
build a composed filter.

.. gp-howto-hli:: observations

You may use the `get_observations()` method to proceed to make the observation filtering.
The observations are stored as a list of `DataStoreObservation` containers.

.. code-block:: python

    >>> analysis.get_observations()
    >>> analysis.observations.list
        [<gammapy.data.observations.DataStoreObservation at 0x11e040320>,
         <gammapy.data.observations.DataStoreObservation at 0x11153d550>,
         <gammapy.data.observations.DataStoreObservation at 0x110a84160>,
         <gammapy.data.observations.DataStoreObservation at 0x110a84b38>]

Data reduction and datasets
---------------------------
The data reduction process needs a choice of a dataset type, declared as the class name
(MapDataset, SpectrumDatasetOnOff) in the `reduction` section of the settings. For the
estimation of the background with a dataset type SpectrumDatasetOnOff, a `background_estimator`
is needed, other parameters related with the `on_region` and `exclusion_mask` FITS file
may be also present. Parameters for geometry are also needed and declared in this section,
as well as a boolean flag `stack-datasets`.

.. gp-howto-hli:: reduction

You may use the `get_datasets()` method to proceed to the data reduction process.
The final reduced datasets are stored in the `.datasets` attribute.
For spectral reduction the information related with the background estimation is
stored in the `background_estimator` property.

.. code-block:: python

    >>> analysis.get_datasets()
    >>> analysis.datasets.datasets
        [SpectrumDatasetOnOff,
         SpectrumDatasetOnOff,
         SpectrumDatasetOnOff,
         SpectrumDatasetOnOff,
         SpectrumDatasetOnOff,
         SpectrumDatasetOnOff,
         SpectrumDatasetOnOff,
         SpectrumDatasetOnOff]
    >>> analysis.background_estimator.on_region
        <CircleSkyRegion(<SkyCoord (ICRS): (ra, dec) in deg
            (83.633, 22.014)>, radius=0.1 deg)>

Model
-----
For now we simply declare the model as a reference to a separate yaml file.
You may use the `get_model()` method to fetch the model and attach it to your
datasets.

Fitting
-------
The parameters used in the fitting process are declared in the `fit` section.

.. gp-howto-hli:: fit

You may use the `run_fit()` method to proceed to the model fitting process. The result
is stored in the `fit_result` property.

.. code-block:: python

    >>> analysis.run_fit()
    >>> analysis.fit_result
        OptimizeResult

            backend    : minuit
            method     : minuit
            success    : True
            message    : Optimization terminated successfully.
            nfev       : 111
            total stat : 239.28

Flux points
-----------
For spectral analysis where we aim to calculate flux points in a range of energies, we
may declare the parameters needed in the `flux` section.

.. gp-howto-hli:: flux

You may use the `get_flux_points()` method to calculate the flux points. The result
is stored in the `flux_points_dataset` property as a `FluxPointsDataset` container.

.. code-block:: python

    >>> analysis.get_flux_points()
        INFO:gammapy.scripts.analysis:Calculating flux points.
        INFO:gammapy.scripts.analysis:
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
    >>> analysis.flux_points_dataset.peek()

Residuals
---------
For 3D analysis we can compute a residual image to check how good are the models
for the source and/or the background.

.. code-block:: python

    >>> analysis.datasets.datasets[0].residuals()
            geom  : WcsGeom
            axes  : ['lon', 'lat', 'energy']
            shape : (250, 250, 4)
            ndim  : 3
            unit  :
            dtype : float64

Using the high-level interface
------------------------------
Gammapy tutorial notebooks that show examples using the high-level interface:

* :gp-notebook:`hess`

Command line tools
==================

.. toctree::
    :maxdepth: 1

    cli

Reference/API
=============

.. automodapi:: gammapy.scripts
    :no-inheritance-diagram:
    :include-all-objects: