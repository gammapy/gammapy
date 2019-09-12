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
in a configuration file in YAML format. hence, it also provides you with different
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

You can have a look at the configuration settings provided by default, and also dump
them into a file that you can edit to start a new analysis from the modified config file.

.. code-block:: python

    >>> print(analysis.config)
    >>> analysis.config.dump("myconfig.yaml")
    >>> analysis = Analysis.from_file("config.yaml")

You could also have started the analysis with your custom settings declared in a Python
nested dictionary which will overwrite the values provided by default. Note how the nested
dictionary must follow the parameters hierarchical structure which may be prone to errors.

.. code-block:: python

    >>> config_dict = {"general": {"logging": {"level": "WARNING"}}}
    >>> analysis = Analysis(config=config_dict)

At any moment you can change the value of one specific parameter needed in the analysis. Note
that it is a good practice to validate your settings when you modify the value of parameters.

.. code-block:: python

    >>> analysis.settings["reduction"]["background"]["on_region"]["frame"] = "galactic"
    >>> analysis.config.validate()

The hierarchical structure of the tens of parameters needed may be hard to follow. You can
print at any moment a *how-to* documentation with example values for all the sections and
parameters or only for one specific section or group of parmeters.

.. code-block:: python

    >>> analysis.config.print_help()
    >>> analysis.config.print_help("flux")

General settings
================




Observations selection
======================
The observations used in the analysis may be selected from a `Datastore` declared in the
settings, as well as using different values to build a filter.



Data reduction
==============



Fitting
=======



Flux points
===========








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