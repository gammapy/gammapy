.. include:: ../references.txt

.. _spectral_fitting:

****************
Spectral Fitting
****************

.. currentmodule:: gammapy.spectrum


TODO: explain class interface


Background estimation methods
=============================

Currently supported background methods

* :ref:`region_reflected`
* Ring (not taking into account excluded regions)

The following example shows how the background estimation method is defined
in the YAML config file

.. include:: off_methods.yaml
    :code: yaml
