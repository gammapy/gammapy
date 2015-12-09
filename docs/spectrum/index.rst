.. include:: ../references.txt

.. _spectrum:

*****************************************************
Spectrum estimation and modeling (`gammapy.spectrum`)
*****************************************************

.. currentmodule:: gammapy.spectrum

Introduction
============
`gammapy.spectrum` holds functions and classes related to 1D region based spectral analysis.
The basic of this type of analysis are explained in `this <https://github.com/gammapy/PyGamma15/tree/gh-pages/talks/analysis-classical>`__ talk

TODO: explain basics

A good reference for the forward-folding on-off likelihood fitting methods is Section 7.5 "Spectra and Light Curves" in [Naurois2012]_,
in publications usually the reference [Piron2001]_ is used.
A standard reference for the unfolding method is [Albert2007]_.


Spectral Fitting
================

.. _spectrum_command_line_tool:

Command line tool
-----------------

Spectral fitting within Gammapy is most easily performed with the ``gammapy-spectrum`` command line tool.

The spectral fitting command-line tool makes use of the data management functionality in Gammapy. In order to download an example dataset from the `gammapy-extra <https://github.com/gammapy/gammapy-extra>`__ repository and set up an example `gammapy.obs.DataManager` please follow the instructions in :ref:`obs_dm`. The following step assume you have this example data set. If you already have a data set, please modify the steps below accordingly.

This is an example config file (YAML format) to be used with the ``gammapy-spectrum`` command line tool.

.. include:: ./analysis_example.yaml
    :code: yaml


Copy it to for example ``crab_config.yaml`` and run

.. code-block:: bash

   gammapy-spectrum crab_config.yaml


Underlying classes
------------------

The spectral fitting procedure is a two step process. Each of the two steps is represented by one class.

 * The `~gammapy.spectrum.SpectrumAnalysis` class converts the data from the Fermi-LAT format
   (add link) into the OGIP format

 * The `~gammapy.spectrum.SpectralFit` class calls Sherpa in order to fit a model to the data

Creating OGIP data
++++++++++++++++++

The following examples creates the 4 OGIP files that are needed for a spectral analysis
* `PHA file <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node5.html>`__
* `ARF file <http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc4>`__
* `RMF file <http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.1>`__
* BKG file (PHA format)

.. code-block:: python

   run_spectrum_analysis.py


Running a Sherpa fit
++++++++++++++++++++



Reference/API
=============

.. automodapi:: gammapy.spectrum.spectrum_analysis
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.counts_spectrum
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.crab
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.fitting_utils
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.models
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.powerlaw
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.sherpa_chi2asym
    :no-inheritance-diagram:
    
