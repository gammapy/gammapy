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

A good reference for the forward-folding on-off likelihood fitting methods
is Section 7.5 "Spectra and Light Curves" in [Naurois2012]_,
in publications usually the reference [Piron2001]_ is used.
A standard reference for the unfolding method is [Albert2007]_.


Spectral Fitting
================

.. _spectrum_command_line_tool:

Command line tool
-----------------

Spectral fitting within Gammapy is most easily performed with the ``gammapy-spectrum`` command line tool.
The example below shows how to use ``gammapy-spectrum`` by specifying analysis
options in a YAML config file. It assumes you have the `gammapy-extra`_
repository available.


.. include:: ./analysis_example.yaml
    :code: yaml


Copy the above config file to your machine, to e.g. ``crab_config.yaml`` and run

.. code-block:: bash

   gammapy-spectrum crab_config.yaml


Underlying classes
------------------

The spectral fitting procedure is a two step process. Each of the two steps is represented by one class.

* The `~gammapy.spectrum.SpectrumAnalysis` class converts the IRFs from the 2D format
  proposed for CTA (see :ref:`gadf:iact-irfs`) into the OGIP format needed for 1D analysis.
  It furthermore creates a source counts vector and a background counts vectors from the event list (see :ref:`gadf:iact-events`).
* The `~gammapy.spectrum.SpectralFit` class calls Sherpa in order to fit a
  model to the data

Creating OGIP data
^^^^^^^^^^^^^^^^^^

The following examples creates the 4 OGIP files that are needed for a spectral analysis

* `PHA`_ file
* `ARF`_ file
* `RMF`_ file
* BKG file (PHA format)

.. literalinclude:: run_spectrum_analysis.py
    :language: python
    :linenos:


In Detail:

* Line 14-16 : Define signal extraction region (ON region) using a `~gammapy.region.SkyCircleRegion`
* Line 18    : Define background methods (see :ref:`spectrum_background_method`)
* Line 20-22 : Read exclusion mask from FITS file
* Line 24    : Define reconstructed energy binning of the analysis
* Line 26-28 : Select `~gammapy.data.DataStore` and observations to be used
* Line 30    : Instantiate `~gammapy.spectrum.SpectrumAnalysis`
* Line 33    : Write OGIP data to disk

At this point one could in principle perform a fit with spectra fitting tools
like XSPEC or Sherpa. Also, note that writing the OGIP files to disk is only one
option, the `~gammapy.spectrum.SpectrumAnalysis` class also has the
functionality to process e.g. counts vectors and IRFs in memory.

Running a Sherpa fit
^^^^^^^^^^^^^^^^^^^^

To avoid having to deal with Sherpa directly or for scripting purposes the `~gammapy.spectrum.SpectralFit`
class can be used to perform a Fit as shown in the example below. It uses the PHA
files created in the example above, so feel free to use your own files instead of
using the ones in `gammapy-extra`_.

.. literalinclude:: run_spectrum_fit.py
    :language: python
    :linenos:

In Detail:

* Line 9-10  : Define input data
* Line 12    : Instantiate `~gammapy.spectrum.SpectralFit`
* Line 13    : Set model, note that you can pass any Sherpa model
* Line 14-15 : Define fit range
* Line 16   : Run Sherpa fit, other option: method = 'hspec'

.. _spectrum_background_method:

Background estimation methods
=============================

Currently supported background methods

* :ref:`region_reflected`
* Ring (not taking into account excluded regions)

The following example shows how the background estimation method is defined
in the YAML config file

.. include:: off_methods.yaml
    :code: yaml

Reference/API
=============

.. automodapi:: gammapy.spectrum
    :no-inheritance-diagram:

