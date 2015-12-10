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
The example below shows how to use ``gammapy-spectrum`` by specifying analysis
options in a YAML config file. It assumes you have the `gammapy-extra <https://github.com/gammapy/gammapy-extra>`__
repository available.


.. include:: ./analysis_example.yaml
    :code: yaml


Copy the above config file to your machine, to e.g. ``crab_config.yaml`` and run

.. code-block:: bash

   gammapy-spectrum crab_config.yaml


Underlying classes
------------------

The spectral fitting procedure is a two step process. Each of the two steps is represented by one class.

* The `~gammapy.spectrum.SpectrumAnalysis` class converts the data from the
  Fermi-LAT format (add link) into the OGIP format
* The `~gammapy.spectrum.SpectralFit` class calls Sherpa in order to fit a
  model to the data

Creating OGIP data
^^^^^^^^^^^^^^^^^^

The following examples creates the 4 OGIP files that are needed for a spectral analysis

* `PHA file <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/node5.html>`__
* `ARF file <http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc4>`__
* `RMF file <http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html#tth_sEc3.1>`__
* BKG file (PHA format)

.. literalinclude:: run_spectrum_analysis.py
    :language: python
    :linenos:


In Detail:

Line 10-12 : Define signal extraction region (ON region) using a `~gammapy.region.SkyCircleRegion`
Line 14    : Define background methods (see :ref:`spectrum_background_method`)
Line 16-18 : Read exclusion mask from FITS file
Line 20    : Define reconstructed energy binning of the analysis
Line 22-24 : Select `~gammapy.obs.DataStore` and observations to be used
Line 26    : Instantiate `~gammapy.spectrum.SpectrumAnalysis`
Line 29    : Write OGIP data to disk

At this point one could in principle perform a fit using the standard `Sherpa session interface <http://cxc.harvard.edu/sherpa/threads/pha_intro/>`__.
Note that writing the OGIP files to disk is only one option, the `~gammapy.spectrum.SpectrumAnalysis`
class also has options to process e.g. counts vectors and IRFs in memory.


Running a Sherpa fit
^^^^^^^^^^^^^^^^^^^^

To avoid having to deal with Sherpa directly or for scripting purposes the `~gammapy.spectrum.SpectralFit`
 class can be used to perform a Fit as shown in the following example. It assumes you have created the OGIP
 data as shown above.


.. _spectrum_background_method:

Background estimation methods
=============================

At the moment two background estimation methods are supported

