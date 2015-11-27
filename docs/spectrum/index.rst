.. include:: ../references.txt

.. _spectrum:

*****************************************************
Spectrum estimation and modeling (`gammapy.spectrum`)
*****************************************************

.. currentmodule:: gammapy.spectrum

Introduction
============

`gammapy.spectrum` holds functions and classes to fit spectral models and compute flux points.

Physical radiative models (synchrotron, inverse Compton and pion-decay emission)
for arbitrary cosmic ray particle spectra are available in the `naima`_ package.

Explain spectrum estimation basics.

Define vocabulary.

A good reference for the forward-folding on-off likelihood fitting methods is Section 7.5 "Spectra and Light Curves" in [Naurois2012]_,
in publications usually the reference [Piron2001]_ is used.
A standard reference for the unfolding method is [Albert2007]_.

.. _spectrum_getting_started:

Getting Started
===============

Spectral fitting within Gammapy is most easily performed with the ``gammapy-spectrum`` command line tool. 

The spectral fitting command-line tool makes use of the data management functionality in Gammapy. In order to download an example dataset from the `gammapy-extra <https://github.com/gammapy/gammapy-extra>`__ repository and set up an example `gammapy.obs.DataManager` please follow the instructions in :ref:`obs_dm`. The following step assume you have this example data set. If you already have a data set, please modify the steps below accordingly.

This is an example config file (YAML format) to be used with the ``gammapy-spectrum`` command line tool.

.. include:: ./analysis_example.yaml
    :code: yaml


Copy it to for example ``crab_config.yaml`` and run

.. code-block:: bash

   gammapy-spectrum crab_config.yaml

.. _energy_handling_gammapy:

Energy handling in Gammapy
==========================

Basics
------

Most objects in Astronomy require an energy axis, e.g. counts spectra or
effective area tables. In general, this axis can be defined in two ways.

* As an array of energy values. E.g. the Fermi-LAT diffuse flux is given at
  certain energies and those are stored in an ENERGY FITS table extension.
  In Gammalib this is represented by GEnergy.
* As an array of energy bin edges. This is usually stored in EBOUNDS tables,
  e.g. for Fermi-LAT counts cubes. In Gammalib this is represented by GEbounds.

In Gammapy both the use cases are handled by two seperate classes: 
`gammapy.utils.energy.Energy` for energy values and
`gammapy.utils.energy.EnergyBounds` for energy bin edges

Energy
------

The Energy class is a subclass of `~astropy.units.Quantity` and thus has the
same functionality plus some convenienve functions for fits I/O

.. code-block:: python
    
    >>> from gammapy.utils.energy import Energy
    >>> energy = Energy([1,2,3], 'TeV')
    >>> hdu = energy.to_fits()
    >>> type(hdu) 
    <class 'astropy.io.fits.hdu.table.BinTableHDU'>

EnergyBounds
------------

The EnergyBounds class is a subclass of Energy. Additional functions are available
e.g. to compute the bin centers

.. code-block:: python
    
    >>> from gammapy.utils.energy import EnergyBounds
    >>> ebounds = EnergyBounds.equal_log_spacing(1, 10, 8, 'GeV')
    >>> ebounds.size
    9
    >>> ebounds.nbins
    8
    >>> center = ebounds.log_centers
    >>> center
    <Energy [ 1.15478198, 1.53992653, 2.05352503, 2.73841963, 3.65174127,
              4.86967525, 6.49381632, 8.65964323] GeV>

Reference/API
=============

.. automodapi:: gammapy.spectrum
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.fitting_utils
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.models
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.powerlaw
    :no-inheritance-diagram:

.. automodapi:: gammapy.spectrum.sherpa_chi2asym
    :no-inheritance-diagram:
    
