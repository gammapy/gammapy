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

Getting Started
===============

TODO

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
`gammapy.spectrum.energy.Energy` for energy values and
`gammapy.spectrum.energy.EnergyBounds` for energy bin edges

Energy
------

The Energy class is a subclass of `~astropy.units.Quantity` and thus has the
same functionality plus some convenienve functions for fits I/O

.. code-block:: python
    
    >>> from gammapy.spectrum import Energy
    >>> energy = Energy([1,2,3], 'TeV')
    >>> hdu = energy.to_fits()
    >>> type(hdu) 
    <class 'astropy.io.fits.hdu.table.BinTableHDU'>

EnergyBounds
------

The EnergyBounds class is a subclass of Energy. Additional functions are available
e.g. to compute the bin centers

.. code-block:: python
    
    >>> from gammapy.spectrum import EnergyBounds
    >>> ebounds = EnergyBounds.equal_log_spacing(1, 10, 8, 'GeV')
    >>> ebounds.size
    9
    >>> ebounds.nbins
    8
    >>> center = ebounds.log_center
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
    
