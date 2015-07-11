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

.. warning::

   This is  completely experimental

Basics
------

Most objects in Astronomy require an energy axis, e.g. counts spectra or effective area tables. In general, this axis can be defined in two ways.

* As an array of energy bin edges. This is usually stored in EBOUNDS tables, e.g. for Fermi-LAT counts cubes. In Gammalib this is represented by GEbounds.
* As an array of energy values. E.g. the Fermi-LAT diffuse flux is given at certain energies and those are stored in an ENERGY FITS table extension. In Gammalib this is represented by GEnergy.

In Gammapy both use cases are handled by `gammapy.spectrum.utils.EnergyBinning` (to be renamed!). By default, the first case, i.e. an array of energy bin edges is expected in the constructor. 

.. code-block:: python
    
    >>> from gammapy.spectrum import EnergyBinning
    >>> from astropy.units import Quantity
    >>> binning = Quantity(np.logspace(-1,1,11))
    >>> e_axis = EnergyBinning(e_axis, 'bin_edges')

Note, that the 'bin_edges' flag can be left out. Of course the second way of defining the axis, i.e. by the energy bin centers, is also supported. 

.. code-block:: python
    
    >>> from gammapy.spectrum import EnergyBinning
    >>> from astropy.units import Quantity
    >>> binning = Quantity(np.logspace(-1,1,11))
    >>> centers = np.sqrt(binning[:-1]*binning[1:])
    >>> e_axis = EnergyBinning(center, 'bin_centers')

The main advantage of the EnergyBinning class with respect to pure numpy arrays or two separate classes handling both use cases is that the transition from energy bin edges to bin centers has to be only done once in constructor. All other class do not need to deal with the energy binning conversions anymore.

FITS I/O
--------

TODO


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
    
