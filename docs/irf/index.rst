.. _irf:

***********************************
irf - Instrument response functions
***********************************

.. currentmodule:: gammapy.irf

Introduction
============
For a definition of the response function you are invited to read 
:ref:`irf-theory`.

`gammapy.irf` handles the following instrument response functions (IRFs):

* :ref:`irf-aeff` (AEFF)
* :ref:`irf-edisp` (EDISP)
* :ref:`irf-psf` (PSF)
* :ref:`irf-bkg` (BKG)

Most of the formats defined at :ref:`gadf:iact-irf` are supported.    
At the moment, there is little support for Fermi-LAT or other instruments.

Most users will not use `gammapy.irf` directly, but will instead use IRFs as
part of their spectrum, image or cube analysis to compute exposure and effective
EDISP and PSF for a given dataset.

Most (at some point maybe all) classes in `gammapy.irf` have an
`gammapy.utils.nddata.NDDataArray` as data attribute to support interpolation.


IRF Axis naming
---------------
In the IRF classes we use the following axis naming convention:

================= ============================================================================
Variable          Definition
================= ============================================================================
``energy``        Reconstructed energy axis (:math:`E` in :ref:`irf-theory`)
``energy_true``   True energy axis (:math:`E_{\rm true}` in :ref:`irf-theory`)
``offset``        Field of view offset from center (:math:`p_{\rm true}` in :ref:`irf-theory`)
``fov_lon``       Field of view	longitude
``fov_lat``       Field of view latitude
``migra``         Energy migration (:math:`\mu` in :ref:`irf-theory`)
``rad``        	  Offset angle from source position (:math:`\delta p` in :ref:`irf-theory`)
================= ============================================================================

Getting Started
===============

See `cta.html <../tutorials/cta.html>`__ for an example how to access IACT IRFs.

Using `gammapy.irf`
===================

If you'd like to learn more about using `gammapy.irf`, read the following 
sub-pages:

.. toctree::
    :maxdepth: 1

    theory
    aeff
    edisp
    psf
    bkg

Reference/API
=============

.. automodapi:: gammapy.irf
    :no-inheritance-diagram:
    :include-all-objects:
