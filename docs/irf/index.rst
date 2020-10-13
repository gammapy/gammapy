.. _irf:

***********************************
irf - Instrument response functions
***********************************

.. currentmodule:: gammapy.irf

Introduction
============
For a definition of the response function you are invited to read :ref:`irf-theory`.

`gammapy.irf` handles the following instrument response functions (IRFs):

* Effective area (AEFF)
* Energy dispersion (EDISP)
* Point spread function (PSF)
* Template background (BKG)

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

================= ===================================
Variable          Definition
================= ===================================
``energy``        Reconstructed energy axis
``energy_true``   True energy axis
``offset``        Field of view offset from center
``fov_lon``       Field of view	longitude
``fov_lat``       Field of view latitude
``migra``         Energy migration
``rad``        	  Offset angle from source position
================= ===================================

Getting Started
===============

See `cta.html <../tutorials/cta.html>`__ for an example how to access IACT IRFs.

Effective area
==============

as a function of true energy and offset angle (:ref:`gadf:aeff_2d`)
-------------------------------------------------------------------
The `~gammapy.irf.EffectiveAreaTable2D` class represents an effective area as a function of true energy and offset angle 
(:math:`A_{\rm eff}(p, E)` following the notation in :ref:`irf-theory`). 
Its format specifications are available in :ref:`gadf:aeff_2d`.

This is the format in which IACT DL3 effective areas are usually provided, as an example

.. plot:: irf/plot_aeff.py
    :include-source:
    
as a function of true energy (:ref:`gadf:ogip-arf`)
---------------------------------------------------
`~gammapy.irf.EffectiveAreaTable` instead represents an effective area as a function of true energy only 
(:math:`A_{\rm eff}(E)` following the notation in :ref:`irf-theory`).
Its format specifications are available in :ref:`gadf:ogip-arf`.

Such an area can be obtained, for example: 

- selecting the value of an `~gammapy.irf.EffectiveAreaTable2D` at a given offset (using `~astropy.coordinates.Angle`)

.. plot:: irf/plot_aeff_table.py
    :include-source:

- using a pre-defined effective area parameterisation

.. plot:: irf/plot_aeff_param.py
    :include-source:


Energy Dispersion
=================

as a function of of true energy and offset angle (:ref:`gadf:edisp_2d`)
-----------------------------------------------------------------------
The `~gammapy.irf.EnergyDispersion2D` class represents the probability density of the energy migration 
:math:`\mu=\frac{E_{\rm reco}}{E_{\rm true}}` as a function of true energy and offset angle (:math:`E_{\rm disp}(E_{\rm reco}|p, E)` in :ref:`irf-theory`).
Its format specifications are available in :ref:`gadf:edisp_2d`

This is the format in which IACT DL3 energy dispersions are usually provided, as an example

.. plot:: irf/plot_edisp.py
    :include-source:

as a function of true energy (:ref:`gadf:ogip-rmf`)
---------------------------------------------------
`~gammapy.irf.EDispKernel` instead represents an energy dispersion as a function of true energy only 
(:math:`E_{\rm disp}(E_{\rm reco}| E)` following the notation in :ref:`irf-theory`).
Its format specifications are available in :ref:`gadf:ogip-rmf`.
Such an energy dispersion can be obtained for example: 

- selecting the value of an `~gammapy.irf.EnergyDispersion2D` at a given offset (using `~astropy.coordinates.Angle`)

.. plot:: irf/plot_edisp_kernel.py
    :include-source:

- or starting from a parameterisation:

.. plot:: irf/plot_edisp_kernel_param.py
    :include-source:


PSF
===

The `~gammapy.irf.TablePSF` and `~gammapy.irf.EnergyDependentTablePSF` classes
represent radially-symmetric PSFs where the PSF is given at a number of offsets.

The `~gammapy.irf.PSFKernel` represents a PSF kernel.

.. plot:: irf/plot_fermi_psf.py

Background
==========

See `~gammapy.irf.Background2D` and `~gammapy.irf.Background3D`.


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

Reference/API
=============

.. automodapi:: gammapy.irf
    :no-inheritance-diagram:
    :include-all-objects:
