.. _irf-edisp:

Energy Dispersion
=================

as a function of of true energy and offset angle (:ref:`gadf:edisp_2d`)
-----------------------------------------------------------------------
The `~gammapy.irf.EnergyDispersion2D` class represents the probability density of the energy migration 
:math:`\mu=\frac{E}{E_{\rm true}}` as a function of true energy and offset angle from the field of view center
(:math:`E_{\rm disp}(E_{\rm true}, \mu|p_{\rm true})` in :ref:`irf-theory`).

Its format specifications are available in :ref:`gadf:edisp_2d`

This is the format in which IACT DL3 energy dispersions are usually provided, as an example:

.. plot:: irf/plot_edisp.py
    :include-source:

as a function of true energy (:ref:`gadf:ogip-rmf`)
---------------------------------------------------
`~gammapy.irf.EDispKernel` instead represents an energy dispersion as a function of true energy only 
(:math:`E_{\rm disp}(E| E_{\rm true})` following the notation in :ref:`irf-theory`).
`~gammapy.irf.EDispKernel` contains the energy redistribution matrix (or redistribution matrix function, RMF, 
in the OGIP standard). The energy redistribution provides the integral of the energy dispersion probability function over 
bins of reconstructed energy. It is used to convert vectors of predicted counts in true energy in vectors of predicted 
counts in reconstructed energy.

Its format specifications are available in :ref:`gadf:ogip-rmf`.

Such an energy dispersion can be obtained for example: 

- selecting the value of an `~gammapy.irf.EnergyDispersion2D` at a given offset (using `~astropy.coordinates.Angle`)

.. plot:: irf/plot_edisp_kernel.py
    :include-source:

- or starting from a parameterisation:

.. plot:: irf/plot_edisp_kernel_param.py
    :include-source:
