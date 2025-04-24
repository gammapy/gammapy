.. _irf-edisp:

Energy Dispersion
=================

As a function of of true energy and offset angle (:ref:`gadf:edisp_2d`)
-----------------------------------------------------------------------

The `~gammapy.irf.EnergyDispersion2D` class represents the probability density of the energy migration
:math:`\mu=\frac{E}{E_{\rm true}}` as a function of true energy and offset angle from the field of view center
(:math:`E_{\rm disp}(E_{\rm true}, \mu|p_{\rm true})` in :ref:`irf`.

Its format specifications are available in :ref:`gadf:edisp_2d`.

This is the format in which IACT DL3 energy dispersions are usually provided, as an example:

.. plot:: user-guide/irf/plot_edisp.py
    :include-source:

As a function of true energy (:ref:`gadf:ogip-rmf`)
---------------------------------------------------

`~gammapy.irf.EDispKernel` instead represents an energy dispersion as a function of true energy only
(:math:`E_{\rm disp}(E| E_{\rm true})` following the notation in :ref:`irf`.
`~gammapy.irf.EDispKernel` contains the energy redistribution matrix (or redistribution matrix function, RMF,
in the OGIP standard). The energy redistribution provides the integral of the energy dispersion probability function over
bins of reconstructed energy. It is used to convert vectors of predicted counts in true energy in vectors of predicted
counts in reconstructed energy.

Its format specifications are available in :ref:`gadf:ogip-rmf`.

Such an energy dispersion can be obtained for example:

- selecting the value of an `~gammapy.irf.EnergyDispersion2D` at a given offset (using `~astropy.coordinates.Angle`)

.. plot:: user-guide/irf/plot_edisp_kernel.py
    :include-source:

- or starting from a parameterisation:

.. plot:: user-guide/irf/plot_edisp_kernel_param.py
    :include-source:

Storing the energy dispersion information as a function of sky position
-----------------------------------------------------------------------

The `gammapy.irf.EDispKernelMap` is a four-dimensional `~gammapy.maps.Map` that stores, for each position in the sky,
an `~gammapy.irf.EDispKernel`, which, as described above, depends on true energy.

The `~gammapy.irf.EDispKernel` at a given position can be extracted with `~gammapy.irf.EDispKernelMap.get_edisp_kernel()` by
providing a `~astropy.coordinates.SkyCoord` or `~regions.SkyRegion`.

.. plot::
    :include-source:

    from gammapy.irf import EDispKernelMap
    from gammapy.maps import MapAxis
    from astropy.coordinates import SkyCoord

    # Create a test EDispKernelMap from a gaussian distribution
    energy_axis_true = MapAxis.from_energy_bounds(1,10, 8, unit="TeV", name="energy_true")
    energy_axis = MapAxis.from_energy_bounds(1,10, 5, unit="TeV", name="energy")

    edisp_map = EDispKernelMap.from_gauss(energy_axis, energy_axis_true, 0.3, 0)
    position = SkyCoord(ra=83, dec=22, unit='deg', frame='icrs')

    edisp_kernel = edisp_map.get_edisp_kernel(position)

    # We can quickly check the edisp kernel via the peek() method
    edisp_kernel.peek()

The `gammapy.irf.EDispMap` serves a similar purpose but instead of a true energy axis,
it contains the information of the energy dispersion as a function of the energy migration (:math:`E/ E_{\rm true}`).
It can be converted into a `gammapy.irf.EDispKernelMap` with `gammapy.irf.EDispMap.to_edisp_kernel_map()` and the
`gammapy.irf.EDispKernelMap` at a given position can be extracted in the same way as described above, using `~gammapy.irf.EDispMap.get_edisp_kernel()`
and providing a `~astropy.coordinates.SkyCoord`.
