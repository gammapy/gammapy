.. _irf-psf:

Point Spread Function
=====================

As a function of of true energy and offset angle (:ref:`gadf:psf_table`)
------------------------------------------------------------------------
The `~gammapy.irf.PSF3D` class represents the radially symmetric probability
density of the angular separation between true and reconstructed directions
:math:`\delta p = p_{\rm true} - p` (or `rad`), as a function of
true energy and offset angle from the field of view center
(:math:`PSF(E_{\rm true}, \delta p|p_{\rm true})` in :ref:`irf-theory`).

Its format specifications are available in :ref:`gadf:psf_table`.

This is the format in which IACT DL3 PSFs are usually provided, as an example:

.. plot:: user-guide/irf/plot_psf.py
    :include-source:

Additional PSF classes
----------------------

The remaining IRF classes implement:

- `~gammapy.irf.EnergyDependentMultiGaussPSF` a PSF whose probability density is parametrised by the sum of 1 to 3 2-dimensional gaussian (definition at :ref:`gadf:psf_3gauss`);
- `~gammapy.irf.PSFKing` a PSF whose probability density is parametrised by the King function (definition at :ref:`gadf:psf_king`);
- `~gammapy.irf.PSFKernel` a PSF that can be used to convolve `~gammapy.maps.WcsNDMap` objects;
- `~gammapy.irf.PSFMap` a four-dimensional `~gammapy.maps.Map` storing a `~gammapy.irf.PSFKernel` for each sky position.
