.. _irf-psf:

Point Spread Function
=====================

as a function of of true energy and offset angle (:ref:`gadf:psf_table`)
------------------------------------------------------------------------
The `~gammapy.irf.PSF3D` class represents the radially symmetric probability 
density of the angular separation between true and reconstructed directions 
:math:`\delta p = p_{\rm true} - p` (or `rad`), as a function of 
true energy and offset angle from the field of view center 
(:math:`PSF(E_{\rm true}, \delta p|p_{\rm true})` in :ref:`irf-theory`).

Its format specifications are available in :ref:`gadf:psf_table`.

This is the format in which IACT DL3 PSFs are usually provided, as an example:

.. plot:: irf/plot_psf.py
    :include-source:


as a function of true energy (:ref:`gadf:psf_gtpsf`)
----------------------------------------------------
`~gammapy.irf.EnergyDependentTablePSF` instead represents the probability density 
of the angular separation between true direction and reconstructed directions 
:math:`\delta p = p_{\rm true} - p` (or `rad`) as a function of true 
energy only (:math:`PSF(\delta p| E_{\rm true})` following the notation in :ref:`irf-theory`). 

Its format specifications are available in :ref:`gadf:psf_gtpsf`.

Such a PSF can be obtained by evaluating the `~gammapy.irf.PSF3D` at a given offset (using `~astropy.coordinates.Angle`)

.. plot:: irf/plot_psf_table.py
    :include-source:
  
**Note:** the offset angle on the x axis is not an offset angle from the field of view center but the above mentioned 
angular separation between true and reconstructed direction :math:`\delta p`
(in the IRF axis naming convention - see :ref:`irf` - the term `rad` is used for this axis).

additional PSF classes
----------------------

The remaining IRF classes implement:

- `~gammapy.irf.EnergyDependentMultiGaussPSF` a PSF whose probability density is parametrised by the sum of 1 to 3 2-dimensional gaussian (definition at :ref:`gadf:psf_3gauss`);
- `~gammapy.irf.PSFKing` a PSF whose probability density is parametrised by the King function (definition at :ref:`gadf:psf_king`);
- `~gammapy.irf.TablePSF` a radially-symmetric PSF which is not energy dependent;
- `~gammapy.irf.PSFKernel` a PSF that can be used to convolve `~gammapy.maps.WcsNDMap` objects;
