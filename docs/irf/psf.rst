.. _irf-psf:

PSF
===
as a function of of true energy and offset angle (:ref:`gadf:psf_table`)
------------------------------------------------------------------------
The `~gammapy.irf.PSF3D` class represents the spatial probability of an event with true position :math:`p_{\rm true}` to
have an estimated position :math:`p_{\rm reco}`, as a function of true energy and offset angle from the field of view center
(:math:`PSF(p_{\rm reco}|p, E)` in :ref:`irf-theory`).

Its format specifications are available in :ref:`gadf:psf_table`.

This is the format in which IACT DL3 PSFs are usually provided, as an example:

.. plot:: irf/plot_psf.py
    :include-source:


as a function of true energy (:ref:`gadf:psf_gtpsf`)
----------------------------------------------------
`~gammapy.irf.EnergyDependentTablePSF` instead represents an energy dispersion as a function of true energy only 
(:math:`PSF(p_{\rm reco}| E)` following the notation in :ref:`irf-theory`). 

Its format specifications are available in :ref:`gadf:psf_gtpsf`.

Such a PSF can be obtained by evaluating the `~gammapy.irf.PSF3D` at a given offset (using `~astropy.coordinates.Angle`)

.. plot:: irf/plot_psf_table.py
    :include-source:

**Note:** the offset angle on the x axis is not an offset angle from the field of view center but the distance between 
the source position and the reconstructed gamma ray position.  

additional PSF classes
----------------------

The remaining IRF classes implement a particular type of PSF:

 - `~gammapy.irf.EnergyDependentMultiGaussPSF` implements :ref:`gadf:psf_3gauss`;
 - `~gammapy.irf.PSFKing` implements :ref:`gadf:psf_king`;
 - `~gammapy.irf.TablePSF` implements a radially-symmetric PSF which is not energy dependent;
 - `~gammapy.irf.PSFKernel` implements a PSF that can be used to convolve `~gammapy.maps.WcsNDMap` objects;
