.. _irf-bkg:

Background
==========

as a function of reconstructed energy and detector coordinates (:ref:`gadf:bkg_3d`) 
--------------------------------------------------------------------------------------
The `~gammapy.irf.Background3D` class represents a background rate per solid 
angle as a function of detector coordinates and reconstructed energy. 

Its format specifications are available in :ref:`gadf:bkg_3d`.

This is the format in which IACT DL3 background rates are usually provided, as an example:

.. plot:: irf/plot_bkg_3d.py
    :include-source:


as a function of reconstructed energy and offset angle, radially symmetric (:ref:`gadf:bkg_2d`)
-----------------------------------------------------------------------------------------------
The `~gammapy.irf.Background2D` class represents a background rate per solid angle 
as a function of offset angle from the field of view center and reconstructed energy.

Its format specifications are available in :ref:`gadf:bkg_2d`.

You can produce a radially-symmetric background from a list of DL3 observations 
devoid of gamma-ray signal using the tutorial in 
`background_model.html <../tutorials/backgorund_model.html>`__ 
