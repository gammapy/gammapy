.. _irf-aeff:

Effective area
==============

as a function of true energy and offset angle (:ref:`gadf:aeff_2d`)
-------------------------------------------------------------------
The `~gammapy.irf.EffectiveAreaTable2D` class represents an effective area as a function of true energy and offset angle from the field of view center
(:math:`A_{\rm eff}(E_{\rm true}, p_{\rm true})`, following the notation in :ref:`irf-theory`).

Its format specifications are available in :ref:`gadf:aeff_2d`.

This is the format in which IACT DL3 effective areas are usually provided, as an example

.. plot:: irf/plot_aeff.py
    :include-source:
    
as a function of true energy (:ref:`gadf:ogip-arf`)
---------------------------------------------------
`~gammapy.irf.EffectiveAreaTable` instead represents an effective area as a function of true energy only 
(:math:`A_{\rm eff}(E_{\rm true})` following the notation in :ref:`irf-theory`).

Its format specifications are available in :ref:`gadf:ogip-arf`.

Such an area can be obtained, for example: 

- selecting the value of an `~gammapy.irf.EffectiveAreaTable2D` at a given offset (using `~astropy.coordinates.Angle`)

.. plot:: irf/plot_aeff_table.py
    :include-source:

- using a pre-defined effective area parameterisation

.. plot:: irf/plot_aeff_param.py
    :include-source:
    