.. _irf-aeff:

Effective area
==============

as a function of true energy and offset angle (:ref:`gadf:aeff_2d`)
-------------------------------------------------------------------
The `~gammapy.irf.EffectiveAreaTable2D` class represents an effective area as a function of true energy and offset angle from the field of view center
(:math:`A_{\rm eff}(E_{\rm true}, p_{\rm true})`, following the notation in :ref:`irf`).

Its format specifications are available in :ref:`gadf:aeff_2d`.

This is the format in which IACT DL3 effective areas are usually provided, as an example

.. plot:: user-guide/irf/plot_aeff.py
    :include-source:

- using a pre-defined effective area parameterisation

.. plot:: user-guide/irf/plot_aeff_param.py
    :include-source:
