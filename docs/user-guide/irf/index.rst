.. _irf:

Instrument Response Functions (DL3)
===================================

Typically the IRFs are stored in the form of multidimensional tables giving
the response functions such as the distribution of gamma-like events or the
probability density functions of the reconstructed energy and position.

Expected number of detected events
----------------------------------

To model the expected number of events a gamma-ray source should produce on a detector
one has to model its effect using an instrument response function (IRF). In general,
such a function gives the probability to detect a photon emitted from true position :math:`p_{\rm true}`
on the sky and true energy :math:`E_{\rm true}` at reconstructed position :math:`p` and energy
:math:`E` and the effective collection area of the detector at position :math:`p_{\rm true}`
on the sky and true energy :math:`E_{\rm true}`.

We can write the expected number of detected events  :math:`N(p, E)`:

.. math::

   N(p, E) {\rm d}p {\rm d}E = 
   t_{\rm obs} \int_{E_{\rm true}} {\rm d}E_{\rm true} \, \int_{p_{\rm true}} {\rm d}p_{\rm true} \, R(p, E|p_{\rm true}, E_{\rm true}) \times \Phi(p_{\rm true}, E_{\rm true})

where:

* :math:`R(p, E| p_{\rm true}, E_{\rm true})` is the instrument response  (unit: :math:`{\rm m}^2\,{\rm TeV}^{-1}`)
* :math:`\Phi(p_{\rm true}, E_{\rm true})` is the sky flux model  (unit: :math:`{\rm m}^{-2}\,{\rm s}^{-1}\,{\rm TeV}^{-1}\,{\rm sr}^{-1}`)
* :math:`t_{\rm obs}` is the observation time:  (unit: :math:`{\rm s}`)


Factorisation of the IRFs
-------------------------

Most of the time, in high level gamma-ray data (DL3), we assume that the instrument response can
be simplified as the product of three independent functions:

.. math::

   R(p, E|p_{\rm true}, E_{\rm true}) = A_{\rm eff}(p_{\rm true}, E_{\rm true}) \times PSF(p|p_{\rm true}, E_{\rm true}) \times E_{\rm disp}(E|p_{\rm true}, E_{\rm true}),

where:

* :math:`A_{\rm eff}(p_{\rm true}, E_{\rm true})` is the effective collection area of the detector  (unit: :math:`{\rm m}^2`). It is the product
  of the detector collection area times its detection efficiency at true energy :math:`E_{\rm true}` and position :math:`p_{\rm true}`.
* :math:`PSF(p|p_{\rm true}, E_{\rm true})` is the point spread function (unit: :math:`{\rm sr}^{-1}`). It gives the probability of
  measuring a direction :math:`p` when the true direction is :math:`p_{\rm true}` and the true energy is :math:`E_{\rm true}`.
  Gamma-ray instruments consider the probability density of the angular separation between true and reconstructed directions 
  :math:`\delta p = p_{\rm true} - p`, i.e. :math:`PSF(\delta p|p_{\rm true}, E_{\rm true})`.
* :math:`E_{\rm disp}(E|p_{\rm true}, E_{\rm true})` is the energy dispersion (unit: :math:`{\rm TeV}^{-1}`). It gives the probability to
  reconstruct the photon at energy :math:`E` when the true energy is :math:`E_{\rm true}` and the true position :math:`p_{\rm true}`.
  Gamma-ray instruments consider the probability density of the migration :math:`\mu=\frac{E}{E_{\rm true}}`, 
  i.e. :math:`E_{\rm disp}(\mu|p_{\rm true}, E_{\rm true})`.

The implicit assumption here is that energy dispersion and PSF are completely independent. This is not totally
valid in some situations.

These functions are obtained through Monte-Carlo simulations of gamma-ray showers for different observing conditions,
e.g.  detector configuration, zenith angle of the pointing position, detector state and different event reconstruction
and selection schemes. In the DL3 format, the IRF are distributed for each observing run.

Further details on individuals responses and how they are implemented in gammapy are given in :ref:`irf-aeff`,
:ref:`irf-edisp` and :ref:`irf-psf`.

Most of the formats defined at :ref:`gadf:iact-irf` are supported.
At the moment, there is little support for Fermi-LAT or other instruments.

Most users will not use `gammapy.irf` directly, but will instead use IRFs as
part of their spectrum, image or cube analysis to compute exposure and effective
EDISP and PSF for a given dataset.


IRF axis naming
---------------
In the IRF classes we use the following axis naming convention:

================= ============================================================================
Variable          Definition
================= ============================================================================
``energy``        Reconstructed energy axis (:math:`E`)
``energy_true``   True energy axis (:math:`E_{\rm true}`)
``offset``        Field of view offset from center (:math:`p_{\rm true}`)
``fov_lon``       Field of view	longitude
``fov_lat``       Field of view latitude
``migra``         Energy migration (:math:`\mu`)
``rad``        	  Offset angle from source position (:math:`\delta p`)
================= ============================================================================


Using gammapy.irf
-----------------

.. minigallery:: gammapy.irf.PSFMap
    :add-heading:


.. minigallery:: gammapy.irf.EDispKernelMap
    :add-heading:


.. minigallery:: gammapy.irf.load_cta_irfs
    :add-heading:


.. toctree::
    :maxdepth: 1
    :hidden:

    aeff
    bkg
    edisp
    psf
