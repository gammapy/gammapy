.. _irf-theory:

IRF Theory
==========

Modeling the expected number of detected events
-----------------------------------------------

To model the expected number of events a gamma-ray source should produce on a detector
one has to model its effect using an instrument responce function (IRF). In general,
such a function gives the probability to detect a photon emitted from true position :math:`p`
on the sky and true energy :math:`E` at reconstructed position :math:`p_{\rm reco}` and energy
:math:`E_{\rm reco}` and the effective collection area of the detector at position :math:`p`
on the sky and true energy :math:`E`.

We can write the expected number of detected events  :math:`N(p_{\rm reco}, E_{\rm reco})`:

.. math::

   N(p_{\rm reco}, E_{\rm reco}) {\rm d}p_{\rm reco} {\rm d}E_{\rm reco} = 
   t_{\rm obs} \int_E {\rm d}E \, \int_p {\rm d}p \, R(p_{\rm reco}, E_{\rm reco}|p, E) \times \Phi(p, E)

where:

* :math:`R(p_{\rm reco}, E_{\rm reco}|p, E)` is the instrument response  (unit: :math:`{\rm m}^2\,{\rm TeV}^{-1}`)
* :math:`\Phi(p, E)` is the sky flux model  (unit: :math:`{\rm m}^{-2}\,{\rm s}^{-1}\,{\rm TeV}^{-1}\,{\rm sr}^{-1}`)
* :math:`t_{\rm obs}` is the observation time:  (unit: :math:`{\rm s}`)


The Instrument Response Functions
---------------------------------

Most of the time, in high-level gamma-ray data (DL3), we assume that the instrument response can
be simplified as the product of three independent functions:

.. math::

   R(p_{\rm reco}, E_{\rm reco}|p, E) = A_{\rm eff}(p, E) \times PSF(p_{\rm reco}|p, E) \times E_{\rm disp}(E_{\rm reco}|p, E),

where:

* :math:`A_{\rm eff}(p, E)` is the effective collection area of the detector  (unit: :math:`{\rm m}^2`). It is the product
  of the detector collection area times its detection efficiency at true energy :math:`E` and position :math:`p`.
* :math:`PSF(p_{\rm reco}|p, E)` is the point spread function (unit: :math:`{\rm sr}^{-1}`). It gives the probability of
  measuring a position :math:`p_{\rm reco}` when the true position is :math:`p` and the true energy is :math:`E`.
* :math:`E_{\rm disp}(E_{\rm reco}|p, E)` is the energy dispersion (unit: :math:`{\rm TeV}^{-1}`). It gives the probability to
  reconstruct the photon at energy :math:`E_{\rm reco}` when the true energy is :math:`E` and the true position :math:`p`.

The implicit assumption here is that energy dispersion and PSF are completely independent. This is not totally
valid in some situations.

These functions are obtained through Monte-Carlo simulations of gamma-ray showers for different observing conditions,
e.g.  detector configuration, zenith angle of the pointing position, detector state and different event reconstruction
and selection schemes. In the DL3 format, the IRF are distributed for each observing run.

Further details on individuals responses and how they are implemented in gammapy are given in :ref:`irf-aeff`,
:ref:`irf-edisp` and :ref:`irf-psf`.


