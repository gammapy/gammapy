.. _irf-theory:

IRF Theory
==========

Modeling the expected number of detected events
-----------------------------------------------

To model the expected number of events a gamma-ray source should produce on a detector
one has to model its effect using an instrument responce function (IRF). In general,
such a function gives the probability to detect a photon emitted from true position :math:`p`
on the sky and true energy :math:`E` at reconstructed position :math:`p_{rec}` and energy
:math:`E_{rec}` and the effective collection area of the detector at position :math:`p`
on the sky and true energy :math:`E`.

We can write the expected number of detected events  :math:`N(p_{rec}, E_{rec})`:

.. math::

   N(p_{rec}, E_{rec}) dp_{rec} dE_{rec} = t_{obs} \int_E \int_p R(p_{rec}, E_{rec}|p, E) \times \phi(p, E) dp dE

where:

* :math:`R(p_{rec}, E_{rec}|p, E)` is the instrument response  (unit: :math:`m^2 TeV^{-1}`)
* :math:`\Phi(p, E)` is the sky flux model  (unit: :math:`m^{-2} s^{-1} TeV^{-1} sr^{-1}`)
* :math:`t_{obs}` is the observation time:  (unit: :math:`s`)


The Instrument Response Functions
---------------------------------

Most of the time, in high-level gamma-ray data (DL3), we assume that the instrument response can
be simplified as the product of three independent functions:

.. math::

   R(p_{rec}, E_{rec}|p, E) = A_{eff}(p, E) \times PSF(p_{rec}|p, E) \times Edisp(E_{rec}|p, E),

where:

* :math:`A_{eff}(p, E)` is the effective collection area of the detector  (unit: :math:`m^2`). It is the product
  of the detector collection area times its detection efficiency at true energy :math:`E` and position :math:`p`.
* :math:`PSF(p_{rec}|p, E)` is the point spread function (unit: :math:`sr^{-1}`). It gives the probability of
  measuring position :math:`p_{rec}` when the true position is :math:`p` as a function of true energy :math:`E`.
* :math:`Edisp(E_{rec}|p, E)` is the energy dispersion (unit: :math:`TeV^{-1}`). It gives the probability to
  reconstruct the photon at energy :math:`E_{rec}` when the true energy is :math:`E` as a function of position :math:`p`.

The implicit assumption here is that energy dispersion and PSF are completely independent. This is not totally
valid in some situations.

These functions are obtained through Monte-Carlo simulations of gamma-ray showers for different observing conditions,
e.g.  detector configuration, zenith angle of the pointing position, detector state and different event reconstruction
and selection schemes. In the DL3 format, the IRF are distributed for each observing run.

Further details on individuals responses and how they are implemented in gammapy are given in :ref:`irf-aeff`,
:ref:`irf-edisp` and :ref:`irf-psf`.


