.. _irf:

Instrument Response Functions (DL3)
===================================

Typically the IRFs are stored in the form of multidimensional tables giving
the response functions such as the distribution of gamma-like events or the
probability density functions of the reconstructed energy and position.

.. _npred:

Expected number of detected events
----------------------------------

To model the expected number of events a gamma-ray source should produce on a detector,
one has to model its effect using an instrument response function (IRF). In general,
such a function gives:

* for the residual instrumental background events, the probability to detect an event at reconstructed
  position :math:`p` and energy :math:`E`,
* for the expected event number from a gamma-ray source, the probability to detect a photon emitted from
  true position :math:`p_{\rm true}` on the sky and true energy :math:`E_{\rm true}` at reconstructed position
  :math:`p` and energy :math:`E` and the effective collection area of the detector at position :math:`p_{\rm true}`
  on the sky and true energy :math:`E_{\rm true}`.

We can write the expected number of detected events in a bin [:math:`{\rm d}p,\,{\rm d}E`]:

.. math::

   N(p, E) \, {\rm d}p {\rm d}E = {N(p, E)_{\rm bkg}} \, {\rm d}p {\rm d}E + {N(p, E)_{\rm src}} \, {\rm d}p {\rm d}E

with:

.. math::

    {N(p, E)_{\rm bkg}}\,  {\rm d}p {\rm d}E = t_{\rm obs} \int_{E} {\rm d}E \, \int_{p} {\rm d}p \, {\rm Bkg}(p, E)

where :math:`{\rm Bkg}(p, E)` is the instrument response on the residual instrumental background rate (unit: :math:`{\rm s}^{-1}\,{\rm sr}^{-1}`)

and with:

.. math::

   {N(p, E)_{\rm src}} \, {\rm d}p {\rm d}E =
   t_{\rm obs} \int_{E_{\rm true}} {\rm d}E_{\rm true} \, \int_{p_{\rm true}} {\rm d}p_{\rm true} \, R(p, E|p_{\rm true}, E_{\rm true}) \times \Phi(p_{\rm true}, E_{\rm true})

where:

* :math:`R(p, E| p_{\rm true}, E_{\rm true})` is the instrument response  (unit: :math:`{\rm m}^2\,{\rm TeV}^{-1}`)
* :math:`\Phi(p_{\rm true}, E_{\rm true})` is the sky flux model  (unit: :math:`{\rm m}^{-2}\,{\rm s}^{-1}\,{\rm TeV}^{-1}\,{\rm sr}^{-1}`)
* :math:`t_{\rm obs}` is the observation time:  (unit: :math:`{\rm s}`)

Residual instrumental background rate
-------------------------------------

The response :math:`{\rm Bkg}(p, E)` is coming from atmospheric cosmic-rays that are mis-classified as gamma-ray candidates.
They originate from hadrons (mainly protons) and leptons (mainly electrons), depending on the energy. These events
constitute an irreducible source of background when studying gamma-ray emissions; they are also subjects of analysis
(e.g. cosmic ray spectrum, charge composition, isotropy and dipole search), which is beyond the scope of this documentation.

:math:`{\rm Bkg}(p, E)` predicts the rate of such events. This response is complex to derive by the observatories, in order to
cover the whole observational phase space (p, E) and the instrument variations (atmosphere and detectors). Most of the
time, real events are used to build such instrument response. This rate is then delivered by the observatories.
In the DL3 format, this IRF is distributed for each observing run for the :term:`IACT` observatories.

For a standard :term:`3D Analysis` as implemented in Gammapy (with a `~gammapy.modeling.models.FoVBackgroundModel`) or
to make maps with the `~gammapy.makers.RingBackgroundMaker` (see :doc:`/user-guide/makers/ring`), this IRF is necessary.
If this is missing, one can use a measurement of the OFF counts in the :term:`FoV`, e.g. as used in :term:`1D Analysis`
(e.g. :doc:`/user-guide/makers/reflected`).

Note that this function is expressed as function of the reconstructed quantities (here :math:`p` and :math:`E`).


Factorisation of the gamma-ray IRFs
-----------------------------------

Most of the time, in high level gamma-ray data (DL3), we assume that the gamma-ray instrument response can
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

Note that all these functions are expressed as function of the true quantities (here :math:`p_{\rm true}` and
:math:`E_{\rm true}`).

These functions are obtained through Monte-Carlo simulations of gamma-ray showers for different observing conditions,
e.g.  detector configuration, zenith angle of the pointing position, detector state and different event reconstruction
and selection schemes. In the DL3 format, the IRF are distributed for each observing run.

Need of four individual responses
---------------------------------

In order to statistically estimate a gamma-ray source model, the expected number of detected events in each
[:math:`{\rm d}p,\,{\rm d}E`] bin are tested in regards to the measured events
(see :doc:`/user-guide/stats/fit_statistics`). To make such statistical studies, the four individual responses are then
mandatory to be delivered with a data release by the observatories.

Further details on individuals responses and how they are implemented in Gammapy are given in:

.. toctree::
    :maxdepth: 1

    aeff
    bkg
    edisp
    psf


Most of the formats defined at :ref:`gadf:iact-irf` are supported. Currently, there is some support for
Fermi-LAT and other instruments, with ongoing efforts to improve this.

Most users will not use `gammapy.irf` directly, but will instead use IRFs as part of their spectrum,
image or cube analysis (via e.g. the `~gammapy.makers.MapDatasetMaker` during the data reduction,
see :doc:`/user-guide/makers/index`).



IRF axis naming
---------------
In the IRF classes, we use the following axis naming convention:

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

.. minigallery::

    ../examples/tutorials/details/irfs.py
    ../examples/tutorials/analysis-1d/cta_sensitivity.py
