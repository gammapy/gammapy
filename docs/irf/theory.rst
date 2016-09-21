.. _irf-theory:

IRF Theory
==========

TODO: do a detailed writeup of how IRFs are implemented and used in Gammapy.

For high-level gamma-ray data analysis (measuring morphology and spectra of sources)
a canonical detector model is used, where the gamma-ray detection process is simplified
as being fully characterized by the following three "instrument response functions":

* Effective area :math:`A(p, E)` (unit: :math:`m^2`)
* Point spread function :math:`PSF(p'|p, E)` (unit: :math:`sr^{-1}`)
* Energy dispersion :math:`D(E'|p, E)` (unit: :math:`TeV^{-1}`)

The effective area represents the gamma-ray detection efficiency,
the PSF the angular resolution and the energy dispersion the energy resolution
of the instrument.

The full instrument response is given by

.. math::

   R(p', E'|p, E) = A(p, E) \times PSF(p'|p, E) \times D(E'|p, E),

where :math:`p` and :math:`E` are the true gamma-ray position and energy
and :math:`p'` and :math:`E'` are the reconstructed gamma-ray position and energy.

The instrument function relates sky flux models to expected observed counts distributions via

.. math::

   N(p', E') = t_{obs} \int_E \int_\Omega R(p', E'|p, E) \times F(p, E) dp dE,

where :math:`F`, :math:`R`, :math:`t_{obs}` and :math:`N` are the following quantities:

* Sky flux model :math:`F(p, E)` (unit: :math:`m^{-2} s^{-1} TeV^{-1} sr^{-1}`)
* Instrument response :math:`R(p', E'|p, E)` (unit: :math:`m^2 TeV^{-1} sr^{-1}`)
* Observation time: :math:`t_{obs}` (unit: :math:`s`)
* Expected observed counts model :math:`N(p', E')` (unit: :math:`sr^{-1} TeV^{-1}`)

If you'd like to learn more about instrument response functions, have a look at the descriptions for
`Fermi <http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_LAT_IRFs/index.html>`__,
for `TeV data analysis <http://inspirehep.net/record/1122589>`__
and for `GammaLib <http://gammalib.sourceforge.net/user_manual/modules/obs.html#handling-the-instrument-response>`__.

TODO: add an overview of what is / isn't available in Gammapy.
