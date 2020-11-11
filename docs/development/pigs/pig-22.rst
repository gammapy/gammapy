.. include:: ../../references.txt

.. _pig-022:

************************************
PIG 22 - Unified flux estimators API
************************************

* Author: Régis Terrier and Axel Donath
* Created: Nov 11, 2020
* Accepted: -
* Status: -
* Discussion: `GH 3075`_

Abstract
========
This pig discusses possible improvements for the API of flux ``Estimator`` results.
We discuss the introduction of a general ``LikelihoodSED`` object that would allow
flux type conversions and that would serve as a base class for ``FluxPoints`` and
``FluxMap`` class. The latter would allow easier handling of ``TSMapEstimator`` results,
in particular regarding serialization and flux conversion.


Introduction
============

Flux estimation is performed by ``Estimators`` in gammapy.  Some perform forward-folding methods to compute flux,
ts, significance and errors at various positions or energy. This is the case of the ``FluxPointsEstimator``,
the ``LightCurveEstimator`` and the ``TSMapEstimator``. Other perform backward folding methods to compute similar
quantities (they compute excesses and associated ts and errors and divide by the exposure in reco energy to deduce flux
quantities). This is the case of ``ExcessMapEstimator`` and ``ExcessProfileEstimator``.

So far, the output of all these estimators is diverse and rely on different conventions for the definition of flux.
There are four types of SED flux estimates described in the gamma astro data format. They are;

- ``dnde`` differential flux which is defined at a given ``e_ref``

- ``e2dnde`` differential energy flux which is defined at a given ``e_ref``

- ``flux`` integral flux defined between ``e_min`` and ``e_max``

- ``eflux`` integral energy flux defined between ``e_min`` and ``e_max``

To convert between these flux types, an assumption must be made on the spectral model.

Besides these, a useful SED type is the so-called ``likelihood`` type introduced by fermipy to represent
SEDs and described in the gamma astro data format (ref). It uses reference fluxes expressed in the above
flux types and a ``norm`` value that is used to derive the actual fluxes. Associated quantities are denoted
``norm_err``, ``norm_ul`` etc.

So far, the only API in gammapy handling these differents flux types is the ``FluxPoints`` object.
It contains a ``Table`` representing one given format above and utility functions allow to convert
the table into another format with e.g.:

.. code::

    fp_dnde = fp.to_sed_type("dnde")
    fp_energy_flux = fp.to_sed_type("eflux", model=PowerLawSpectralModel(index=3))

The conversion is not possible in all directions.

The various estimators implemented so far return different objects.

- Map estimators return a dictionary of ``Map`` objects which are defined as ``flux`` types. Beyond the
fixed flux type, there is no easy API to allow the user to serialize all the ``Maps`` at once.

- ``FluxPointsEstimator`` returns a ``FluxPoints`` object using the ``likelihood`` normalization scheme.

- ``LightCurveEstimator`` relies on the ``FluxPointsEstimator`` in each time interval but converts the output
into a ``Table`` with one row per time interval and flux points stored as an array in each row.

- ``ExcessProfileEstimator`` computes an integral flux (``flux`` type) in a list of regions and a list of energies.
It returns a ``Table`` with one region per row, and energy dependent fluxes stored as an array in each row.

This diversity of output formats and flux types could be simplified with better design for flux quantities in gammapy.
We propose below a generalized flux points API.


Proposal
========

Rely internally on likelihood SED type
--------------------------------------

First we propose that all ``Estimators`` compute quantities following the ``likelihood`` SED type. Beyond the
uniform behavior, his has the advantage of making flux type conversion easier.

To limit code duplication (e.g. for flux conversions), we propose a common base class to describe the format
and contain the required quantities.

.. code::

    class LikelihoodSED: # FluxData / EstimatorResult /... ?
        """General likelihood sed conversion class

        Converts norm values into dnde, flux, etc.

        Parameters
        ----------
        data : dict of `Map` or `Table`
            Mappable containing the sed likelihood data
        spectral_model : `SpectralModel`
            Reference spectral model
        energy_axis : `MapAxis`
            Reference energy axis
        """
        def __init__(self, data, spectral_model, energy_axis=None):
            self._data = _data
            self.spectral_model = spectral_model
            self.energy_axis = energy_axis

        @property
        def energy_axis(self):
            """TODO: either we create the map axis here or it is just passed on init..."""
            try:
                return self.data["norm"].geom.axes["energy"]
            except AttributeError
                return MapAxis.from_table()

        def norm(self):
            """"""
            return self.data["norm"]

        def norm_ul(self):
            """"""
            return self.data["norm"]

        def dnde(self):
            """"""
            # TODO: take care of broadcasting here depending on data
            e_ref = self.energy_axis.center
            dnde_ref = self.spectra_model(e_ref)
            return self.norm * dnde_ref

        def flux(self):
            """"""
            # TODO: take care of broadcasting here depending on data
            e_edges = self.energy_axis.edges
            emin, emax = e_edges[:-1], e_edges[1:]
            dnde_ref = self.spectra_model.integral(emin, emax)
            return self.norm * dnde_ref


Introduce a FluxMap API
-----------------------

.. code::

    class FluxMaps(LikelihoodSED):
        """"""
        def __init__(self, maps, ref_model):
            super().__init__(data=table, spectra_model=spectra_model, energy_axis)

        @property
        def maps(self):
            return self._data

        def get_flux_points(self, positions, regions):
            return FluxPointsCollection()

        def sparsify(self, ts_threshold=None):
            """"""
            return FluxPointsCollection()

        def read(filename):
            pass

        def write(filename):
            pass





