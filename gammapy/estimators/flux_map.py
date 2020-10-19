# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

from gammapy.maps import Map, MapAxis, RegionGeom, RegionNDMap, MapCoord
from gammapy.estimators import FluxPoints
from gammapy.utils.regions import make_region
from gammapy.utils.table import table_from_row_data
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
)

class FluxMap:
    """A flux map container.

    It contains a set of `~gammapy.maps.Map` objects that store the estimated flux as a function of energy as well as
    associated quantities (typically errors, upper limits, delta TS and possibly raw quantities such counts,
    excesses etc). It also contains a reference model to convert the flux values in different formats. Usually, this
    should be the model used to produce the flux map.

    The associated map geometry can use a `RegionGeom` to store the equivalent of flux points, or a `WcsGeom`/`HpxGeom`
    to store an energy dependent flux map.

    The container relies internally on the 'Likelihood' SED type defined in :ref:`gadf:flux-points`
    and offers convenience properties to convert to other flux formats, namely:
    ``dnde``, ``flux``, ``eflux`` or ``e2dnde``. The conversion is done according to the reference model spectral shape.

    Parameters
    ----------
    dnde_ref : `~gammapy.maps.Map`
        the reference spectrum value
    norm : `~gammapy.maps.Map`
        the norm factor
    norm_err : `~gammapy.maps.Map`, optional
        the error on the norm factor. Default is None.
    norm_errn : `~gammapy.maps.Map`, optional
        the negative error on the norm factor. Default is None.
    norm_errp : `~gammapy.maps.Map`, optional
        the positive error on the norm factor. Default is None.
    norm_ul : `~gammapy.maps.Map`, optional
        the upper limit on the norm factor. Default is None.
    ts : `~gammapy.maps.Map`, optional
        the delta TS associated with the flux value. Default is None.
    counts : `~gammapy.maps.Map`, optional
        the number counts value. Default is None.
    ref_model : `~gammapy.modeling.models.SkyModel`, optional
        the reference model to use for conversions. Default in None.
        If None, a model consisting of a point source with a power law spectrum of index 2 is assumed.

    """
    def __init__(
        self,
        dnde_ref,
        norm,
        norm_err=None,
        norm_errn=None,
        norm_errp=None,
        norm_ul=None,
        ts=None,
        counts=None,
        ref_model=None,
    ):
        self.geom = norm.geom
        self.dnde_ref = dnde_ref
        self.norm = norm
        self.norm_err = norm_err
        self.norm_errn = norm_errn
        self.norm_errp = norm_errp
        self.norm_ul = norm_ul
        self.ts = ts
        self.counts = counts

        if ref_model is None:
            ref_model = SkyModel(
                spatial_model=PointSpatialModel(),
                spectral_model=PowerLawSpectralModel(index=2),
            )
        self.ref_model = ref_model

    @property
    def e_ref(self):
        axis = self.geom.axes["energy"]
        return axis.center

    @property
    def e_min(self):
        axis = self.geom.axes["energy"]
        return axis.edges[:-1]

    @property
    def e_max(self):
        axis = self.geom.axes["energy"]
        return axis.edges[1:]

    @property
    def dnde(self):
        return self.dnde_ref * self.norm

    @property
    def dnde_err(self):
        return self.dnde_ref * self.norm_err

    @property
    def dnde_errn(self):
        return self.dnde_ref * self.norm_errn

    @property
    def dnde_errp(self):
        return self.dnde_ref * self.norm_errp

    @property
    def dnde_ul(self):
        return self.dnde_ref * self.norm_ul

    def _to_flux(self):
        """Conversion factor to apply to dnde-like quantities to obtain fluxes."""
        ref_flux = self.ref_model.spectral_model.integral(
            self.e_min, self.e_max
        )
        ref_dnde = self.ref_model.spectral_model(self.e_ref)
        factor = ref_flux / ref_dnde
        return factor[:, np.newaxis, np.newaxis]

    @property
    def flux(self):
        return self.dnde * self._to_flux()

    @property
    def flux_err(self):
        return self.dnde_err * self._to_flux()

    @property
    def flux_errn(self):
        return self.dnde_errn * self._to_flux()

    @property
    def flux_errp(self):
        return self.dnde_errp * self._to_flux()

    @property
    def flux_ul(self):
        return self.dnde_ul * self._to_flux()

    def _to_eflux(self):
        """Conversion factor to apply to dnde-like quantities to obtain fluxes."""
        ref_eflux = self.ref_model.spectral_model.energy_flux(
            self.e_min, self.e_max
        )
        ref_dnde = self.ref_model.spectral_model(self.e_ref)
        factor = ref_eflux / ref_dnde
        return factor[:, np.newaxis, np.newaxis]

    @property
    def eflux(self):
        return self.dnde * self._to_eflux()

    @property
    def eflux_err(self):
        return self.dnde_err * self._to_eflux()

    @property
    def eflux_errn(self):
        return self.dnde_errn * self._to_eflux()

    @property
    def eflux_errp(self):
        return self.dnde_errp * self._to_eflux()

    @property
    def eflux_ul(self):
        return self.dnde_ul * self._to_eflux()

    def _to_e2dnde(self):
        """Conversion factor to apply to dnde-like quantities to obtain e2dnde."""
        factor = self.e_ref ** 2
        return factor[:, np.newaxis, np.newaxis]

    @property
    def e2dnde(self):
        return self.dnde * self._to_e2dnde()

    @property
    def e2dnde_err(self):
        return self.dnde_err * self._to_e2dnde()

    @property
    def e2dnde_errn(self):
        return self.dnde_errn * self._to_e2dnde()

    @property
    def e2dnde_errp(self):
        return self.dnde_errp * self._to_e2dnde()

    @property
    def e2dnde_ul(self):
        return self.dnde_ul * self._to_e2dnde()

    def get_flux_points(self, coord=None):
        """Extract flux point at a given position.

        The flux points are returned in the the form of a `~gammapy.estimators.FluxPoints` object
        (i.e. an `~astropy.table.Table`)

        Parameters
        ----------
        coord : `~astropy.coordinates.SkyCoord`
            the coordinate where the flux points are extracted.

        Returns
        -------
        fluxpoints : `~gammapy.estimators.FluxPoints`
            the flux points object
        """
        if coord is None:
            coord = self.geom.center_skydir
        energies = self.e_ref
        coords = MapCoord.create(dict(skycoord=coord, energy=energies))

        ref = self.dnde_ref.get_by_coord(coords) * self.dnde_ref.unit
        norm = self.norm.get_by_coord(coords) * self.norm.unit
        norm_err, norm_errn, norm_errp, norm_ul = None, None, None, None
        norm_scan, stat_scan = None, None
        if self.norm_err is not None:
            norm_err = self.norm_err.get_by_coord(coords) * self.norm_err.unit
        if self.norm_errn is not None:
            norm_err = (
                self.norm_errn.get_by_coord(coords) * self.norm_errn.unit
            )
        if self.norm_errp is not None:
            norm_err = (
                self.norm_errp.get_by_coord(coords) * self.norm_errp.unit
            )

        rows = []
        for idx, energy in enumerate(self.e_ref):
            result = dict()
            result["e_ref"] = energy
            result["e_min"] = self.e_min[idx]
            result["e_max"] = self.e_max[idx]
            result["ref_dnde"] = ref[idx]
            result["norm"] = norm[idx]
            if norm_err is not None:
                result["norm_err"] = norm_err[idx]
            if norm_errn is not None:
                result["norm_errn"] = norm_errn[idx]
            if norm_errp is not None:
                result["norm_errp"] = norm_errp[idx]
            if norm_ul is not None:
                result["norm_ul"] = norm_ul[idx]
            if norm_scan is not None:
                result["norm_scan"] = norm_scan[idx]
                result["stat_scan"] = stat_scan[idx]
            rows.append(result)
        table = table_from_row_data(rows=rows, meta={"SED_TYPE": "likelihood"})
        return FluxPoints(table)