# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.io import fits

from gammapy.maps import MapCoord, Map
from gammapy.estimators import FluxPoints
from gammapy.utils.table import table_from_row_data
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.utils.scripts import make_path

__all__ = ["FluxMap"]


#TODO replace by ref_dnde in all code
REQUIRED_MAPS = {
    "dnde": ["dnde"],
    "e2dnde": ["e2dnde"],
    "flux": ["flux"],
    "eflux": ["eflux"],
    "likelihood": [
        "dnde_ref",
        "norm",
    ],
}

OPTIONAL_MAPS = {
    "dnde": ["dnde_err", "dnde_errp", "dnde_errn", "dnde_ul", "is_ul"],
    "e2dnde": ["e2dnde_err", "e2dnde_errp", "e2dnde_errn", "e2dnde_ul", "is_ul"],
    "flux": ["flux_err", "flux_errp", "flux_errn", "flux_ul", "is_ul"],
    "eflux": ["eflux_err", "eflux_errp", "eflux_errn", "eflux_ul", "is_ul"],
    "likelihood": ["norm_err", "norm_errn", "norm_errp","norm_ul", "norm_scan", "stat_scan"],
}

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
    norm_scan : `~gammapy.maps.Map`, optional
        the norm values of the test statistic scan. Default is None.
    stat_scan : `~gammapy.maps.Map`, optional
        the test statistic scan values. Default is None.
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
        norm_scan=None,
        stat_scan=None,
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
        self.norm_scan = norm_scan
        self.stat_scan = stat_scan
        self.ts = ts
        self.counts = counts

        if ref_model is None:
            ref_model = self._default_model()
        self.ref_model = ref_model

    @staticmethod
    def _default_model():
        return SkyModel(spatial_model=PointSpatialModel(), spectral_model=PowerLawSpectralModel(index=2))

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
        (which stores the flux points in an `~astropy.table.Table`)

        Parameters
        ---------
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

    def get_map_dict(self, sed_type="likelihood"):
        """Return maps in a given SED type in the form of a dictionary.

        Parameters
        ----------
        sed_type : str
            sed type to convert to. Default is `Likelihood`

        Returns
        -------
        map_dict : dict
            dictionary containing the requested maps.
        """
        result = {}
        for entry in REQUIRED_MAPS[sed_type]:
            result[entry] = self.__getattribute__(entry)

        for entry in OPTIONAL_MAPS[sed_type]:
            res = self.__getattribute__(entry)
            if res is not None:
                result[entry] = res

        return result

    def write(self, filename, overwrite=False, sed_type="likelihood"):
        """Write flux map to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool
            Overwrite file if it exists.
        sed_type : str
            sed type to convert to. Default is `Lielihood`
        """
        self.to_hdulist().writeto(str(make_path(filename)), overwrite=overwrite)

    def to_hdulist(self, sed_type="likelihood"):
        """Convert flux map to list of HDUs.

        For now, one cannot export the reference model.

        Parameters
        ----------
        sed_type : str
            sed type to convert to. Default is `Likelihood`

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            Map dataset list of HDUs.
        """
        exclude_primary = slice(1, None)

        hdu_primary = fits.PrimaryHDU()
        hdulist = fits.HDUList([hdu_primary])

        hdu_primary.header["SED_TYPE"] = sed_type

        map_dict = self.get_map_dict(sed_type)

        for key in map_dict:
            hdulist += map_dict[key].to_hdulist(hdu=key)[exclude_primary]

        return hdulist

    @classmethod
    def from_hdulist(cls, hdulist):
        """Create flux map dataset from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.

        Returns
        -------
        fluxmap : `~gammapy.estimators.FluxMap`
            the flux map.
        """
        try:
            sed_type = hdulist[0].header["SED_TYPE"]
        except KeyError:
            raise ValueError(f"Cannot determine SED type of flux map from primary header.")

        result = {}
        for map_type in REQUIRED_MAPS[sed_type]:
            if map_type.upper() in hdulist:
                result[map_type] = Map.from_hdulist(hdulist, hdu=map_type)
            else:
                raise ValueError(f"Cannot find required map {map_type} for SED type {sed_type}.")

        for map_type in OPTIONAL_MAPS[sed_type]:
            if map_type.upper() in hdulist:
                result[map_type] = Map.from_hdulist(hdulist, hdu=map_type)

        return cls.from_dict(result, sed_type)

    @staticmethod
    def _validate_type(maps, sed_type):
        """Check that map input is valid and correspond to one of the SED type."""
        try:
            required = set(REQUIRED_MAPS[sed_type])
        except:
            raise ValueError(f"Unknown SED type.")

        if not required.issubset(maps.keys()):
            missing = required.difference(maps.keys())
            raise ValueError(
                "Missing maps for sed type '{}':" " {}".format(sed_type, missing)
            )


    @classmethod
    def from_dict(cls, maps, sed_type='likelihood', ref_model=None):
        """Create FluxMap from a dictionary of maps.

        Parameters
        ----------
        maps : dict
            dictionary containing the requested maps.
        sed_type : str
            sed type to convert to. Default is `Likelihood`
        ref_model : `~gammapy.modeling.models.SkyModel`, optional
            the reference model to use for conversions. Default in None.
            If None, a model consisting of a point source with a power law spectrum of index 2 is assumed.

        Returns
        -------
        fluxmap : `~gammapy.estimators.FluxMap`
            the flux map.
        """
        cls._validate_type(maps, sed_type)

        if sed_type == 'likelihood':
            return cls(**maps)
        elif sed_type == 'dnde':
            return cls._from_dnde_dict(maps, ref_model)
        elif sed_type == 'flux':
            return cls._from_flux_dict(maps, ref_model)
        elif sed_type == 'eflux':
            return cls._from_eflux_dict(maps, ref_model)
        elif sed_type == 'e2dnde':
            return cls._from_e2dnde_dict(maps, ref_model)

    @classmethod
    def _from_dnde_dict(cls, maps, ref_model):
        e_ref = maps["dnde"].geom.axes["energy"].center
        ref_dnde = ref_model.spectral_model(e_ref)

        kwargs = {}
        kwargs["norm"] = maps["dnde"]/ref_dnde[:,np.newaxis, np.newaxis]

        for map_type in OPTIONAL_MAPS["dnde"]:
            norm_type = "norm"
            if map_type in maps:
                maps[map_type]/ref_dnde[:,np.newaxis, np.newaxis]
