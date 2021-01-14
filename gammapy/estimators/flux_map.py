# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.io import fits
from gammapy.maps import MapCoord, Map
from gammapy.estimators.flux_point import FluxPoints
from gammapy.estimators.flux_estimate import FluxEstimate
from gammapy.utils.table import table_from_row_data
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
)
from gammapy.utils.scripts import make_path

__all__ = ["FluxMap"]


REQUIRED_MAPS = {
    "dnde": ["dnde"],
    "e2dnde": ["e2dnde"],
    "flux": ["flux"],
    "eflux": ["eflux"],
    "likelihood": ["norm"],
}

OPTIONAL_MAPS = {
    "dnde": ["dnde_err", "dnde_errp", "dnde_errn", "dnde_ul", "is_ul"],
    "e2dnde": ["e2dnde_err", "e2dnde_errp", "e2dnde_errn", "e2dnde_ul", "is_ul"],
    "flux": ["flux_err", "flux_errp", "flux_errn", "flux_ul", "is_ul"],
    "eflux": ["eflux_err", "eflux_errp", "eflux_errn", "eflux_ul", "is_ul"],
    "likelihood": ["norm_err", "norm_errn", "norm_errp","norm_ul", "norm_scan", "stat_scan"],
}

log = logging.getLogger(__name__)

class FluxMap(FluxEstimate):
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
    data : dict of `~gammapy.maps.Map`
        the maps dictionary. Expected entries are the following:
        * norm : the norm factor
        * norm_err : optional, the error on the norm factor.
        * norm_errn : optional, the negative error on the norm factor.
        * norm_errp : optional, the positive error on the norm factor.
        * norm_ul : optional, the upper limit on the norm factor.
        * norm_scan : optional, the norm values of the test statistic scan.
        * stat_scan : optional, the test statistic scan values.
        * ts : optional, the delta TS associated with the flux value.
        * counts : optional, the number counts value.
    reference_model : `~gammapy.modeling.models.SkyModel`, optional
        the reference model to use for conversions. Default in None.
        If None, a model consisting of a point source with a power law spectrum of index 2 is assumed.

    """
    def __init__(self, data, reference_model=None):
        self.geom = data['norm'].geom

        if reference_model == None:
            log.warning("No reference model set for FluxMap. Assuming point source with E^-2 spectrum.")
            reference_model = self._default_model()
            
        self.reference_model = reference_model

        super().__init__(data, spectral_model=reference_model.spectral_model)

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

        ref = self.dnde_ref

        fp = dict()
        fp["norm"] = self.norm.get_by_coord(coords) * self.norm.unit

        for quantity in self._available_quantities:
            norm_quantity = f"norm_{quantity}"
            res = self.__getattribute__(norm_quantity).get_by_coord(coords)
            res *= self.__getattribute__(norm_quantity).unit
            fp[norm_quantity] = res

        # TODO: add support of norm and stat scan

        rows = []
        for idx, energy in enumerate(self.e_ref):
            result = dict()
            result["e_ref"] = energy
            result["e_min"] = self.e_min[idx]
            result["e_max"] = self.e_max[idx]
            result["ref_dnde"] = ref[idx]
            result["norm"] = fp["norm"][idx]
            for quantity in self._available_quantities:
                norm_quantity = f"norm_{quantity}"
                result[norm_quantity] = fp[norm_quantity][idx]
            rows.append(result)
        table = table_from_row_data(rows=rows, meta={"SED_TYPE": "likelihood"})
        return FluxPoints(table)

    def to_dict(self, sed_type="likelihood"):
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
        if sed_type == "likelihood":
            result = self.data
        else:
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

        map_dict = self.to_dict(sed_type)

        for key in map_dict:
            hdulist += map_dict[key].to_hdulist(hdu=key)[exclude_primary]

        return hdulist

    @classmethod
    def read(cls, filename):
        """Read map dataset from file.

        Parameters
        ----------
        filename : str
            Filename to read from.

        Returns
        -------
        flux_map : `~gammapy.estimators.FluxMap`
            Flux map.
        """
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist)


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
    def from_dict(cls, maps, sed_type='likelihood', reference_model=None):
        """Create FluxMap from a dictionary of maps.

        Parameters
        ----------
        maps : dict
            dictionary containing the requested maps.
        sed_type : str
            sed type to convert to. Default is `Likelihood`
        reference_model : `~gammapy.modeling.models.SkyModel`, optional
            the reference model to use for conversions. Default in None.
            If None, a model consisting of a point source with a power law spectrum of index 2 is assumed.

        Returns
        -------
        fluxmap : `~gammapy.estimators.FluxMap`
            the flux map.
        """
        cls._validate_type(maps, sed_type)

        if sed_type == 'likelihood':
            return cls(maps, reference_model)

        e_ref = maps[sed_type].geom.axes["energy"].center
        e_edges = maps[sed_type].geom.axes["energy"].edges
        ref_dnde = reference_model.spectral_model(e_ref)

        if sed_type == "dnde":
            factor = ref_dnde
        elif sed_type == "flux":
            factor = reference_model.spectral_model.integral(e_edges[:-1], e_edges[1:])
        elif sed_type == "eflux":
            factor = reference_model.spectral_model.energy_flux(e_edges[:-1], e_edges[1:])
        elif sed_type == "e2dnde":
            factor = e_ref ** 2 * ref_dnde

        data = dict()
        data["norm"] = maps[sed_type]/factor[:,np.newaxis, np.newaxis]

        for map_type in OPTIONAL_MAPS[sed_type]:
            if map_type in maps:
                norm_type = map_type.replace(sed_type, "norm")
                data[norm_type] = maps[map_type]/factor[:,np.newaxis, np.newaxis]

        return cls(data, reference_model)
