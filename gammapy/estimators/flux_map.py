# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from gammapy.data import GTI
from gammapy.maps import MapCoord, Map
from gammapy.estimators.core import FluxEstimate
from gammapy.estimators.flux_point import FluxPoints
from gammapy.utils.table import table_from_row_data
from gammapy.modeling.models import (
    SkyModel,
    PowerLawSpectralModel,
    PointSpatialModel,
    Models,
)
from gammapy.utils.scripts import make_path

__all__ = ["FluxMaps"]


REQUIRED_MAPS = {
    "dnde": ["dnde"],
    "e2dnde": ["e2dnde"],
    "flux": ["flux"],
    "eflux": ["eflux"],
    "likelihood": ["norm"],
}

#TODO: add an entry for is_ul?
OPTIONAL_MAPS = {
    "dnde": ["dnde_err", "dnde_errp", "dnde_errn", "dnde_ul"],
    "e2dnde": ["e2dnde_err", "e2dnde_errp", "e2dnde_errn", "e2dnde_ul"],
    "flux": ["flux_err", "flux_errp", "flux_errn", "flux_ul"],
    "eflux": ["eflux_err", "eflux_errp", "eflux_errn", "eflux_ul"],
    "likelihood": ["norm_err", "norm_errn", "norm_errp","norm_ul", "norm_scan", "stat_scan"],
}

log = logging.getLogger(__name__)

class FluxMaps(FluxEstimate):
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
    gti : `~gammapy.data.GTI`
        the maps GTI information. Default is None.
    """
    def __init__(self, data, reference_model=None, gti=None):
        self.geom = data['norm'].geom

        if reference_model == None:
            log.warning("No reference model set for FluxMaps. Assuming point source with E^-2 spectrum.")
            reference_model = self._default_model()

        self.reference_model = reference_model

        self.gti = gti

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
            map_dict = self.data
        else:
            map_dict = {}
            for entry in REQUIRED_MAPS[sed_type]:
                map_dict[entry] = self.__getattribute__(entry)

            for entry in OPTIONAL_MAPS[sed_type]:
                try:
                    map_dict[entry] = self.__getattribute__(entry)
                except KeyError:
                    pass

            for key in self.data.keys() - (REQUIRED_MAPS["likelihood"] + OPTIONAL_MAPS["likelihood"]):
                map_dict[key] = self.data[key]

        return map_dict

    def write(self, filename, filename_model=None, overwrite=False, sed_type="likelihood"):
        """Write flux map to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        filename_model : str
            Filename of the model (yaml format).
            If None, keep string before '.' and add '_model.yaml' suffix
        overwrite : bool
            Overwrite file if it exists.
        sed_type : str
            sed type to convert to. Default is `Lielihood`
        """
        filename = make_path(filename)

        if filename_model is None:
            name_string = filename.as_posix()
            for suffix in filename.suffixes:
                name_string.replace(suffix,'')
            filename_model = name_string + '_model.yaml'
        filename_model=make_path(filename_model)

        hdulist = self.to_hdulist(sed_type)

        models = Models(self.reference_model)
        models.write(filename_model, overwrite=overwrite)
        hdulist[0].header['MODEL'] = filename_model.as_posix()

        hdulist.writeto(str(make_path(filename)), overwrite=overwrite)

    def to_hdulist(self, sed_type="likelihood", hdu_bands=None):
        """Convert flux map to list of HDUs.

        For now, one cannot export the reference model.

        Parameters
        ----------
        sed_type : str
            sed type to convert to. Default is `Likelihood`
        hdu_bands : str
            Name of the HDU with the BANDS table. Default is 'BANDS'
            If set to None, each map will have its own hdu_band

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
            hdulist += map_dict[key].to_hdulist(hdu=key, hdu_bands=hdu_bands)[exclude_primary]

        if self.gti:
            hdu = fits.BinTableHDU(self.gti.table, name="GTI")
            hdulist.append(hdu)

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
        flux_map : `~gammapy.estimators.FluxMaps`
            Flux map.
        """
        with fits.open(str(make_path(filename)), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist)


    @classmethod
    def from_hdulist(cls, hdulist, hdu_bands=None):
        """Create flux map dataset from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.
        hdu_bands : str
            Name of the HDU with the BANDS table. Default is 'BANDS'
            If set to None, each map should have its own hdu_band

        Returns
        -------
        fluxmaps : `~gammapy.estimators.FluxMaps`
            the flux map.
        """
        try:
            sed_type = hdulist[0].header["SED_TYPE"]
        except KeyError:
            raise ValueError(f"Cannot determine SED type of flux map from primary header.")

        result = {}
        for map_type in REQUIRED_MAPS[sed_type]:
            if map_type.upper() in hdulist:
                result[map_type] = Map.from_hdulist(hdulist, hdu=map_type, hdu_bands=hdu_bands)
            else:
                raise ValueError(f"Cannot find required map {map_type} for SED type {sed_type}.")

        for map_type in OPTIONAL_MAPS[sed_type]:
            if map_type.upper() in hdulist:
                result[map_type] = Map.from_hdulist(hdulist, hdu=map_type, hdu_bands=hdu_bands)

        # Read additional image hdus
        for hdu in hdulist[1:]:
            if hdu.is_image:
                if hdu.name.lower() not in (REQUIRED_MAPS[sed_type]+OPTIONAL_MAPS[sed_type]):
                    result[hdu.name.lower()] = Map.from_hdulist(hdulist, hdu=hdu.name, hdu_bands=hdu_bands)

        model_filename = hdulist[0].header.get("MODEL", None)

        reference_model = None
        if model_filename:
            try:
                reference_model = Models.read(model_filename)[0]
            except FileNotFoundError:
                raise FileNotFoundError(f"Cannot find {model_filename} model file. Check MODEL keyword.")

        if "GTI" in hdulist:
            gti = GTI(Table.read(hdulist["GTI"]))
        else:
            gti = None

        return cls.from_dict(result, sed_type, reference_model, gti)

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
    def from_dict(cls, maps, sed_type='likelihood', reference_model=None, gti=None):
        """Create FluxMaps from a dictionary of maps.

        Parameters
        ----------
        maps : dict
            dictionary containing the requested maps.
        sed_type : str
            sed type to convert to. Default is `Likelihood`
        reference_model : `~gammapy.modeling.models.SkyModel`, optional
            the reference model to use for conversions. Default in None.
            If None, a model consisting of a point source with a power law spectrum of index 2 is assumed.
        gti : `~gammapy.data.GTI`
            the maps GTI information. Default is None.

        Returns
        -------
        fluxmaps : `~gammapy.estimators.FluxMaps`
            the flux map.
        """
        cls._validate_type(maps, sed_type)

        if sed_type == 'likelihood':
            return cls(maps, reference_model)

        e_ref = maps[sed_type].geom.axes["energy"].center
        e_edges = maps[sed_type].geom.axes["energy"].edges

        if reference_model is None:
            log.warning("No reference model set for FluxMaps. Assuming point source with E^-2 spectrum.")
            reference_model = cls._default_model()

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

        # We add the remaining maps
        for key in maps.keys() - (REQUIRED_MAPS[sed_type] + OPTIONAL_MAPS[sed_type]):
            data[key] = maps[key]

        return cls(data, reference_model, gti)
