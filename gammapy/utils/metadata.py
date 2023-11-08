# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Metadata base container for Gammapy."""
import json
from typing import Optional, Union, get_args
import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, Angle, EarthLocation, SkyCoord
from astropy.time import Time
import yaml
from pydantic import BaseModel, ValidationError, validator
from gammapy.utils.fits import skycoord_from_dict
from gammapy.version import version

__all__ = ["MetaData", "CreatorMetaData"]

METADATA_FITS_KEYS = {
    "creator": {
        "creator": "CREATOR",
        "date": {
            "input": lambda v: v.get("CREATED"),
            "output": lambda v: {"CREATED": v.iso},
        },
        "origin": "ORIGIN",
    },
    "obs_info": {
        "telescope": "TELESCOP",
        "instrument": "INSTRUME",
        "observation_mode": "OBS_MODE",
        "obs_id": "OBS_ID",
    },
    "pointing": {
        "radec_mean": {
            "input": lambda v: skycoord_from_dict(v, frame="icrs", ext="PNT"),
            "output": lambda v: {"RA_PNT": v.ra.deg, "DEC_PNT": v.dec.deg},
        },
        "altaz_mean": {
            "input": lambda v: skycoord_from_dict(v, frame="altaz", ext="PNT"),
            "output": lambda v: {"ALT_PNT": v.alt.deg, "AZ_PNT": v.az.deg},
        },
    },
    "target": {
        "name": "OBJECT",
        "position": {
            "input": lambda v: skycoord_from_dict(v, frame="icrs", ext="OBJ"),
            "output": lambda v: {"RA_OBJ": v.ra.deg, "DEC_OBJ": v.dec.deg},
        },
    },
}


class MetaData(BaseModel):
    """Base model for all metadata classes in Gammapy."""

    class Config:
        """Global configuration for all metadata."""

        extra = "allow"
        arbitrary_types_allowed = True
        validate_all = True
        validate_assignment = True

        # provides a recipe to export arbitrary types to json
        json_encoders = {
            Angle: lambda v: f"{v.value} {v.unit}",
            u.Quantity: lambda v: f"{v.value} {v.unit}",
            Time: lambda v: f"{v.iso}",
            EarthLocation: lambda v: f"lon : {v.lon.value} {v.lon.unit}, "
            f"lat : {v.lat.value} {v.lat.unit}, "
            f"height : {v.height.value} {v.height.unit}",
            SkyCoord: lambda v: f"lon: {v.spherical.lon.value} {v.spherical.lon.unit}, "
            f"lat: {v.spherical.lat.value} {v.spherical.lat.unit}, "
            f"frame: {v.frame.name} ",
        }

    @property
    def tag(self):
        """Returns MetaData tag."""
        return self._tag

    def to_header(self, format="gadf"):
        """Export MetaData to a FITS header.

        Conversion is performed following the definition in the METADATA_FITS_EXPORT_KEYS.

        Parameters
        ----------
        format : {'gadf'}, optional
            Header format. Default is 'gadf'.

        Returns
        -------
        header : dict
            The header dictionary.
        """

        if format != "gadf":
            raise ValueError(f"Metadata to header: format {format} is not supported.")

        hdr_dict = {}

        fits_export_keys = METADATA_FITS_KEYS.get(self.tag)

        if fits_export_keys is None:
            # TODO: Should we raise an exception or simply a warning and return empty dict?
            raise TypeError(f"No FITS export is defined for metadata {self.tag}.")

        for key, item in fits_export_keys.items():
            value = self.dict().get(key)
            if not isinstance(item, str):
                # Not a one to one conversion
                hdr_dict.update(item["output"](value))
            else:
                if value is not None:
                    hdr_dict[item] = value

        extra_keys = set(self.dict().keys()) - set(fits_export_keys.keys())
        for key in extra_keys:
            entry = getattr(self, key)
            if isinstance(entry, MetaData):
                hdr_dict.update(entry.to_header(format))
        return hdr_dict

    @classmethod
    def from_header(cls, header, format="gadf"):
        """Import MetaData from a FITS header.

        Conversion is performed following the definition in the METADATA_FITS_EXPORT_KEYS.

        Parameters
        ----------
        header : dict
            The header dictionary.
        format : {'gadf'}, optional
            Header format. Default is 'gadf'.
        """
        if format != "gadf":
            raise ValueError(f"Metadata from header: format {format} is not supported.")

        fits_export_keys = METADATA_FITS_KEYS.get(cls._tag)

        if fits_export_keys is None:
            raise TypeError(f"No FITS export is defined for metadata {cls._tag}.")

        kwargs = {}

        for key, item in fits_export_keys.items():
            if not isinstance(item, str):
                # Not a one to one conversion
                kwargs[key] = item["input"](header)
            else:
                kwargs[key] = header.get(item)

        extra_keys = set(cls.__annotations__.keys()) - set(fits_export_keys.keys())
        for key in extra_keys:
            args = get_args(cls.__annotations__[key])
            if issubclass(args[0], MetaData):
                kwargs[key] = args[0].from_header(header, format)

        return cls(**kwargs)

    def to_yaml(self):
        """Dump metadata content to yaml."""
        meta = json.loads(self.json())
        return yaml.dump(
            meta, sort_keys=False, indent=4, width=80, default_flow_style=False
        )


class CreatorMetaData(MetaData):
    """Metadata containing information about the object creation.

    Parameters
    ----------
    creator : str
        The software used to create the data contained in the parent object.
    date : `~astropy.time.Time` or str
        The creation date.
    origin : str
        The organization at the origin of the data.
    """

    _tag = "creator"
    creator: Optional[str]
    date: Optional[Union[str, Time]]
    origin: Optional[str]

    @validator("date")
    def validate_time(cls, v):
        if v is not None:
            return Time(v)
        else:
            return v

    def to_header(self, format="gadf"):
        """Convert creator metadata to fits header.

        Parameters
        ----------
        format : str, optional
            Header format. Default is 'gadf'.

        Returns
        -------
        header : dict
            The header dictionary.
        """
        if format != "gadf":
            raise ValueError(f"Creator metadata: format {format} is not supported.")

        hdr_dict = {}
        hdr_dict["CREATED"] = self.date.iso
        hdr_dict["CREATOR"] = self.creator
        hdr_dict["ORIGIN"] = self.origin

        return hdr_dict

    @classmethod
    def from_default(cls):
        """Creation metadata containing current time and Gammapy version."""
        date = Time.now().iso
        creator = f"Gammapy {version}"
        return cls(creator=creator, date=date)


class ObsInfoMetaData(MetaData):
    """General metadata information about the observation.

    Parameters
    ----------
    obs_id : str or int
        The observation identifier.
    telescope : str, optional
        The telescope/observatory name.
    instrument : str, optional
        The specific instrument used.
    sub_array : str, optional
        The specific sub-array used.
    observation_mode : str, optional
        The observation mode.
    """

    _tag = "obs_info"

    obs_id: Union[str, int]
    telescope: Optional[str]
    instrument: Optional[str]
    sub_array: Optional[str]
    observation_mode: Optional[str]


class PointingInfoMetaData(MetaData):
    """General metadata information about the pointing.

    Parameters
    ----------
    radec_mean : `~astropy.coordinates.SkyCoord`, optional
        Mean pointing position of the observation in `icrs` frame.
    altaz_mean : `~astropy.coordinates.SkyCoord`, or `~astropy.coordinates.AltAz`, optional
        Mean pointing position of the observation in local AltAz frame.
    """

    _tag = "pointing"

    radec_mean: Optional[SkyCoord]
    altaz_mean: Optional[Union[SkyCoord, AltAz]]

    @validator("radec_mean")
    def validate_icrs_position(cls, v):
        if v is None:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="icrs")
        elif isinstance(v, SkyCoord):
            return v.icrs
        else:
            raise ValidationError(
                f"Incorrect position. Expect SkyCoord got {type(v)} instead."
            )

    @validator("altaz_mean")
    def validate_altaz_position(cls, v):
        if v is None:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="altaz")
        elif isinstance(v, AltAz):
            return SkyCoord(v)
        elif isinstance(v, SkyCoord):
            return v.altaz
        else:
            raise ValidationError(
                f"Incorrect position. Expect SkyCoord in altaz frame got {type(v)} instead."
            )


class TargetMetaData(MetaData):
    """General metadata information about the target.

    Parameters
    ----------
    name : str, optional
        The target name.
    position : `~astropy.coordinates.SkyCoord`, optional
        Position of the observation in `icrs` frame.

    """

    _tag = "target"
    name: Optional[str]
    position: Optional[SkyCoord]

    @validator("position")
    def validate_icrs_position(cls, v):
        if v is None:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="icrs")
        elif isinstance(v, SkyCoord):
            return v.icrs
        else:
            raise ValidationError(
                f"Incorrect position. Expect SkyCoord got {type(v)} instead."
            )
