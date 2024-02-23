# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Metadata base container for Gammapy."""
import json
from typing import ClassVar, Literal, Optional, get_args
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from gammapy.utils.fits import skycoord_from_dict
from gammapy.utils.time import (
    time_ref_from_dict,
    time_ref_to_dict,
    time_relative_to_ref,
)
from gammapy.version import version
from .types import AltAzSkyCoordType, ICRSSkyCoordType, SkyCoordType, TimeType

__all__ = [
    "MetaData",
    "CreatorMetaData",
    "ObsInfoMetaData",
    "PointingInfoMetaData",
    "TimeInfoMetaData",
    "TargetMetaData",
]

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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        validate_default=True,
        use_enum_values=True,
    )

    @property
    def tag(self):
        """Returns MetaData tag."""
        return self._tag

    def to_header(self, format="gadf"):
        """Export MetaData to a FITS header.

        Conversion is performed following the definition in the METADATA_FITS_EXPORT_KEYS.

        Parameters
        ----------
        format : {'gadf'}
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
            value = self.model_dump().get(key)
            if value is not None:
                if not isinstance(item, str):
                    # Not a one to one conversion
                    hdr_dict.update(item["output"](value))
                else:
                    hdr_dict[item] = value

        extra_keys = set(self.model_fields.keys()) - set(fits_export_keys.keys())

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
        format : {'gadf'}
            Header format. Default is 'gadf'.
        """
        # TODO: implement storage of optional metadata
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

        extra_keys = set(cls.model_fields.keys()) - set(fits_export_keys.keys())

        for key in extra_keys:
            value = cls.model_fields[key]
            args = get_args(value.annotation)
            try:
                if issubclass(args[0], MetaData):
                    kwargs[key] = args[0].from_header(header, format)
            except TypeError:
                pass

        try:
            return cls(**kwargs)
        except ValidationError:
            return cls.model_construct(**kwargs)

    def to_yaml(self):
        """Dump metadata content to yaml."""
        meta = {"metadata": json.loads(self.model_dump_json())}
        return yaml.dump(
            meta, sort_keys=False, indent=4, width=80, default_flow_style=False
        )


class CreatorMetaData(MetaData):
    """Metadata containing information about the object creation.

    Parameters
    ----------
    creator : str
        The software used to create the data contained in the parent object.
        Default is the used Gammapy version.
    date : `~astropy.time.Time` or str
        The creation date. Default is the current date.
    origin : str
        The organization at the origin of the data.
    """

    _tag: ClassVar[Literal["creator"]] = "creator"
    creator: Optional[str] = f"Gammapy {version}"
    date: Optional[TimeType] = Field(default_factory=Time.now)
    origin: Optional[str] = None


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

    _tag: ClassVar[Literal["obs_info"]] = "obs_info"

    obs_id: int
    telescope: Optional[str] = None
    instrument: Optional[str] = None
    sub_array: Optional[str] = None
    observation_mode: Optional[str] = None


class PointingInfoMetaData(MetaData):
    """General metadata information about the pointing.

    Parameters
    ----------
    radec_mean : `~astropy.coordinates.SkyCoord`, optional
        Mean pointing position of the observation in `icrs` frame.
    altaz_mean : `~astropy.coordinates.SkyCoord`, or `~astropy.coordinates.AltAz`, optional
        Mean pointing position of the observation in local AltAz frame.
    """

    _tag: ClassVar[Literal["pointing"]] = "pointing"

    radec_mean: Optional[ICRSSkyCoordType] = None
    altaz_mean: Optional[AltAzSkyCoordType] = None


class TimeInfoMetaData(MetaData):
    """General metadata information about the time range of an observation.

    Parameters
    ----------
    time_start : `~astropy.time.Time` or str
        The observation start time.
    time_stop : `~astropy.time.Time` or str
        The observation stop time.
    reference_time : `~astropy.time.Time` or str
        The observation reference time."""

    _tag: ClassVar[Literal["time_info"]] = "time_info"

    time_start: Optional[TimeType] = None
    time_stop: Optional[TimeType] = None
    reference_time: Optional[TimeType] = None

    def to_header(self, format="gadf"):
        if self.reference_time is None:
            return {}
        result = time_ref_to_dict(self.reference_time)
        if self.time_start is not None:
            result["TSTART"] = time_relative_to_ref(self.time_start, result).to_value(
                "s"
            )
        if self.time_stop is not None:
            result["TSTOP"] = time_relative_to_ref(self.time_stop, result).to_value("s")
        return result

    @classmethod
    def from_header(cls, header, format="gadf"):
        kwargs = {}
        try:
            time_ref = time_ref_from_dict(header)
        except KeyError:
            return cls()

        kwargs["reference_time"] = time_ref
        if "TSTART" in header:
            kwargs["time_start"] = time_ref + header["TSTART"] * u.s
        if "TSTOP" in header:
            kwargs["time_stop"] = time_ref + header["TSTOP"] * u.s

        return cls(**kwargs)


class TargetMetaData(MetaData):
    """General metadata information about the target.

    Parameters
    ----------
    name : str, optional
        The target name.
    position : `~astropy.coordinates.SkyCoord`, optional
        Position of the observation in `icrs` frame.

    """

    _tag: ClassVar[Literal["target"]] = "target"
    name: Optional[str] = None
    position: Optional[SkyCoordType] = None

    @field_validator("position", mode="after")
    def validate_radec_mean(cls, v):
        if isinstance(v, SkyCoord):
            return v.icrs
