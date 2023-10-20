# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import ClassVar, Literal
from typing import Optional, Union
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.units.quantity import Quantity
from pydantic import Field, ValidationError, validator
from gammapy.utils.fits import earth_location_from_dict
from gammapy.utils.metadata import (
    METADATA_FITS_KEYS,
    CreatorMetaData,
    MetaData,
    ObsInfoMetaData,
    PointingInfoMetaData,
    TargetMetaData,
)
from gammapy.utils.types import EarthLocationType, TimeType
from gammapy.utils.time import time_ref_from_dict
from .observers import observatory_locations


__all__ = ["ObservationMetaData", "GTIMetaData", "EventListMetaData"]

OBSERVATION_METADATA_FITS_KEYS = {
    "location": {
        "input": lambda v: earth_location_from_dict(v),
        "output": lambda v: {
            "GEOLON": v.lon.deg,
            "GEOLAT": v.lat.deg,
            "ALTITUDE": v.height.to_value("m"),
        },
    },
    "deadtime_fraction": {
        "input": lambda v: 1 - v["DEADC"],
        "output": lambda v: {"DEADC": 1 - v},
    },
}

METADATA_FITS_KEYS["observation"] = OBSERVATION_METADATA_FITS_KEYS


class ObservationMetaData(MetaData):
    """Metadata containing information about the Observation.

    Parameters
    ----------
    obs_info : `~gammapy.utils.ObsInfoMetaData`
        The general observation information.
    pointing : `~gammapy.utils.PointingInfoMetaData
        The pointing metadata.
    target : `~gammapy.utils.TargetMetaData
        The target metadata.
    creation : `~gammapy.utils.CreatorMetaData`
        The creation metadata.
    location : `~astropy.coordinates.EarthLocation` or str, optional
        The observatory location.
    deadtime_fraction : float
        The observation deadtime fraction. Default is 0.
    time_start : `~astropy.time.Time` or str
        The observation start time.
    time_stop : `~astropy.time.Time` or str
        The observation stop time.
    reference_time : `~astropy.time.Time` or str
        The observation reference time.
    optional : dict, optional
        Additional optional metadata.
    """

    _tag: ClassVar[Literal["observation"]] = "observation"
    obs_info: Optional[ObsInfoMetaData] = None
    pointing: Optional[PointingInfoMetaData] = None
    target: Optional[TargetMetaData] = None
    location: Optional[EarthLocationType] = None
    deadtime_fraction: float = Field(0.0, ge=0, le=1.0)
    time_start: Optional[TimeType] = None
    time_stop: Optional[TimeType] = None
    reference_time: Optional[TimeType] = None
    creation: Optional[CreatorMetaData] = None
    optional: Optional[dict] = None

    @classmethod
    def from_header(cls, header, format="gadf"):
        meta = super(ObservationMetaData, cls).from_header(header, format)

        meta.creation = CreatorMetaData()
        # Include additional gadf keywords not specified as ObservationMetaData attributes
        optional_keywords = [
            "OBSERVER",
            "EV_CLASS",
            "TELAPSE",
            "TELLIST",
            "N_TELS",
            "TASSIGN",
            "DST_VER",
            "ANA_VER",
            "CAL_VER",
            "CONV_DEP",
            "CONV_RA",
            "CONV_DEC",
            "TRGRATE",
            "ZTRGRATE",
            "MUONEFF",
            "BROKPIX",
            "AIRTEMP",
            "PRESSURE",
            "RELHUM",
            "NSBLEVEL",
        ]
        optional = dict()
        for key in optional_keywords:
            if key in header.keys():
                optional[key] = header[key]
        meta.optional = optional

        return meta


class GTIMetaData(MetaData):
    """Metadata containing information about the GTI.

    Parameters
    ----------
    reference_time : Time, str
        The GTI reference time.
    """

    _tag: ClassVar[Literal["GTI"]] = "GTI"
    reference_time: Optional[TimeType] = None

    def from_header(cls, header, format="gadf"):
        meta = super(GTIMetaData, cls).from_header(header, format)

        return meta


class EventListMetaData(MetaData):
    """Metadata containing information about the EventList.

    Parameters
    ----------
    obs_id : str
        The observation id.
    telescope : str, optional
        The telescope/observatory name.
    instrument : str, optional
        The specific instrument used.
    observation_mode : str, optional
        The observation mode.
    time_start : Time, optional
        The observation start time.
    time_stop : Time, optional
        The observation stop time.
    reference_time : Time, optional
        The observation reference time.
    live_time : Time
        The livetime of observations.
    deadtime_fraction : float
        The observation deadtime fraction. Default is 0.
    location : `~astropy.coordinates.EarthLocation` or str, optional
        The observatory location.
    creation : `~gammapy.utils.CreatorMetaData`
        The creation metadata.
    optional : dict
        Additional optional metadata.
    """

    obs_id: Union[str]
    telescope: Optional[str]
    instrument: Optional[str]
    observation_mode: Optional[str]
    time_start: Optional[Union[Time, str]]
    time_stop: Optional[Union[Time, str]]
    reference_time: Optional[Union[Time, str]]
    # do we want this to be optional?
    live_time: Optional[Quantity]
    deadtime_fraction: float = Field(0.0, ge=0, le=1.0)
    location: Optional[Union[str, EarthLocation]]
    creation: Optional[CreatorMetaData]
    optional: Optional[dict]

    @validator("location")
    def validate_location(cls, v):
        if isinstance(v, str) and v in observatory_locations.keys():
            return observatory_locations[v]
        elif isinstance(v, EarthLocation):
            return v
        else:
            raise ValueError("Incorrect location value")

    @validator("time_start", "time_stop", "reference_time")
    def validate_time(cls, v):
        if isinstance(v, str):
            return Time(v)
        elif isinstance(v, Time) or v is None:
            return v
        else:
            raise ValueError("Incorrect time input value.")

    @classmethod
    def from_header(cls, events_hdr, format="gadf"):
        """Create and fill the metadata from the event list metadata.

        Parameters
        ----------
        format : str
            the header data format. Default is gadf.
        """
        # TODO: read really from events.meta once it is properly defined
        if not format == "gadf":
            raise ValueError(
                f"Metadata creation from format {format} is not supported."
            )

        kwargs = {}
        kwargs["obs_id"] = events_hdr.get("OBS_ID")
        kwargs["telescope"] = events_hdr.get("TELESCOP")
        kwargs["instrument"] = events_hdr.get("INSTRUME")
        kwargs["observation_mode"] = events_hdr.get("OBS_MODE")

        deadc = events_hdr.get("DEADC")
        if deadc is None:
            raise ValueError("No deadtime correction factor defined.")
        kwargs["deadtime_fraction"] = 1 - deadc

        if set(["GEOLON", "GEOLAT"]).issubset(set(events_hdr)):
            kwargs["location"] = earth_location_from_dict(events_hdr)

        kwargs["live_time"] = events_hdr.get("LIVETIME") * u.s

        reference_time = time_ref_from_dict(events_hdr)
        kwargs["reference_time"] = reference_time

        if "TSTART" in events_hdr:
            kwargs["time_start"] = reference_time + events_hdr.get("TSTART") * u.s
        if "TSTOP" in events_hdr:
            kwargs["time_stop"] = reference_time + events_hdr.get("TSTOP") * u.s
        kwargs["creation"] = CreatorMetaData.from_default()

        # optional gadf entries
        kwargs["target_name"] = events_hdr.get("OBJECT")
        if "RA_OBJ" in events_hdr and "DEC_OBJ" in events_hdr:
            kwargs["target_position"] = SkyCoord(
                events_hdr["RA_OBJ"], events_hdr["DEC_OBJ"], unit="deg", frame="icrs"
            )

        # Include additional gadf keywords not specified as EventListMetaData attributes
        # What do we want/need to include here?
        optional_keywords = [
            "EV_CLASS",
            "TELAPSE",
            "TELLIST",
            "N_TELS",
            "TASSIGN",
            "CONV_DEP",
            "CONV_RA",
            "CONV_DEC",
            "MUONEFF",
        ]
        optional = dict()
        for key in optional_keywords:
            if key in events_hdr.keys():
                optional[key] = events_hdr[key]
        kwargs["optional"] = optional

        return cls(**kwargs)
