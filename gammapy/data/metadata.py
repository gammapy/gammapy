# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import Optional, Union
import numpy as np
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pydantic import Field, ValidationError, validator
from gammapy.utils.fits import earth_location_from_dict
from gammapy.utils.metadata import CreatorMetaData, MetaData
from gammapy.utils.time import time_ref_from_dict

__all__ = ["ObservationMetaData"]


class ObservationMetaData(MetaData):
    """Metadata containing information about the Observation.

    Parameters
    ----------
    telescope : str, optional
        the telescope/observatory name
    instrument : str, optional
        the specific instrument used
    observation_mode : str, optional
        the observation mode
    location : `~astropy.coordinates.EarthLocation` or str, optional
        the observatory location
    deadtime_fraction : float
        the observation deadtime fraction. Default is 0.
    time_start : Time, str
        the observation start time
    time_stop : Time, str
        the observation stop time
    reference_time : Time, str
        the observation reference time
    target_name : str
        the observation target name
    target_position : SkyCoord
        the target coordinate
    creation : `~gammapy.utils.CreatorMetaData`
        the creation metadata
    optional : dict
        additional optional metadata
    """

    telescope: Optional[str]
    instrument: Optional[str]
    observation_mode: Optional[str]
    location: Optional[Union[str, EarthLocation, None]]
    deadtime_fraction: float = Field(0.0, ge=0, le=1.0)
    time_start: Optional[Union[Time, str, None]]
    time_stop: Optional[Union[Time, str, None]]
    reference_time: Optional[Union[Time, str, None]]
    target_name: Optional[str]
    target_position: Optional[SkyCoord]
    creation: Optional[CreatorMetaData]
    optional: Optional[dict]

    @validator("location")
    def validate_location(cls, v):
        from gammapy.data import observatory_locations

        if isinstance(v, str) and v in observatory_locations.keys():
            return observatory_locations[v]
        elif v is None or isinstance(v, EarthLocation):
            return v
        else:
            raise ValueError("Incorrect location value")

    @validator("time_start", "time_stop", "reference_time")
    def validate_time(cls, v):
        if isinstance(v, str):
            return Time(v)
        elif isinstance(v, Time) or v is None:
            # check size?
            return v
        else:
            raise ValueError("Incorrect time input value.")

    @validator("target_position")
    def validate_position(cls, v):
        if v is None:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="icrs")
        elif isinstance(v, SkyCoord):
            return v
        else:
            raise ValidationError(
                f"Incorrect position. Expect SkyCoord got {type(v)} instead."
            )

    @classmethod
    def from_header(cls, events_hdr, format="gadf"):
        """Create and fill the observation metadata from the event list metadata.

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
        kwargs["telescope"] = events_hdr.get("TELESCOP")
        kwargs["instrument"] = events_hdr.get("INSTRUME")
        kwargs["observation_mode"] = events_hdr.get("OBS_MODE")
        kwargs["deadtime_fraction"] = 1 - events_hdr.get("DEADC")
        kwargs["location"] = earth_location_from_dict(events_hdr)

        reference_time = time_ref_from_dict(events_hdr)
        kwargs["reference_time"] = reference_time
        if "TIME_START" in events_hdr:
            kwargs["time_start"] = reference_time + events_hdr.get("TIME_START") * u.s
        if "TIME_STOP" in events_hdr:
            kwargs["time_stop"] = reference_time + events_hdr.get("TIME_STOP") * u.s

        kwargs["creation"] = CreatorMetaData.from_default()

        # optional gadf entries that are defined attributes of the ObservationMetaData
        kwargs["target_name"] = events_hdr.get("OBJECT")
        if "RA_OBJ" in events_hdr and "DEC_OBJ" in events_hdr:
            kwargs["target_position"] = SkyCoord(
                events_hdr["RA_OBJ"], events_hdr["DEC_OBJ"], unit="deg", frame="icrs"
            )

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
            if key in events_hdr.keys():
                optional[key] = events_hdr[key]
        kwargs["optional"] = optional

        return cls(**kwargs)
