# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import Optional, Union
from astropy.coordinates import EarthLocation
from astropy.time import Time
from pydantic import Field, validator
from gammapy.utils.fits import earth_location_from_dict
from gammapy.utils.metadata import (
    METADATA_FITS_KEYS,
    CreatorMetaData,
    MetaData,
    ObsInfoMetaData,
    PointingInfoMetaData,
    TargetMetaData,
)

__all__ = ["ObservationMetaData"]

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
    creation : `~gammapy.utils.CreatorMetaData`
        The creation metadata.
    optional : dict
        Additional optional metadata.
    """

    _tag = "observation"
    obs_info: Optional[ObsInfoMetaData]
    pointing: Optional[PointingInfoMetaData]
    target: Optional[TargetMetaData]
    location: Optional[Union[EarthLocation, str]]
    deadtime_fraction: float = Field(0.0, ge=0, le=1.0)
    time_start: Optional[Union[Time, str]]
    time_stop: Optional[Union[Time, str]]
    reference_time: Optional[Union[Time, str]]
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

    @classmethod
    def from_header(cls, header, format="gadf"):
        meta = super(ObservationMetaData, cls).from_header(header, format)

        meta.creation = CreatorMetaData.from_default()
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
