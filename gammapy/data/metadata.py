# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import Optional, Union
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from pydantic import Field, validator
from gammapy.utils.fits import earth_location_from_dict
from gammapy.utils.metadata import CreatorMetaData, MetaData

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
    deadtime : float
        the observation deadtime. Default is 1.
    muon_efficiency : float
        the observation muon efficiency. Default is 1.
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
    """

    telescope: Optional[str]
    instrument: Optional[str]
    observation_mode: Optional[str]
    location: Optional[Union[str, EarthLocation, None]]
    deadtime: float = Field(1.0, gt=0, le=1.0)
    muon_efficiency: float = Field(1.0, gt=0, le=1.0)
    time_start: Optional[Union[Time, str, None]]
    time_stop: Optional[Union[Time, str, None]]
    reference_time: Optional[Union[Time, str, None]]
    target_name: Optional[str]
    target_position: Optional[SkyCoord]
    creation: Optional[CreatorMetaData]

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
    def from_gadf_header(cls, events_hdr):
        """Create and fill the observation metadata from the event list metadata."""
        # TODO: read really from events.meta once it is properly defined
        kwargs = {}
        kwargs["location"] = earth_location_from_dict(events_hdr)

        if "RA_OBJ" in events_hdr and "DEC_OBJ" in events_hdr:
            kwargs["target_position"] = SkyCoord(
                events_hdr["RA_OBJ"], events_hdr["DEC_OBJ"], unit="deg", frame="icrs"
            )

        kwargs["creation"] = CreatorMetaData.from_header(events_hdr)
        return cls(**kwargs)
