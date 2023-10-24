import logging
from typing import Optional, Union
import numpy as np
from astropy.coordinates import SkyCoord
from pydantic import ValidationError, validator
from gammapy.utils.metadata import CreatorMetaData, MetaData

__all__ = ["MapDatasetMetaData"]


class MapDatasetMetaData(MetaData):
    """Metadata containing information about the GTI.

    Parameters
    ----------
    creation : `~gammapy.utils.CreatorMetaData`
        the creation metadata
    instrument : str
        the instrument used during observation
    telescope : str
        The specific telescope subarray
    observation_mode : str
        observing mode
    pointing : ~astropy.coordinates.SkyCoord
        Telescope pointing direction
    obs_ids : int
        Observation ids stacked in the dataset
    event_types : int
        Event types used in analysis
    optional : dict
        Any other meta information
    """

    creation: Optional[CreatorMetaData]
    instrument: Optional[str]
    telescope: Optional[Union[str, list[str]]]
    observation_mode: Optional[Union[str, list]]
    pointing: Optional[Union[SkyCoord, list[SkyCoord]]]
    obs_ids: Optional[Union[int, list[int]]]
    event_type: Optional[Union[int, list[int]]]
    optional: Optional[dict]

    @validator("creation")
    def validate_creation(cls, v):
        if v is None:
            return CreatorMetaData.from_default()
        elif isinstance(v, CreatorMetaData):
            return v
        else:
            raise ValidationError(
                f"Incorrect pointing. Expect CreatorMetaData got {type(v)} instead."
            )

    @validator("instrument")
    def validate_instrument(cls, v):
        if isinstance(v, str):
            return v
        elif v is None:
            return v
        else:
            raise ValidationError(
                f"Incorrect instrument. Expect str got {type(v)} instead."
            )

    @validator("telescope")
    def validate_telescope(cls, v):
        if isinstance(v, str):
            return v
        elif v is None:
            return v
        elif all(isinstance(_, str) for _ in v):
            return v
        else:
            raise ValidationError(
                f"Incorrect telescope type. Expect str got {type(v)} instead."
            )

    @validator("pointing")
    def validate_pointing(cls, v):
        if v is None:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="icrs")
        elif isinstance(v, SkyCoord):
            return v
        elif all(isinstance(_, SkyCoord) for _ in v):
            return v
        else:
            raise ValidationError(
                f"Incorrect pointing. Expect SkyCoord got {type(v)} instead."
            )

    @validator("obs_ids", "event_type")
    def validate_obs_ids(cls, v):
        if v is None:
            return -999
        elif isinstance(v, int):
            return v
        elif all(isinstance(_, int) for _ in v):
            return v
        else:
            raise ValidationError(
                f"Incorrect pointing. Expect int got {type(v)} instead."
            )

    @classmethod
    def from_default(cls):
        """Creation metadata containing Gammapy version."""
        creation = CreatorMetaData.from_default()
        return cls(creation=creation)

    def stack(self, other):
        kwargs = {}
        kwargs["creation"] = self.creation
        kwargs["instrument"] = self.instrument
        if self.instrument != other.instrument:
            logging.warning(
                f"Stacking data from different instruments {self.instrument} and {other.instrument}"
            )
        tel = self.telescope
        if isinstance(tel, str):
            tel = [tel]
        if other.telescope not in tel:
            tel.append(other.telescope)
        kwargs["telescope"] = tel

        observation_mode = self.observation_mode
        if isinstance(observation_mode, str):
            observation_mode = [observation_mode]
        observation_mode.append(other.observation_mode)
        kwargs["observation_mode"] = observation_mode

        pointing = self.pointing
        if isinstance(pointing, SkyCoord):
            pointing = [pointing]
        pointing.append(other.pointing)
        kwargs["pointing"] = pointing

        obs_ids = self.obs_ids
        if isinstance(obs_ids, int):
            obs_ids = [obs_ids]
        obs_ids.append(other.obs_ids)
        kwargs["obs_ids"] = obs_ids

        event_type = self.event_type
        if not isinstance(event_type, list):
            event_type = [event_type]
        event_type.append(other.event_type)
        kwargs["event_type"] = event_type

        if self.optional:
            optional = self.optional
            for k in other.optional.keys():
                if not isinstance(optional[k], list):
                    optional[k] = [optional[k]]
                optional[k].append(other.optional[k])
            kwargs["optional"] = optional

        return self.__class__(**kwargs)
