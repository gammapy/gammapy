from typing import ClassVar, Literal, Optional, Union
from pydantic import ConfigDict
from gammapy.utils.metadata import (
    METADATA_FITS_KEYS,
    CreatorMetaData,
    MetaData,
    PointingInfoMetaData,
)

__all__ = ["MapDatasetMetaData"]

MapDataset_METADATA_FITS_KEYS = {
    "MapDataset": {
        "creation": "CREATION",
        "instrument": "INSTRUM",
        "telescope": "TELESCOP",
        "observation_mode": "OBS_MODE",
        "pointing": "POINTING",
        "obs_ids": "OBS_IDS",
        "event_types": "EVT_TYPE",
        "optional": "OPTIONAL",
    },
}

METADATA_FITS_KEYS.update(MapDataset_METADATA_FITS_KEYS)


class MapDatasetMetaData(MetaData):
    """Metadata containing information about the GTI.

    Parameters
    ----------
    creation : `~gammapy.utils.CreatorMetaData`, optional
         The creation metadata.
    instrument : str
        the instrument used during observation.
    telescope : str
        The specific telescope subarray.
    observation_mode : str
        observing mode.
    pointing : ~astropy.coordinates.SkyCoord
        Telescope pointing direction.
    obs_ids : int
        Observation ids stacked in the dataset.
    event_types : int
        Event types used in analysis.
    optional : dict
        Additional optional metadata.
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)

    _tag: ClassVar[Literal["MapDataset"]] = "MapDataset"
    creation: Optional[CreatorMetaData] = CreatorMetaData()
    instrument: Optional[Union[str, list[str]]] = None
    telescope: Optional[Union[str, list[str]]] = None
    observation_mode: Optional[Union[str, list]] = None
    pointing: Optional[Union[PointingInfoMetaData, list[PointingInfoMetaData]]] = None
    obs_ids: Optional[Union[str, list[str]]] = None
    event_type: Optional[Union[str, list[str]]] = None
    optional: Optional[dict] = None

    def stack(self, other):
        kwargs = {}
        kwargs["creation"] = CreatorMetaData()
        return self.__class__(**kwargs)
