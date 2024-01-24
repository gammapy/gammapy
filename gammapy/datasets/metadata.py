import logging
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

    def _stack_linear(self, obj_name, other, kwargs):
        obj = getattr(self, obj_name)
        if obj:
            if not isinstance(obj, list):
                obj = [obj]
            obj.append(getattr(other, obj_name))
        else:
            obj = getattr(other, obj_name)
        kwargs[obj_name] = obj
        return kwargs

    def _stack_unique(self, obj_name, other, kwargs):
        obj = getattr(self, obj_name)
        obj_other = getattr(other, obj_name)
        if obj:
            if not isinstance(obj, list):
                obj = [obj]
            if obj_other not in obj:
                obj.append(obj_other)
        else:
            obj = obj_other
        kwargs[obj_name] = obj
        return kwargs

    def stack(self, other):
        kwargs = {}
        kwargs["creation"] = self.creation
        linear_stack = ["pointing", "event_type", "obs_ids"]
        for i in linear_stack:
            kwargs = self._stack_linear(i, other, kwargs)
        unique_stack = ["observation_mode", "instrument", "telescope"]
        for i in unique_stack:
            if i == "instrument":
                if self.instrument != other.instrument:
                    logging.warning(
                        f"Stacking data from different instruments {self.instrument} and {other.instrument}"
                    )
            kwargs = self._stack_unique(i, other, kwargs)

        if self.optional:
            optional = self.optional
            for k in other.optional.keys():
                if not isinstance(optional[k], list):
                    optional[k] = [optional[k]]
                optional[k].append(other.optional[k])
            kwargs["optional"] = optional

        return self.__class__(**kwargs)
