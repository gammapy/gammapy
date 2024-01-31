from typing import ClassVar, Literal, Optional, Union
import numpy as np
from astropy.coordinates import SkyCoord
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
        "obs_id": "OBS_ID",
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
    obs_id : int
        Observation id in the dataset.
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
    obs_id: Optional[Union[str, list[str]]] = None
    event_type: Optional[Union[str, list[str]]] = None
    optional: Optional[dict] = None

    def stack(self, other):
        kwargs = {}
        kwargs["creation"] = CreatorMetaData()
        return self.__class__(**kwargs)

    @classmethod
    def from_meta_table(cls, table):
        """Create MapDatasetMetaData from MapDataset.meta_table

        Parameters
        ----------
            table: `~astropy.table.Table`

        """
        kwargs = {}
        kwargs["creation"] = CreatorMetaData()
        if "TELESCOP" in table.colnames:
            kwargs["telescope"] = table["TELESCOP"].data[0]
        if "OBS_ID" in table.colnames:
            kwargs["obs_id"] = table["OBS_ID"].data[0].astype(str)
        if "OBS_MODE" in table.colnames:
            kwargs["observation_mode"] = table["OBS_MODE"].data[0]
        pointing_radec, pointing_altaz = None, None
        if "RA_PNT" in table.colnames:
            pointing_radec = SkyCoord(
                ra=table["RA_PNT"].data[0], dec=table["DEC_PNT"].data[0], unit="deg"
            )
        if "ALT_PNT" in table.colnames:
            pointing_altaz = SkyCoord(
                alt=table["ALT_PNT"].data[0],
                az=table["AZ_PNT"].data[0],
                unit="deg",
                frame="altaz",
            )
        pointings = []
        for pra, paz in zip(
            np.atleast_1d(pointing_radec), np.atleast_1d(pointing_altaz)
        ):
            pointings = PointingInfoMetaData(radec_mean=pra, altaz_mean=paz)
        kwargs["pointing"] = pointings
        return cls(**kwargs)
