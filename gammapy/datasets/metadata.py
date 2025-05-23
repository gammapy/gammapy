from typing import ClassVar, Literal, Optional, Union
import numpy as np
from astropy.coordinates import SkyCoord
from pydantic import ConfigDict
from gammapy.utils.metadata import (
    METADATA_FITS_KEYS,
    CreatorMetaData,
    MetaData,
    ObsInfoMetaData,
    PointingInfoMetaData,
)

__all__ = ["MapDatasetMetaData"]

MapDataset_METADATA_FITS_KEYS = {
    "MapDataset": {
        "event_types": "EVT_TYPE",
        "optional": "OPTIONAL",
    },
}

METADATA_FITS_KEYS.update(MapDataset_METADATA_FITS_KEYS)


class MapDatasetMetaData(MetaData):
    """Metadata containing information about the Dataset.

    Parameters
    ----------
    creation : `~gammapy.utils.CreatorMetaData`, optional
         The creation metadata.
    obs_info : list of `~gammapy.utils.ObsInfoMetaData`
        info about the observation.
    event_types : list of int or str
        Event types used in analysis.
    pointing: list of `~gammapy.utils.PointingInfoMetaData`
        Telescope pointing directions.
    optional : dict
        Additional optional metadata.
    """

    model_config = ConfigDict(coerce_numbers_to_str=True)

    _tag: ClassVar[Literal["MapDataset"]] = "MapDataset"
    creation: Optional[CreatorMetaData] = CreatorMetaData()
    obs_info: Optional[Union[ObsInfoMetaData, list[ObsInfoMetaData]]] = None
    pointing: Optional[Union[PointingInfoMetaData, list[PointingInfoMetaData]]] = None
    event_type: Optional[Union[str, list[str]]] = None
    optional: Optional[dict] = None

    def stack(self, other):
        kwargs = {}
        kwargs["creation"] = self.creation
        return self.__class__(**kwargs)

    @classmethod
    def _from_meta_table(cls, table):
        """Create MapDatasetMetaData from MapDataset.meta_table

        Parameters
        ----------
            table: `~astropy.table.Table`

        """
        kwargs = {}
        kwargs["creation"] = CreatorMetaData()
        telescope = np.atleast_1d(table["TELESCOP"].data[0])
        obs_id = np.atleast_1d(table["OBS_ID"].data[0].astype(str))
        observation_mode = np.atleast_1d(table["OBS_MODE"].data[0])

        obs_info = []
        for i in range(len(obs_id)):
            obs_meta = ObsInfoMetaData(
                **{
                    "telescope": telescope[i],
                    "obs_id": obs_id[i],
                    "observation_mode": observation_mode[i],
                }
            )
            obs_info.append(obs_meta)
        kwargs["obs_info"] = obs_info

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
            pointings.append(PointingInfoMetaData(radec_mean=pra, altaz_mean=paz))
        kwargs["pointing"] = pointings
        return cls(**kwargs)
