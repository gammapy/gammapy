# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from enum import Enum
from typing import List, Optional
import numpy as np
from astropy.coordinates import SkyCoord
from pydantic import ValidationError, validator
from gammapy.utils.metadata import FITS_META_KEYS as CREATOR_META_KEYS
from gammapy.utils.metadata import CreatorMetaData, MetaData

__all__ = ["FluxMetaData"]


FITS_META_KEYS = {
    "sed_type": "SED_TYPE",
    "sed_type_init": "SEDTYPEI",
    "n_sigma": "N_SIGMA",
    "ul_conf": "UL_CONF",
    # "n_sigma_ul": "NSIGMAUL",
    "sqrt_ts_threshold_ul": "STSTHUL",
    "n_sigma_sensitivity": "NSIGMSEN",
    "target_name": "TARGETNA",
    "target_position": ["RA_OBJ", "DEC_OBJ"],
    "obs_ids": "OBS_IDS",
    "dataset_names": "DATASETS",
    "instrument": "INSTRU",
}

log = logging.getLogger(__name__)


class SEDTYPEEnum(str, Enum):
    dnde = "dnde"
    flux = "flux"
    eflux = "eflux"
    e2dnde = "e2dnde"
    likelihood = "likelihood"


class FluxMetaData(MetaData):
    """Metadata containing information about the FluxPoints and FluxMaps.

    Attributes
    ----------
    sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
        SED type.
    sed_type_init : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
        SED type of the initial data.
    ul_conf : float, optional
        Confidence level used for the upper limit computation.
    n_sigma : float, optional
        Significance threshold above which upper limits should be used.
    # n_sigma_ul : float, optional
    #     Sigma number used to compute the upper limits.
    sqrt_ts_threshold_ul : float, optional
        Threshold on the square root of the likelihood value above which upper limits should be used.
    n_sigma_sensitivity : float, optional
        Sigma number for which the flux sensitivity is computed
    target_name : str, optional
        Name of the target.
    target_position : `~astropy.coordinates.SkyCoord`, optional
        Coordinates of the target.
    obs_ids : list of str, optional
        ID list of the used observations.
    dataset_names : list of str, optional
        Name list of the used datasets.
    instrument : str, optional
        Name of the instrument.
    creation : `~gammapy.utils.CreatorMetaData`, optional
        The creation metadata.
    optional : dict, optional
        additional optional metadata.

    Note: these quantities are serialized in FITS header with the keywords stored in the dictionary FITS_META_KEYS
    """

    sed_type: Optional[SEDTYPEEnum]
    sed_type_init: Optional[SEDTYPEEnum]
    ul_conf: Optional[float]
    n_sigma: Optional[float]
    # n_sigma_ul: Optional[float]
    sqrt_ts_threshold_ul: Optional[float]
    n_sigma_sensitivity: Optional[float]
    target_name: Optional[str]
    target_position: Optional[SkyCoord]
    obs_ids: Optional[List[str]]
    dataset_names: Optional[List[str]]
    instrument: Optional[str]
    creation: Optional[CreatorMetaData]
    optional: Optional[dict]

    @validator("target_position")
    def validate_position(cls, v):
        if v is None:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="icrs")
        elif isinstance(v, SkyCoord):
            return v.transform_to("icrs")
        else:
            raise ValidationError(
                f"Incorrect position. Expect SkyCoord got {type(v)} instead."
            )

    # @validator("sed_type")
    # def validate_sed_type(cls, v):
    #     if isinstance(v, str):
    #         if v not in SEDTYPE:
    #             raise ValidationError(f"Incorrect [sed_type]. Expect {SEDTYPE}")
    #     return v

    # @validator("sed_type_init")
    # def validate_sed_type_init(cls, v):
    #     if isinstance(v, str):
    #         if v not in SEDTYPE:
    #             raise ValidationError(f"Incorrect [sed_type_init]. Expect {SEDTYPE}")
    #     return v

    @classmethod
    def from_default(cls):
        return cls(creation=CreatorMetaData.from_default())

    @classmethod
    def from_header(cls, data, format=None):
        """Extract metadata from a FITS header.

        Parameters
        ----------
        data : dict
            Dictionary containing data.
        format : str
            Header format. Not yet used.
        """

        creation = CreatorMetaData.from_header(data)
        if creation.date is None:
            creation = CreatorMetaData.from_default()
        meta = cls(creation=creation)

        for item in FITS_META_KEYS.items():
            if (
                item[0] == "target_position"
                and item[1][0] in data
                and item[1][1] in data
            ):
                meta.target_position = FluxMetaData._target_from_string(
                    data[item[1][0]], data[item[1][1]]
                )
            elif item[1] in data:
                val = data[item[1]]
                if item[0] == "obs_ids" or item[0] == "dataset_names":
                    val = val.split()
                setattr(meta, item[0], val)

        # if "NSIGMAUL" in data:
        #     meta.n_sigma_ul = float(data["NSIGMAUL"])
        #     if meta.ul_conf and meta.n_sigma_ul != np.round(
        #         stats.norm.isf(0.5 * (1 - meta.ul_conf)), 1
        #     ):
        #         log.warning(
        #             f"Inconsistency between n_sigma_ul={meta.n_sigma_ul} and ul_conf={meta.ul_conf}"
        #         )
        # elif meta.ul_conf:
        #     meta.n_sigma_ul = np.round(stats.norm.isf(0.5 * (1 - meta.ul_conf)), 1)

        for kk in data:
            if (
                kk not in FITS_META_KEYS.values()
                and kk not in CREATOR_META_KEYS.values()
                and kk not in FITS_META_KEYS["target_position"]
            ):
                if meta.optional is None:
                    meta.optional = {}
                meta.optional[str(kk).lower()] = data[kk]

        return meta

    def to_table(self, table):
        """Write the metadata into a table. Only the non-null information are stored.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Flux table.
        """

        for item in FITS_META_KEYS.items():
            if item[0] == "target_position" and np.isfinite(self.target_position.ra):
                val = FluxMetaData._target_to_string(self.target_position)
                table.meta[item[1][0]] = val[0]
                table.meta[item[1][1]] = val[1]
            elif item[0] == "obs_ids" or item[0] == "dataset_names":
                table.meta[item[1]] = " ".join(getattr(self, item[0]))
            elif getattr(self, item[0]):
                table.meta[item[1]] = str(getattr(self, item[0]))

        table.meta.update(self.creation.to_header())

        if self.optional:
            for k in self.optional:
                table.meta[str(k).upper()] = self.optional[k]

    @staticmethod
    def _target_from_string(v1, v2):
        if "nan" in [v1, v2]:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="icrs")
        return SkyCoord(v1, v2, unit="deg", frame="icrs")

    @staticmethod
    def _target_to_string(coord):
        return coord.to_string(precision=6).split()
