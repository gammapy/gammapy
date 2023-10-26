# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from typing import Optional
import numpy as np
from scipy import stats
from astropy.coordinates import SkyCoord
from pydantic import ValidationError, validator
from gammapy.utils.metadata import CreatorMetaData, MetaData

__all__ = ["FluxMetaData"]

SEDTYPE = ["dnde", "flux", "eflux", "e2dnde", "likelihood"]
FPFORMAT = ["gadf-sed", "lightcurve"]
STANDARD_KEYS = [
    "SED_TYPE",
    "SEDTYPEI",
    "UL_CONF",
    "N_SIGMA",
    "NSIGMAUL",
    "STSTHUL",
    "NSIGMSEN",
    "TARGETNA",
    "TARGETPO",
    "OBS_IDS",
    "DATASETS",
    "INSTRU",
]

log = logging.getLogger(__name__)


class FluxMetaData(MetaData):
    """Metadata containing information about the FluxPoints and FluxMaps.

    Parameters
    ----------
    sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
        SED type.
    sed_type_init : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
        SED type of the initial data.
    ul_conf : float, optional
        Confidence level used for the upper limit computation.
    n_sigma : float, optional
        Significance threshold above which upper limits should be used.
    n_sigma_ul : float, optional
        Sigma number used to compute the upper limits.
    sqrt_ts_threshold_ul : float, optional
        Threshold on the square root of the likelihood value above which upper limits should be used.
    n_sigma_sensitivity : float, optional
        Sigma number for which the flux sensitivity is computed
    # gti : `~gammapy.data.gti`, optional
    #     used Good Time Intervals.
    target_name : str, optional
        Name of the target.
    target_position : `~astropy.coordinates.SkyCoord`, optional
        Coordinates of the target.
    obs_ids : list of int, optional
        ID list of the used observations.
    dataset_names : list of str, optional
        Name list of the used datasets.
    instrument : str, optional
        Name of the instrument.
    creation : `~gammapy.utils.CreatorMetaData`, optional
        The creation metadata.
    optional : dict, optional
        additional optional metadata.

    Note: these quantities are serialized in FITS header with the keywords stored in STANDARD_KEYS
    """

    sed_type: Optional[str]  # Are these 6 fields really optional?
    sed_type_init: Optional[str]
    ul_conf: Optional[float]
    n_sigma: Optional[float]
    n_sigma_ul: Optional[float]
    sqrt_ts_threshold_ul: Optional[float]
    n_sigma_sensitivity: Optional[float]
    # gti: Optional[GTI]
    target_name: Optional[str]  # Are these 2 fields really optional?
    target_position: Optional[SkyCoord]
    obs_ids: Optional[list[int]]
    dataset_names: Optional[list[str]]  # Are these 2 fields really optional?
    instrument: Optional[str]
    creation: Optional[CreatorMetaData]  # Is this field really optional?
    optional: Optional[dict]

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

    @validator("sed_type")
    def validate_sed_type(cls, v):
        if isinstance(v, str):
            if v not in SEDTYPE:
                raise ValidationError(f"Incorrect [sed_type]. Expect {SEDTYPE}")
        return v

    @validator("sed_type_init")
    def validate_sed_type_init(cls, v):
        if isinstance(v, str):
            if v not in SEDTYPE:
                raise ValidationError(f"Incorrect [sed_type_init]. Expect {SEDTYPE}")
        return v

    @validator("obs_ids")
    def validate_obs_ids(cls, v):
        if isinstance(v, list) and not all(np.isfinite(_) for _ in v):
            raise ValidationError("Incorrect [obs_ids]. Expect [int]")
        return v

    @validator("creation")
    def validate_creation(cls, v):
        if v is None:
            raise ValidationError(
                f"[creation] should be precised. Expect {type(CreatorMetaData)}."
            )
        elif isinstance(v, CreatorMetaData):
            return v
        else:
            raise ValidationError(
                f"Incorrect [creation]. Expect 'CreatorMetaData' got {type(v)} instead."
            )

    @classmethod
    def from_default(cls):
        return cls(
            # creation=CreatorMetaData.from_default(), target_position=None, gti=None
            creation=CreatorMetaData.from_default()
        )

    # def to_header(self, format=None):
    #     """Store the FluxPoints metadata into a fits header.
    #
    #     Parameters
    #     ----------
    #     format : {"gadf-sed", "lightcurve"}
    #         The header data format.
    #
    #     Returns
    #     -------
    #     header : dict
    #         The header dictionary.
    #     """
    #
    #     if format is None or format not in FPFORMAT:
    #         raise ValueError(
    #             f"Metadata creation with format {format} is not supported. Use {FPFORMAT}"
    #         )
    #
    #     hdr_dict = self.creation.to_header()
    #     hdr_dict["SED_TYPE"] = self.sed_type
    #     hdr_dict["SEDTYPEI"] = self.sed_type_init
    #     if np.isfinite(self.ul_conf):
    #         hdr_dict["UL_CONF"] = self.ul_conf
    #     if np.isfinite(self.n_sigma):
    #         hdr_dict["N_SIGMA"] = self.n_sigma
    #     if np.isfinite(self.n_sigma_ul):
    #         hdr_dict["NSIGMAUL"] = self.n_sigma_ul
    #     if np.isfinite(self.sqrt_ts_threshold_ul):
    #         hdr_dict["STSTHUL"] = self.sqrt_ts_threshold_ul
    #     # hdr_dict["GTI"] = self.gti #They should be written in a HDU, in in the header
    #     if self.target_name is not None:
    #         hdr_dict["TARGETNA"] = self.target_name
    #     if np.isfinite(self.target_position.ra):
    #         hdr_dict["TARGETPO"] = self._target_to_string(self.target_position)
    #     if np.isfinite(self.obs_ids[0]):
    #         hdr_dict["OBS_IDS"] = self.obs_ids
    #     if self.dataset_names[0] is not None:
    #         hdr_dict["DATASETS"] = self.dataset_names
    #     if self.instrument is not None:
    #         hdr_dict["INSTRU"] = self.instrument
    #     # Do not forget that we have optional metadata
    #
    #     if self.optional:
    #         for k in self.optional:
    #             hdr_dict[str(k).upper()] = self.optional[k]
    #
    #     return hdr_dict

    # @classmethod
    # def from_header(cls, hdr, format="gadf"):
    #     """Builds creator metadata from fits header.
    #     Parameters
    #     ----------
    #     hdr : dict
    #         the header dictionary
    #     format : str
    #         header format. Default is 'gadf'.
    #     """
    #     if format != "gadf":
    #         raise ValueError(f"Creator metadata: format {format} is not supported.")
    #
    #     from_header(hdr, format)

    @classmethod
    def from_dict(cls, data):
        """Extract metadata from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing data.
        """

        sed_type = data["SED_TYPE"] if "SED_TYPE" in data else None
        sed_type_init = data["SEDTYPEI"] if "SEDTYPEI" in data else None
        creation = CreatorMetaData.from_dict(data)
        meta = cls(sed_type=sed_type, sed_type_init=sed_type_init, creation=creation)

        if "UL_CONF" in data:
            meta.ul_conf = float(data["UL_CONF"])
        if "N_SIGMA" in data:
            meta.n_sigma = float(data["N_SIGMA"])

        if "NSIGMAUL" in data:
            meta.n_sigma_ul = float(data["NSIGMAUL"])
            if meta.ul_conf and meta.n_sigma_ul != np.round(
                stats.norm.isf(0.5 * (1 - meta.ul_conf)), 1
            ):
                log.warning(
                    f"Inconsistency between n_sigma_ul={meta.n_sigma_ul} and ul_conf={meta.ul_conf}"
                )
        elif meta.ul_conf:
            meta.n_sigma_ul = np.round(stats.norm.isf(0.5 * (1 - meta.ul_conf)), 1)

        if "STSTHUL" in data:
            meta.sqrt_ts_threshold_ul = float(data["STSTHUL"])
        if "NSIGMSEN" in data:
            meta.n_sigma_sensitivity = float(data["NSIGMSEN"])
        if "TARGETNA" in data:
            meta.target_name = data["TARGETNA"]
        if "OBS_IDS" in data:
            meta.obs_ids = FluxMetaData._obsids_from_string(data["OBS_IDS"])
        if "DATASETS" in data:
            meta.dataset_names = data["DATASETS"]
        if "INSTRU" in data:
            meta.instrument = data["INSTRU"]
        if "TARGETPO" in data:
            meta.target_position = FluxMetaData._target_from_string(data["TARGETPO"])

        for kk in data:
            if kk not in STANDARD_KEYS:
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

        if self.sed_type:
            table.meta["SED_TYPE"] = self.sed_type
        if self.sed_type_init:
            table.meta["SEDTYPEI"] = self.sed_type_init
        if self.ul_conf:
            table.meta["UL_CONF"] = str(self.ul_conf)
        if self.n_sigma:
            table.meta["N_SIGMA"] = str(self.n_sigma)
        if self.n_sigma_ul:
            table.meta["NSIGMAUL"] = str(self.n_sigma_ul)
        if self.sqrt_ts_threshold_ul:
            table.meta["STSTHUL"] = str(self.sqrt_ts_threshold_ul)
        if self.n_sigma_sensitivity:
            table.meta["NSIGMSEN"] = str(self.n_sigma_sensitivity)
        if self.target_name:
            table.meta["TARGETNA"] = self.target_name
        if self.obs_ids:
            table.meta["OBS_IDS"] = FluxMetaData._obsids_to_string(self.obs_ids)
        if self.dataset_names:
            table.meta["DATASETS"] = self.dataset_names
        if self.instrument:
            table.meta["INSTRU"] = self.instrument
        if self.target_position and np.isfinite(self.target_position.ra):
            table.meta["TARGETPO"] = FluxMetaData._target_to_string(
                self.target_position
            )
        # table.meta["GTI"] = self.gti #There are stored in a dedicated HDU

        self.creation.to_table(table)

        if self.optional:
            for k in self.optional:
                table.meta[str(k).upper()] = self.optional[k]

    @staticmethod
    def _target_from_string(data):
        values = data.split()
        if "nan" in values:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="icrs")
        return SkyCoord(values[0], values[1], unit="deg", frame="icrs")

    @staticmethod
    def _target_to_string(coord):
        return coord.transform_to("icrs").to_string(precision=6)

    @staticmethod
    def _obsids_from_string(data):
        values = data.split()
        return [int(_) for _ in values]

    @staticmethod
    def _obsids_to_string(values):
        out = ""
        for _ in values:
            out += str(_) + " "
        return out
