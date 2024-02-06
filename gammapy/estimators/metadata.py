# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from enum import Enum
from typing import ClassVar, Literal, Optional
from gammapy.utils.metadata import (
    METADATA_FITS_KEYS,
    CreatorMetaData,
    MetaData,
    TargetMetaData,
)

__all__ = ["FluxMetaData"]


FLUX_METADATA_FITS_KEYS = {
    "flux": {
        "sed_type": "SED_TYPE",
        "sed_type_init": "SEDTYPEI",
        "n_sigma": "N_SIGMA",
        "n_sigma_ul": "NSIGMAUL",
        "sqrt_ts_threshold_ul": "STSTHUL",
        "n_sigma_sensitivity": "NSIGMSEN",
    },
}

METADATA_FITS_KEYS.update(FLUX_METADATA_FITS_KEYS)

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
    n_sigma : float, optional
        Significance threshold above which upper limits should be used.
    n_sigma_ul : float, optional
        Significance value used for the upper limit computation.
    sqrt_ts_threshold_ul : float, optional
        Threshold on the square root of the likelihood value above which upper limits should be used.
    n_sigma_sensitivity : float, optional
        Sigma number for which the flux sensitivity is computed
    target : `~gammapy.utils.TargetMetaData`, optional
        General metadata information about the target.
    creation : `~gammapy.utils.CreatorMetaData`, optional
        The creation metadata.
    optional : dict, optional
        additional optional metadata.

    Note : these quantities are serialized in FITS header with the keywords stored in the dictionary FLUX_METADATA_FITS_KEYS
    """

    _tag: ClassVar[Literal["flux"]] = "flux"
    sed_type: Optional[SEDTYPEEnum] = None
    sed_type_init: Optional[SEDTYPEEnum] = None
    n_sigma: Optional[float] = None
    n_sigma_ul: Optional[float] = None
    sqrt_ts_threshold_ul: Optional[float] = None
    n_sigma_sensitivity: Optional[float] = None
    target: Optional[TargetMetaData] = None
    # TODO: add obs_ids: Optional[List[int]] and instrument: Optional[str], or a List[ObsInfoMetaData]
    # TODO : add dataset_names: Optional[List[str]]
    creation: Optional[CreatorMetaData] = CreatorMetaData()
    optional: Optional[dict] = None
