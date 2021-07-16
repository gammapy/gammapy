# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.utils import classproperty
from gammapy.data import GTI
from gammapy.maps import MapCoord, Map
from gammapy.estimators.core import FluxEstimate, OPTIONAL_QUANTITIES_COMMON, REQUIRED_MAPS, OPTIONAL_QUANTITIES
from gammapy.estimators.flux_point import FluxPoints
from gammapy.modeling.models import SkyModel, Models
from gammapy.utils.scripts import make_path

__all__ = ["FluxMaps"]


log = logging.getLogger(__name__)


class FluxMaps(FluxEstimate):
    """A flux map container.

    It contains a set of `~gammapy.maps.Map` objects that store the estimated flux as a function of energy as well as
    associated quantities (typically errors, upper limits, delta TS and possibly raw quantities such counts,
    excesses etc). It also contains a reference model to convert the flux values in different formats. Usually, this
    should be the model used to produce the flux map.

    The associated map geometry can use a `RegionGeom` to store the equivalent of flux points, or a `WcsGeom`/`HpxGeom`
    to store an energy dependent flux map.

    The container relies internally on the 'Likelihood' SED type defined in :ref:`gadf:flux-points`
    and offers convenience properties to convert to other flux formats, namely:
    ``dnde``, ``flux``, ``eflux`` or ``e2dnde``. The conversion is done according to the reference model spectral shape.

    Parameters
    ----------
    data : dict of `~gammapy.maps.Map`
        the maps dictionary. Expected entries are the following:
        * norm : the norm factor
        * norm_err : optional, the error on the norm factor.
        * norm_errn : optional, the negative error on the norm factor.
        * norm_errp : optional, the positive error on the norm factor.
        * norm_ul : optional, the upper limit on the norm factor.
        * norm_scan : optional, the norm values of the test statistic scan.
        * stat_scan : optional, the test statistic scan values.
        * ts : optional, the delta TS associated with the flux value.
        * sqrt_ts : optional, the square root of the TS, when relevant.
    reference_model : `~gammapy.modeling.models.SkyModel`, optional
        the reference model to use for conversions. Default in None.
        If None, a model consisting of a point source with a power law spectrum of index 2 is assumed.
    gti : `~gammapy.data.GTI`
        the maps GTI information. Default is None.
    """

    def __init__(self, data, reference_model, gti=None, meta=None):
        self.reference_model = reference_model

        super().__init__(
            data=data,
            reference_spectral_model=reference_model.spectral_model,
            meta=meta,
            gti=gti
        )

    @classproperty
    def reference_model_default(cls):
        """Default reference model: a point source with index = 2  (`SkyModel`)"""
        return SkyModel.create("pl", "point")

