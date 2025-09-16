# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel
from gammapy.utils.scripts import make_name
from gammapy.utils.fits import LazyFitsData, HDULocation
from gammapy.irf import PSFMap, EDispMap, EDispKernelMap
from .unbinned_evaluator import UnbinnedEvaluator
from .core import Dataset

EVALUATION_MODE = "local"
USE_NPRED_CACHE = True


class EventDataset(Dataset):
    """ """

    stat_type = "unbinned"
    tag = "EventDataset"
    exposure = LazyFitsData(cache=True)
    edisp = LazyFitsData(cache=True)
    background = LazyFitsData(cache=True)
    psf = LazyFitsData(cache=True)
    mask_fit = LazyFitsData(cache=True)
    mask_safe = LazyFitsData(cache=True)

    _lazy_data_members = [
        "background",
        "exposure",
        "edisp",
        "psf",
        "mask_fit",
        "mask_safe",
    ]

    gti = None
    meta_table = None

    def __init__(
        self,
        events=None,
        position=None,
        geom=None,
        geom_normalization=None,
        models=None,
        exposure=None,
        psf=None,
        edisp=None,
        background=None,
        mask_safe=None,
        mask_fit=None,
        meta_table=None,
        name=None,
        gti=None,
        meta=None,
        edisp_original_irf=None,
        exposure_original_irf=None,
    ):
        self._name = make_name(name)
        self._evaluators = {}
        self.position = position
        self.geom = geom
        self.geom_normalization = geom_normalization

        self.events = events
        self.exposure = exposure
        self.background = background
        self._background_cached = None
        self._background_parameters_cached = None

        self.mask_fit = mask_fit
        self.mask_safe = mask_safe

        self.gti = gti
        self.models = models
        self.meta_table = meta_table

        if psf and not isinstance(psf, (PSFMap, HDULocation)):
            raise ValueError(
                f"'psf' must be a 'PSFMap' or `HDULocation` object, got {type(psf)}"
            )
        self.psf = psf

        if edisp is not None and not isinstance(
            edisp, (EDispMap, EDispKernelMap, HDULocation)
        ):
            raise ValueError(
                "'edisp' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp)}` instead."
            )
        if edisp_original_irf is not None and not isinstance(
            edisp_original_irf, (EDispMap, EDispKernelMap, HDULocation)
        ):
            raise ValueError(
                "'edisp_original_irf' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp_original_irf)}` instead."
            )

        self.edisp = edisp
        self.meta = meta
        self.edisp_original_irf = edisp_original_irf
        self.exposure_original_irf = exposure_original_irf

    @property
    def _geom(self):
        """Main analysis geometry."""
        return self.geom

    def __str__(self):
        pass

    @property
    def evaluators(self):
        """Model evaluators."""
        return self._evaluators

    @property
    def models(self):
        """Models set on the dataset (`~gammapy.modeling.models.Models`)."""
        return self._models

    @models.setter
    def models(self, models):
        """Models setter."""
        self._evaluators = {}
        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)
            for model in models:
                if not isinstance(model, FoVBackgroundModel):
                    evaluator = UnbinnedEvaluator(
                        model=model,
                        geom=self.geom,
                        geom_normalization=self.geom_normalization,
                        psf=self.psf,
                        edisp=self.edisp,
                        edisp_original_irf=self.edisp_original_irf,
                        exposure=self.exposure,
                        exposure_original_irf=self.exposure_original_irf,
                        evaluation_mode=EVALUATION_MODE,
                        gti=self.gti,
                        use_cache=USE_NPRED_CACHE,
                    )
                    self._evaluators[model.name] = evaluator
        self._models = models

    @property
    def events(self):
        return self._events.select_row_subset(self.event_mask)

    @events.setter
    def events(self, value):
        # if not isinstance(value, EventList):
        #    raise TypeError(
        #        f"'events' must ba an instance of `EventList`, got `{type(value)}` instead."
        #    )
        self._events = value

    @property
    def mask_event(self):
        """Entry for each event whether it is inside the mask or not"""
        if self.mask is None:
            return np.ones(len(self.events.table), dtype=bool)
        coords = self.events.map_coord(self.mask.geom)
        return self.mask.get_by_coord(coords) == 1

    def info_dict(self):
        pass

    def stat_array(self):
        pass

    @classmethod
    def create(cls, position=None, name=None):
        """Create an empty event dataset."""
        return cls(
            name=name,
            position=position,
        )
