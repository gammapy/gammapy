from .core import Dataset
from gammapy.modeling.models import DatasetModels
from gammapy.data import EventList
from gammapy.utils.scripts import make_name
from gammapy.utils.fits import LazyFitsData
from gammapy.irf import PSFMap, HDULocation, EDispMap, EDispKernelMap


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
        geom=None,
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
    ):
        self._name = make_name(name)
        self._evaluators = {}

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

        if edisp and not isinstance(edisp, (EDispMap, EDispKernelMap, HDULocation)):
            raise ValueError(
                "'edisp' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp)}` instead."
            )

        self.edisp = edisp
        self.meta = meta

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
        self._models = models

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        if not isinstance(value, EventList):
            raise TypeError(
                f"'events' must ba an instance of `EventList`, got `{type(value)}` instead."
            )
        self._events = value
