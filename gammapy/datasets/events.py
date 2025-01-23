from .core import Dataset
from gammapy.utils.scripts import make_name


class EventsDatasets(Dataset):
    """ """

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
        events,
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

        self.psf = psf
        self.edisp = edisp

        self.meta = meta
