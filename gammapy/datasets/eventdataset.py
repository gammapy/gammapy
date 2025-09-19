# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from gammapy.modeling.models import DatasetModels, FoVBackgroundModel
from gammapy.utils.scripts import make_name
from gammapy.utils.fits import LazyFitsData, HDULocation
from gammapy.irf import EDispMap, EDispKernelMap, PSFMap, RecoPSFMap
from gammapy.maps import Map, MapAxes
from gammapy.data import GTI
from .unbinned_evaluator import UnbinnedEvaluator
from .core import Dataset
from .map import BINSZ_IRF_DEFAULT, RAD_AXIS_DEFAULT, MIGRA_AXIS_DEFAULT

EVALUATION_MODE = "local"
USE_NPRED_CACHE = True


class EventDataset(Dataset):
    """ """

    stat_type = "unbinned"
    tag = "EventDataset"
    exposure = LazyFitsData(cache=True)
    edisp = LazyFitsData(cache=True)
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
        geom_normalization=None,
        models=None,
        exposure=None,
        psf=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        meta_table=None,
        name=None,
        reference_time="2000-01-01",
        gti=None,
        meta=None,
        edisp_e_reco_binned=None,
        # exposure_original_irf=None,
    ):
        self._name = make_name(name)
        self._evaluators = {}
        # self.position = position
        self.geom = geom
        self.geom_normalization = geom_normalization

        self.events = events
        self.exposure = exposure
        # self.background = background
        # self._background_cached = None
        # self._background_parameters_cached = None

        self.mask_fit = mask_fit
        self.mask_safe = mask_safe

        self.reference_time = reference_time
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
        if edisp_e_reco_binned is not None and not isinstance(
            edisp_e_reco_binned, (EDispMap, EDispKernelMap, HDULocation)
        ):
            raise ValueError(
                "'edisp_e_reco_binned' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp_e_reco_binned)}` instead."
            )

        self.edisp = edisp
        self.meta = meta
        self.edisp_e_reco_binned = edisp_e_reco_binned
        # self.exposure_original_irf = exposure_original_irf

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
                        edisp_e_reco_binned=self.edisp_e_reco_binned,
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
    def create(
        cls,
        geom,
        energy_axis_true=None,
        migra_axis=None,
        rad_axis=None,
        binsz_irf=BINSZ_IRF_DEFAULT,
        reco_psf=False,
        reference_time="2000-01-01",
        name=None,
        meta_table=None,
        **kwargs,
    ):
        """Create an empty event dataset."""
        geoms = create_event_dataset_geoms(
            geom=geom,
            energy_axis_true=energy_axis_true,
            migra_axis=migra_axis,
            rad_axis=rad_axis,
            binsz_irf=binsz_irf,
            reco_psf=reco_psf,
        )
        kwargs.update(geoms)
        return cls.from_geoms(
            name=name, reference_time=reference_time, meta_table=meta_table, **kwargs
        )

    @classmethod
    def from_geoms(
        cls,
        geom,
        geom_exposure=None,
        geom_psf=None,
        geom_edisp=None,
        reference_time="2000-01-01",
        name=None,
        **kwargs,
    ):
        name = make_name(name)
        kwargs = kwargs.copy()
        kwargs["name"] = name

        if geom_exposure:
            kwargs["exposure"] = Map.from_geom(geom_exposure, unit="m2 s")

        if geom_edisp:
            if "energy" in geom_edisp.axes.names:
                kwargs["edisp"] = EDispKernelMap.from_geom(geom_edisp)
            else:
                kwargs["edisp"] = EDispMap.from_geom(geom_edisp)

        if geom_psf:
            if "energy_true" in geom_psf.axes.names:
                kwargs["psf"] = PSFMap.from_geom(geom_psf)
            elif "energy" in geom_psf.axes.names:
                kwargs["psf"] = RecoPSFMap.from_geom(geom_psf)

        kwargs.setdefault(
            "gti", GTI.create([] * u.s, [] * u.s, reference_time=reference_time)
        )
        kwargs["mask_safe"] = Map.from_geom(geom, unit="", dtype=bool)
        return cls(geom=geom, **kwargs)


def create_event_dataset_geoms(
    geom,
    energy_axis_true=None,
    migra_axis=None,
    rad_axis=None,
    binsz_irf=BINSZ_IRF_DEFAULT,
    reco_psf=False,
):
    """Create geometries needed for event dataset.
    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom` or `~gammapy.maps.RegionGeom`
        Reference geometry.
    energy_axis_true : `~gammapy.maps.MapAxis`
        True energy axis.
    migra_axis : `~gammapy.maps.MapAxis`
        Migration axis.
    rad_axis : `~gammapy.maps.MapAxis`
        Offset axis.
    binsz_irf : float
        Bin size for IRF maps in deg.
    reco_psf : bool
        Use reconstructed energy axis for PSF map.
    Returns
    -------
    dict
        Dictionary of geometries.
    """
    rad_axis = rad_axis or RAD_AXIS_DEFAULT
    migra_axis = migra_axis or MIGRA_AXIS_DEFAULT

    if energy_axis_true is not None:
        energy_axis_true.assert_name("energy_true")
    else:
        energy_axis_true = geom.axes["energy_true"].copy(name="energy_true")

    external_axes = geom.axes.drop("energy_true")
    geom_image = geom.to_image()
    geom_exposure = geom_image.to_cube(MapAxes([energy_axis_true]) + external_axes)
    geom_irf = geom_image.to_binsz(binsz=binsz_irf)

    if reco_psf:
        raise NotImplementedError(
            "PSF map with reco energy axis not implemented yet for event dataset."
        )
    geom_psf = geom_irf.to_cube(MapAxes([rad_axis, energy_axis_true]) + external_axes)
    geom_edisp = geom_irf.to_cube(
        MapAxes([migra_axis, energy_axis_true]) + external_axes
    )
    return {
        "geom": geom,
        "geom_exposure": geom_exposure,
        "geom_psf": geom_psf,
        "geom_edisp": geom_edisp,
    }
