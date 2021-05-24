# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from functools import lru_cache
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.utils import lazyproperty
from regions import CircleSkyRegion
from gammapy.data import GTI
from gammapy.irf import EDispKernelMap, EDispMap, PSFKernel, PSFMap
from gammapy.maps import Map, MapAxis, RegionGeom, WcsGeom
from gammapy.modeling.models import (
    BackgroundModel,
    DatasetModels,
    FoVBackgroundModel,
    PointSpatialModel
)
from gammapy.stats import (
    CashCountsStatistic,
    WStatCountsStatistic,
    cash,
    cash_sum_cython,
    get_wstat_mu_bkg,
    wstat,
)
from gammapy.utils.fits import HDULocation, LazyFitsData
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name, make_path
from gammapy.utils.table import hstack_columns
from .core import Dataset
from .utils import get_axes

__all__ = ["MapDataset", "MapDatasetOnOff", "create_map_dataset_geoms"]

log = logging.getLogger(__name__)

CUTOUT_MARGIN = 0.1 * u.deg
RAD_MAX = 0.66
RAD_AXIS_DEFAULT = MapAxis.from_bounds(
    0, RAD_MAX, nbin=66, node_type="edges", name="rad", unit="deg"
)
MIGRA_AXIS_DEFAULT = MapAxis.from_bounds(
    0.2, 5, nbin=48, node_type="edges", name="migra"
)

BINSZ_IRF_DEFAULT = 0.2

EVALUATION_MODE = "local"
USE_NPRED_CACHE = True


def create_map_dataset_geoms(
    geom, energy_axis_true=None, migra_axis=None, rad_axis=None, binsz_irf=None,
):
    """Create map geometries for a `MapDataset`

    Parameters
    ----------
    geom : `~gammapy.maps.WcsGeom`
        Reference target geometry in reco energy, used for counts and background maps
    energy_axis_true : `~gammapy.maps.MapAxis`
        True energy axis used for IRF maps
    migra_axis : `~gammapy.maps.MapAxis`
        If set, this provides the migration axis for the energy dispersion map.
        If not set, an EDispKernelMap is produced instead. Default is None
    rad_axis : `~gammapy.maps.MapAxis`
        Rad axis for the psf map
    binsz_irf : float
        IRF Map pixel size in degrees.

    Returns
    -------
    geoms : dict
        Dict with map geometries.
    """
    rad_axis = rad_axis or RAD_AXIS_DEFAULT

    if energy_axis_true is not None:
        energy_axis_true.assert_name("energy_true")
    else:
        energy_axis_true = geom.axes["energy"].copy(name="energy_true")

    binsz_irf = binsz_irf or BINSZ_IRF_DEFAULT
    geom_image = geom.to_image()
    geom_exposure = geom_image.to_cube([energy_axis_true])
    geom_irf = geom_image.to_binsz(binsz=binsz_irf)
    geom_psf = geom_irf.to_cube([rad_axis, energy_axis_true])

    if migra_axis:
        geom_edisp = geom_irf.to_cube([migra_axis, energy_axis_true])
    else:
        geom_edisp = geom_irf.to_cube([geom.axes["energy"], energy_axis_true])

    return {
        "geom": geom,
        "geom_exposure": geom_exposure,
        "geom_psf": geom_psf,
        "geom_edisp": geom_edisp,
    }


class MapDataset(Dataset):
    """Perform sky model likelihood fit on maps.

    If an `HDULocation` is passed the map is loaded lazily. This means the
    map data is only loaded in memeory as the corresponding data attribute
    on the MapDataset is accessed. If it was accesed once it is cached for
    the next time.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Exposure cube
    background : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Background cube
    mask_fit : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.irf.PSFMap` or `~gammapy.utils.fits.HDULocation`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel` or `~gammapy.irf.EDispMap` or `~gammapy.utils.fits.HDULocation`
        Energy dispersion kernel
    mask_safe : `~gammapy.maps.WcsNDMap` or `~gammapy.utils.fits.HDULocation`
        Mask defining the safe data range.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    meta_table : `~astropy.table.Table`
        Table listing informations on observations used to create the dataset.
        One line per observation for stacked datasets.


    See Also
    --------
    MapDatasetOnOff, SpectrumDataset, FluxPointsDataset
    """

    stat_type = "cash"
    tag = "MapDataset"
    counts = LazyFitsData(cache=True)
    exposure = LazyFitsData(cache=True)
    edisp = LazyFitsData(cache=True)
    background = LazyFitsData(cache=True)
    psf = LazyFitsData(cache=True)
    mask_fit = LazyFitsData(cache=True)
    mask_safe = LazyFitsData(cache=True)

    _lazy_data_members = [
        "counts",
        "exposure",
        "edisp",
        "psf",
        "mask_fit",
        "mask_safe",
        "background",
    ]

    def __init__(
        self,
        models=None,
        counts=None,
        exposure=None,
        background=None,
        psf=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        gti=None,
        meta_table=None,
        name=None,
    ):
        self._name = make_name(name)
        self._evaluators = {}

        self.counts = counts
        self.exposure = exposure
        self.background = background
        self.mask_fit = mask_fit

        if psf and not isinstance(psf, (PSFMap, HDULocation)):
            raise ValueError(
                f"'psf' must be a 'PSFMap' or `HDULocation` object, got {type(psf)}"
            )

        self.psf = psf

        if edisp and not isinstance(edisp, (EDispMap, EDispKernelMap, HDULocation)):
            raise ValueError(
                f"'edisp' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' object, got {type(edisp)}"
            )

        self.edisp = edisp
        self.mask_safe = mask_safe
        self.gti = gti
        self.models = models
        self.meta_table = meta_table

    # TODO: keep or remove?
    @property
    def background_model(self):
        try:
            return self.models[f"{self.name}-bkg"]
        except (ValueError, TypeError):
            pass

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"
        str_ += "\t{:32}: {{name}} \n\n".format("Name")
        str_ += "\t{:32}: {{counts:.0f}} \n".format("Total counts")
        str_ += "\t{:32}: {{background:.2f}}\n".format("Total background counts")
        str_ += "\t{:32}: {{excess:.2f}}\n\n".format("Total excess counts")

        str_ += "\t{:32}: {{npred:.2f}}\n".format("Predicted counts")
        str_ += "\t{:32}: {{npred_background:.2f}}\n".format(
            "Predicted background counts"
        )
        str_ += "\t{:32}: {{npred_signal:.2f}}\n\n".format("Predicted excess counts")

        str_ += "\t{:32}: {{exposure_min:.2e}}\n".format("Exposure min")
        str_ += "\t{:32}: {{exposure_max:.2e}}\n\n".format("Exposure max")

        str_ += "\t{:32}: {{n_bins}} \n".format("Number of total bins")
        str_ += "\t{:32}: {{n_fit_bins}} \n\n".format("Number of fit bins")

        # likelihood section
        str_ += "\t{:32}: {{stat_type}}\n".format("Fit statistic type")
        str_ += "\t{:32}: {{stat_sum:.2f}}\n\n".format(
            "Fit statistic value (-2 log(L))"
        )

        info = self.info_dict()
        str_ = str_.format(**info)

        # model section
        n_models, n_pars, n_free_pars = 0, 0, 0
        if self.models is not None:
            n_models = len(self.models)
            n_pars = len(self.models.parameters)
            n_free_pars = len(self.models.parameters.free_parameters)

        str_ += "\t{:32}: {} \n".format("Number of models", n_models)
        str_ += "\t{:32}: {}\n".format("Number of parameters", n_pars)
        str_ += "\t{:32}: {}\n\n".format("Number of free parameters", n_free_pars)

        if self.models is not None:
            str_ += "\t" + "\n\t".join(str(self.models).split("\n")[2:])

        return str_.expandtabs(tabsize=2)

    @property
    def geoms(self):
        """Map geometries

        Returns
        -------
        geoms : dict
            Dict of map geometries involved in the dataset.
        """
        geoms = {}

        geoms["geom"] = self._geom

        if self.exposure:
            geoms["geom_exposure"] = self.exposure.geom

        if self.psf:
            geoms["geom_psf"] = self.psf.psf_map.geom

        if self.edisp:
            geoms["geom_edisp"] = self.edisp.edisp_map.geom

        return geoms

    @property
    def models(self):
        """Models (`~gammapy.modeling.models.Models`)."""
        return self._models

    @property
    def excess(self):
        """Excess"""
        return self.counts - self.background

    @models.setter
    def models(self, models):
        """Models setter"""
        self._evaluators = {}

        if models is not None:
            models = DatasetModels(models)
            models = models.select(datasets_names=self.name)

            for model in models:
                if not isinstance(model, FoVBackgroundModel):
                    evaluator = MapEvaluator(
                        model=model,
                        evaluation_mode=EVALUATION_MODE,
                        gti=self.gti,
                        use_cache=USE_NPRED_CACHE,
                    )
                    self._evaluators[model.name] = evaluator

        self._models = models

    @property
    def evaluators(self):
        """Model evaluators"""
        return self._evaluators

    @property
    def _geom(self):
        """Main analysis geometry"""
        if self.counts is not None:
            return self.counts.geom
        elif self.background is not None:
            return self.background.geom
        elif self.mask_safe is not None:
            return self.mask_safe.geom
        elif self.mask_fit is not None:
            return self.mask_fit.geom
        else:
            raise ValueError(
                "Either 'counts', 'background', 'mask_fit'"
                " or 'mask_safe' must be defined."
            )

    @property
    def data_shape(self):
        """Shape of the counts or background data (tuple)"""
        return self._geom.data_shape

    # TODO: make this support different methods?
    def energy_range(self, region=None):
        """Energy range of the region in the safe mask.

        By default, the whole dataset map region is considered.

        Parameters
        ----------
        region : `~regions.Region` or `~astropy.coordinates.SkyCoord`
            Region for extraction.

        Returns
        -------
        energy_range : `~astropy.units.Quantity`
            The safe energy range.
        """
        energy = self._geom.axes["energy"].edges
        energy_min, energy_max = energy[:-1], energy[1:]

        if self.mask_safe is not None:
            if self.mask_safe.data.any():
                mask = self.mask_safe.get_spectrum(region, np.any).data[:, 0, 0]
                energy_min, energy_max = energy_min[mask], energy_max[mask]
            else:
                return None, None

        return u.Quantity([energy_min[0], energy_max[-1]])

    def npred(self):
        """Predicted source and background counts

        Returns
        -------
        npred : `Map`
            Total predicted counts
        """
        npred_total = self.npred_signal()

        if self.background:
            npred_total += self.npred_background()

        return npred_total

    def npred_background(self):
        """Predicted background counts

        The predicted background counts depend on the parameters
        of the `FoVBackgroundModel` defined in the dataset.

        Returns
        -------
        npred_background : `Map`
            Predicted counts from the background.
        """
        background = self.background

        if self.background_model and background:
            values = self.background_model.evaluate_geom(geom=self.background.geom)
            background = background * values

        return background

    def npred_signal(self, model=None):
        """"Model predicted signal counts.

        If a model is passed, predicted counts from that component is returned.
        Else, the total signal counts are returned.

        Parameters
        -------------
        model: `~gammapy.modeling.models.SkyModel`, optional
            Sky model to compute the npred for.
            If none, the sum of all components (minus the background model)
            is returned

        Returns
        ----------
        npred_sig: `gammapy.maps.Map`
            Map of the predicted signal counts
        """
        npred_total = Map.from_geom(self._geom, dtype=float)

        for evaluator in self.evaluators.values():
            if model is evaluator.model:
                return evaluator.compute_npred()

            if evaluator.needs_update:
                evaluator.update(
                    self.exposure,
                    self.psf,
                    self.edisp,
                    self._geom,
                    self.mask_image,
                )

            if evaluator.contributes:
                npred = evaluator.compute_npred()
                npred_total.stack(npred)

        return npred_total

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
        """
        Create a MapDataset object with zero filled maps according to the specified geometries

        Parameters
        ----------
        geom : `Geom`
            geometry for the counts and background maps
        geom_exposure : `Geom`
            geometry for the exposure map
        geom_psf : `Geom`
            geometry for the psf map
        geom_edisp : `Geom`
            geometry for the energy dispersion kernel map.
            If geom_edisp has a migra axis, this wil create an EDispMap instead.
        reference_time : `~astropy.time.Time`
            the reference time to use in GTI definition
        name : str
            Name of the returned dataset.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            A dataset containing zero filled maps
        """
        name = make_name(name)
        kwargs = kwargs.copy()
        kwargs["name"] = name
        kwargs["counts"] = Map.from_geom(geom, unit="")
        kwargs["background"] = Map.from_geom(geom, unit="")

        if geom_exposure:
            kwargs["exposure"] = Map.from_geom(geom_exposure, unit="m2 s")

        if geom_edisp:
            if "energy" in geom_edisp.axes.names:
                kwargs["edisp"] = EDispKernelMap.from_geom(geom_edisp)
            else:
                kwargs["edisp"] = EDispMap.from_geom(geom_edisp)

        if geom_psf:
            kwargs["psf"] = PSFMap.from_geom(geom_psf)

        kwargs.setdefault(
            "gti", GTI.create([] * u.s, [] * u.s, reference_time=reference_time)
        )
        kwargs["mask_safe"] = Map.from_geom(geom, unit="", dtype=bool)
        return cls(**kwargs)

    @classmethod
    def create(
        cls,
        geom,
        energy_axis_true=None,
        migra_axis=None,
        rad_axis=None,
        binsz_irf=None,
        reference_time="2000-01-01",
        name=None,
        meta_table=None,
        **kwargs,
    ):
        """Create a MapDataset object with zero filled maps.

        Parameters
        ----------
        geom : `~gammapy.maps.WcsGeom`
            Reference target geometry in reco energy, used for counts and background maps
        energy_axis_true : `~gammapy.maps.MapAxis`
            True energy axis used for IRF maps
        migra_axis : `~gammapy.maps.MapAxis`
            If set, this provides the migration axis for the energy dispersion map.
            If not set, an EDispKernelMap is produced instead. Default is None
        rad_axis : `~gammapy.maps.MapAxis`
            Rad axis for the psf map
        binsz_irf : float
            IRF Map pixel size in degrees.
        reference_time : `~astropy.time.Time`
            the reference time to use in GTI definition
        name : str
            Name of the returned dataset.
        meta_table : `~astropy.table.Table`
            Table listing informations on observations used to create the dataset.
            One line per observation for stacked datasets.

        Returns
        -------
        empty_maps : `MapDataset`
            A MapDataset containing zero filled maps
        """
        geoms = create_map_dataset_geoms(
            geom=geom,
            energy_axis_true=energy_axis_true,
            rad_axis=rad_axis,
            migra_axis=migra_axis,
            binsz_irf=binsz_irf,
        )

        kwargs.update(geoms)
        return cls.from_geoms(reference_time=reference_time, name=name, **kwargs)

    @property
    def mask_safe_image(self):
        """Reduced mask safe"""
        if self.mask_safe is None:
            return None
        return self.mask_safe.reduce_over_axes(func=np.logical_or)

    @property
    def mask_image(self):
        """Reduced mask"""
        if self.mask is None:
            mask = Map.from_geom(self._geom.to_image(), dtype=bool)
            mask.data |= True
            return mask

        return self.mask.reduce_over_axes(func=np.logical_or)

    @property
    def mask_safe_psf(self):
        """Mask safe for psf maps"""
        if self.mask_safe is None or self.psf is None:
            return None

        geom = self.psf.psf_map.geom.squash("energy_true").squash("rad")
        mask_safe_psf = self.mask_safe_image.interp_to_geom(geom.to_image())
        return mask_safe_psf.to_cube(geom.axes)

    @property
    def mask_safe_edisp(self):
        """Mask safe for edisp maps"""
        if self.mask_safe is None or self.edisp is None:
            return None

        if self.mask_safe.geom.is_region:
            return self.mask_safe

        geom = self.edisp.edisp_map.geom.squash("energy_true")

        if "migra" in geom.axes.names:
            geom = geom.squash("migra")
            mask_safe_edisp = self.mask_safe_image.interp_to_geom(geom.to_image())
            return mask_safe_edisp.to_cube(geom.axes)

        return self.mask_safe.interp_to_geom(geom)

    def to_masked(self, name=None):
        """Return masked dataset

        Parameters
        ----------
        name : str
            Name of the masked dataset.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            Masked dataset
        """
        dataset = self.__class__.from_geoms(**self.geoms, name=name)
        dataset.stack(self)
        return dataset

    def stack(self, other):
        r"""Stack another dataset in place.

        Safe mask is applied to compute the stacked counts data. Counts outside
        each dataset safe mask are lost.

        The stacking of 2 datasets is implemented as follows. Here, :math:`k`
        denotes a bin in reconstructed energy and :math:`j = {1,2}` is the dataset number

        The ``mask_safe`` of each dataset is defined as:

        .. math::

            \epsilon_{jk} =\left\{\begin{array}{cl} 1, &
            \mbox{if bin k is inside the thresholds}\\ 0, &
            \mbox{otherwise} \end{array}\right.

        Then the total ``counts`` and model background ``bkg`` are computed according to:

        .. math::

            \overline{\mathrm{n_{on}}}_k =  \mathrm{n_{on}}_{1k} \cdot \epsilon_{1k} +
             \mathrm{n_{on}}_{2k} \cdot \epsilon_{2k}

            \overline{bkg}_k = bkg_{1k} \cdot \epsilon_{1k} +
             bkg_{2k} \cdot \epsilon_{2k}

        The stacked ``safe_mask`` is then:

        .. math::

            \overline{\epsilon_k} = \epsilon_{1k} OR \epsilon_{2k}


        Parameters
        ----------
        other: `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            Map dataset to be stacked with this one. If other is an on-off
            dataset alpha * counts_off is used as a background model.
        """
        if self.counts and other.counts:
            self.counts.stack(other.counts, weights=other.mask_safe)

        if self.exposure and other.exposure:
            self.exposure.stack(other.exposure, weights=other.mask_safe_image)
            # TODO: check whether this can be improved e.g. handling this in GTI

            if "livetime" in other.exposure.meta and np.any(other.mask_safe_image):
                if "livetime" in self.exposure.meta:
                    self.exposure.meta["livetime"] += other.exposure.meta["livetime"]
                else:
                    self.exposure.meta["livetime"] = other.exposure.meta["livetime"].copy()

        if self.stat_type == "cash":
            if self.background and other.background:
                background = self.npred_background() * self.mask_safe
                background.stack(other.npred_background(), other.mask_safe)
                self.background = background

        if self.psf and other.psf:
            self.psf.stack(other.psf, weights=other.mask_safe_psf)

        if self.edisp and other.edisp:
            self.edisp.stack(other.edisp, weights=other.mask_safe_edisp)

        if self.mask_safe and other.mask_safe:
            self.mask_safe.stack(other.mask_safe)

        if self.gti and other.gti:
            self.gti.stack(other.gti)
            self.gti = self.gti.union()

        if self.meta_table and other.meta_table:
            self.meta_table = hstack_columns(self.meta_table, other.meta_table)
        elif other.meta_table:
            self.meta_table = other.meta_table.copy()

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def residuals(self, method="diff", **kwargs):
        """Compute residuals map.

        Parameters
        ----------
        method: {"diff", "diff/model", "diff/sqrt(model)"}
            Method used to compute the residuals. Available options are:
                - "diff" (default): data - model
                - "diff/model": (data - model) / model
                - "diff/sqrt(model)": (data - model) / sqrt(model)
        **kwargs : dict
            Keyword arguments forwarded to `Map.smooth()`

        Returns
        -------
        residuals : `gammapy.maps.Map`
            Residual map.
        """
        npred, counts = self.npred(), self.counts.copy()

        if self.mask:
            npred = npred * self.mask
            counts = counts * self.mask

        if kwargs:
            kwargs.setdefault("mode", "constant")
            kwargs.setdefault("width", "0.1 deg")
            kwargs.setdefault("kernel", "gauss")
            with np.errstate(invalid="ignore", divide="ignore"):
                npred = npred.smooth(**kwargs)
                counts = counts.smooth(**kwargs)
                if self.mask:
                    mask = self.mask.smooth(**kwargs)
                    npred /= mask
                    counts /= mask

        residuals = self._compute_residuals(counts, npred, method=method)

        if self.mask:
            residuals.data[~self.mask.data] = np.nan

        return residuals

    def plot_residuals_spatial(
        self,
        ax=None,
        method="diff",
        smooth_kernel="gauss",
        smooth_radius="0.1 deg",
        **kwargs,
    ):
        """Plot spatial residuals.

        The normalization used for the residuals computation can be controlled
        using the method parameter.

        Parameters
        ----------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            Axes to plot on.
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Normalization used to compute the residuals, see `MapDataset.residuals`.
        smooth_kernel : {"gauss", "box"}
            Kernel shape.
        smooth_radius: `~astropy.units.Quantity`, str or float
            Smoothing width given as quantity or float. If a float is given, it
            is interpreted as smoothing width in pixels.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.imshow`.

        Returns
        -------
        ax : `~astropy.visualization.wcsaxes.WCSAxes`
            WCSAxes object.
        """
        counts, npred = self.counts.copy(), self.npred()

        if counts.geom.is_region:
            raise ValueError("Cannot plot spatial residuals for RegionNDMap")

        if self.mask is not None:
            counts *= self.mask
            npred *= self.mask

        counts_spatial = counts.sum_over_axes().smooth(
            width=smooth_radius, kernel=smooth_kernel
        )
        npred_spatial = npred.sum_over_axes().smooth(
            width=smooth_radius, kernel=smooth_kernel
        )
        residuals = self._compute_residuals(counts_spatial, npred_spatial, method)

        if self.mask_safe is not None:
            mask = self.mask_safe.reduce_over_axes(func=np.logical_or, keepdims=True)
            residuals.data[~mask.data] = np.nan

        kwargs.setdefault("add_cbar", True)
        kwargs.setdefault("cmap", "coolwarm")
        kwargs.setdefault("vmin", -5)
        kwargs.setdefault("vmax", 5)
        _, ax, _ = residuals.plot(ax, **kwargs)

        return ax

    def plot_residuals_spectral(self, ax=None, method="diff", region=None, **kwargs):
        """Plot spectral residuals.

        The residuals are extracted from the provided region, and the normalization
        used for its computation can be controlled using the method parameter.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            Axes to plot on.
        method : {"diff", "diff/sqrt(model)"}
            Normalization used to compute the residuals, see `SpectrumDataset.residuals`.
        region: `~regions.SkyRegion` (required)
            Target sky region.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.axes.Axes.errorbar`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            Axes object.
        """
        counts, npred = self.counts.copy(), self.npred()

        if self.mask is None:
            mask = self.counts.copy()
            mask.data = 1
        else:
            mask = self.mask
        counts *= mask
        npred *= mask

        counts_spec = counts.get_spectrum(region)
        npred_spec = npred.get_spectrum(region)
        residuals = self._compute_residuals(counts_spec, npred_spec, method)

        if method == "diff":
            if self.stat_type == "wstat":
                counts_off = (self.counts_off * mask).get_spectrum(region).data
                norm = (self.background * mask).get_spectrum(region).data
                mu_sig = (self.npred_signal() * mask).get_spectrum(region).data
                stat = WStatCountsStatistic(
                    n_on=counts_spec.data,
                    n_off=counts_off,
                    alpha=norm / counts_off,
                    mu_sig=mu_sig,
                )
            elif self.stat_type == "cash":
                stat = CashCountsStatistic(counts_spec.data, npred_spec.data)
            yerr = stat.error.flatten()
        elif method == "diff/sqrt(model)":
            yerr = np.ones_like(residuals.data.flatten())
        else:
            raise ValueError(
                'Invalid method, choose between "diff" and "diff/sqrt(model)"'
            )

        kwargs.setdefault("color", kwargs.pop("c", "black"))
        ax = residuals.plot(ax, yerr=yerr, **kwargs)
        ax.axhline(0, color=kwargs["color"], lw=0.5)

        label = self._residuals_labels[method]
        ax.set_ylabel(f"Residuals ({label})")
        ax.set_yscale("linear")
        ymin = 1.05 * np.nanmin(residuals.data - yerr)
        ymax = 1.05 * np.nanmax(residuals.data + yerr)
        ax.set_ylim(ymin, ymax)
        return ax

    def plot_residuals(
        self,
        ax_spatial=None,
        ax_spectral=None,
        kwargs_spatial=None,
        kwargs_spectral=None,
    ):
        """Plot spatial and spectral residuals in two panels.

        Calls `~MapDataset.plot_residuals_spatial` and `~MapDataset.plot_residuals_spectral`.
        The spectral residuals are extracted from the provided region, and the
        normalization used for its computation can be controlled using the method
        parameter. The region outline is overlaid on the residuals map.

        Parameters
        ----------
        ax_spatial : `~astropy.visualization.wcsaxes.WCSAxes`
            Axes to plot spatial residuals on.
        ax_spectral : `~matplotlib.axes.Axes`
            Axes to plot spectral residuals on.
        kwargs_spatial : dict
            Keyword arguments passed to `~MapDataset.plot_residuals_spatial`.
        kwargs_spectral : dict (``region`` required)
            Keyword arguments passed to `~MapDataset.plot_residuals_spectral`.

        Returns
        -------
        ax_spatial, ax_spectral : `~astropy.visualization.wcsaxes.WCSAxes`, `~matplotlib.axes.Axes`
            Spatial and spectral residuals plots.
        """
        ax_spatial, ax_spectral = get_axes(
            ax_spatial,
            ax_spectral,
            12,
            4,
            [1, 2, 1],
            [1, 2, 2],
            {"projection": self._geom.to_image().wcs},
        )
        kwargs_spatial = kwargs_spatial or {}

        self.plot_residuals_spatial(ax_spatial, **kwargs_spatial)
        self.plot_residuals_spectral(ax_spectral, **kwargs_spectral)

        # Overlay spectral extraction region on the spatial residuals
        region = kwargs_spectral["region"]
        pix_region = region.to_pixel(self._geom.to_image().wcs)
        pix_region.plot(ax=ax_spatial)

        return ax_spatial, ax_spectral

    @lazyproperty
    def _counts_data(self):
        return self.counts.data.astype(float)

    def stat_sum(self):
        """Total likelihood given the current model parameters."""
        counts, npred = self._counts_data, self.npred().data

        if self.mask is not None:
            return cash_sum_cython(counts[self.mask.data], npred[self.mask.data])
        else:
            return cash_sum_cython(counts.ravel(), npred.ravel())

    def fake(self, random_state="random-seed"):
        """Simulate fake counts for the current model and reduced IRFs.

        This method overwrites the counts defined on the dataset object.

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
                Defines random number generator initialisation.
                Passed to `~gammapy.utils.random.get_random_state`.
        """
        random_state = get_random_state(random_state)
        npred = self.npred()
        npred.data = random_state.poisson(npred.data)
        self.counts = npred

    def to_hdulist(self):
        """Convert map dataset to list of HDUs.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            Map dataset list of HDUs.
        """
        # TODO: what todo about the model and background model parameters?
        exclude_primary = slice(1, None)

        hdu_primary = fits.PrimaryHDU()
        hdulist = fits.HDUList([hdu_primary])
        if self.counts is not None:
            hdulist += self.counts.to_hdulist(hdu="counts")[exclude_primary]

        if self.exposure is not None:
            hdulist += self.exposure.to_hdulist(hdu="exposure")[exclude_primary]

        if self.background is not None:
            hdulist += self.background.to_hdulist(hdu="background")[exclude_primary]

        if self.edisp is not None:
            hdulist += self.edisp.to_hdulist()[exclude_primary]

        if self.psf is not None:
            hdulist += self.psf.to_hdulist()[exclude_primary]

        if self.mask_safe is not None:
            hdulist += self.mask_safe.to_hdulist(hdu="mask_safe")[exclude_primary]

        if self.mask_fit is not None:
            hdulist += self.mask_fit.to_hdulist(hdu="mask_fit")[exclude_primary]

        if self.gti is not None:
            hdulist.append(fits.BinTableHDU(self.gti.table, name="GTI"))

        return hdulist

    @classmethod
    def from_hdulist(cls, hdulist, name=None, lazy=False, format="gadf"):
        """Create map dataset from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.
        name : str
            Name of the new dataset.
        format : {"gadf"}
            Format the hdulist is given in.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        name = make_name(name)
        kwargs = {"name": name}

        if "COUNTS" in hdulist:
            kwargs["counts"] = Map.from_hdulist(hdulist, hdu="counts", format=format)

        if "EXPOSURE" in hdulist:
            exposure = Map.from_hdulist(hdulist, hdu="exposure", format=format)
            if exposure.geom.axes[0].name == "energy":
                exposure.geom.axes[0].name = "energy_true"
            kwargs["exposure"] = exposure

        if "BACKGROUND" in hdulist:
            kwargs["background"] = Map.from_hdulist(hdulist, hdu="background", format=format)

        if "EDISP" in hdulist:
            edisp_map = Map.from_hdulist(hdulist, hdu="edisp", format=format)

            try:
                exposure_map = Map.from_hdulist(hdulist, hdu="edisp_exposure", format=format)
            except KeyError:
                exposure_map = None

            if edisp_map.geom.axes[0].name == "energy":
                kwargs["edisp"] = EDispKernelMap(edisp_map, exposure_map)
            else:
                kwargs["edisp"] = EDispMap(edisp_map, exposure_map)

        if "PSF" in hdulist:
            psf_map = Map.from_hdulist(hdulist, hdu="psf", format=format)
            try:
                exposure_map = Map.from_hdulist(hdulist, hdu="psf_exposure", format=format)
            except KeyError:
                exposure_map = None
            kwargs["psf"] = PSFMap(psf_map, exposure_map)

        if "MASK_SAFE" in hdulist:
            mask_safe = Map.from_hdulist(hdulist, hdu="mask_safe", format=format)
            mask_safe.data = mask_safe.data.astype(bool)
            kwargs["mask_safe"] = mask_safe

        if "MASK_FIT" in hdulist:
            mask_fit = Map.from_hdulist(hdulist, hdu="mask_fit", format=format)
            mask_fit.data = mask_fit.data.astype(bool)
            kwargs["mask_fit"] = mask_fit

        if "GTI" in hdulist:
            gti = GTI(Table.read(hdulist, hdu="GTI"))
            kwargs["gti"] = gti

        return cls(**kwargs)

    def write(self, filename, overwrite=False):
        """Write map dataset to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool
            Overwrite file if it exists.
        """
        self.to_hdulist().writeto(str(make_path(filename)), overwrite=overwrite)

    @classmethod
    def _read_lazy(cls, name, filename, cache, format=format):
        kwargs = {"name": name}
        try:
            kwargs["gti"] = GTI.read(filename)
        except KeyError:
            pass

        path = make_path(filename)
        for hdu_name in ["counts", "exposure", "mask_fit", "mask_safe", "background"]:
            kwargs[hdu_name] = HDULocation(
                hdu_class="map",
                file_dir=path.parent,
                file_name=path.name,
                hdu_name=hdu_name.upper(),
                cache=cache,
                format=format
            )

        kwargs["edisp"] = HDULocation(
            hdu_class="edisp_kernel_map",
            file_dir=path.parent,
            file_name=path.name,
            hdu_name="EDISP",
            cache=cache,
            format=format
        )

        kwargs["psf"] = HDULocation(
            hdu_class="psf_map",
            file_dir=path.parent,
            file_name=path.name,
            hdu_name="PSF",
            cache=cache,
            format=format
        )

        return cls(**kwargs)

    @classmethod
    def read(cls, filename, name=None, lazy=False, cache=True, format="gadf"):
        """Read map dataset from file.

        Parameters
        ----------
        filename : str
            Filename to read from.
        name : str
            Name of the new dataset.
        lazy : bool
            Whether to lazy load data into memory
        cache : bool
            Whether to cache the data after loading.
        format : {"gadf"}
            Format of the dataset file.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        name = make_name(name)

        if lazy:
            return cls._read_lazy(name=name, filename=filename, cache=cache, format=format)
        else:
            with fits.open(str(make_path(filename)), memmap=False) as hdulist:
                return cls.from_hdulist(hdulist, name=name, format=format)

    @classmethod
    def from_dict(cls, data, lazy=False, cache=True):
        """Create from dicts and models list generated from YAML serialization."""
        filename = make_path(data["filename"])
        dataset = cls.read(filename, name=data["name"], lazy=lazy, cache=cache)
        return dataset

    def info_dict(self, in_safe_data_range=True):
        """Info dict with summary statistics, summed over energy

        Parameters
        ----------
        in_safe_data_range : bool
            Whether to sum only in the safe energy range

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info = {}
        info["name"] = self.name

        if self.mask_safe and in_safe_data_range:
            mask = self.mask_safe.data.astype(bool)
        else:
            mask = slice(None)

        counts = np.nan
        if self.counts:
            counts = self.counts.data[mask].sum()

        info["counts"] = counts

        background = np.nan
        if self.background:
            background = self.background.data[mask].sum()

        info["background"] = background

        info["excess"] = counts - background
        info["sqrt_ts"] = CashCountsStatistic(counts, background).sqrt_ts

        npred = np.nan
        if self.models or not np.isnan(background):
            npred = self.npred().data[mask].sum()

        info["npred"] = npred

        npred_background = np.nan
        if self.background:
            npred_background = self.npred_background().data[mask].sum()

        info["npred_background"] = npred_background

        npred_signal = np.nan
        if self.models:
            npred_signal = self.npred_signal().data[mask].sum()

        info["npred_signal"] = npred_signal

        exposure_min, exposure_max, livetime = np.nan, np.nan, np.nan

        if self.exposure is not None:
            mask_exposure = self.exposure.data > 0

            if self.mask_safe is not None:
                mask_spatial = self.mask_safe.reduce_over_axes(func=np.logical_or).data
                mask_exposure = mask_exposure & mask_spatial[np.newaxis, :, :]
                if not mask_exposure.any():
                    mask_exposure = slice(None)

            exposure_min = np.min(self.exposure.quantity[mask_exposure])
            exposure_max = np.max(self.exposure.quantity[mask_exposure])
            livetime = self.exposure.meta.get("livetime", np.nan * u.s).copy()

        info["exposure_min"] = exposure_min
        info["exposure_max"] = exposure_max
        info["livetime"] = livetime

        ontime = u.Quantity(np.nan, "s")
        if self.gti:
            ontime = self.gti.time_sum

        info["ontime"] = ontime

        info["counts_rate"] = info["counts"] / info["livetime"]
        info["background_rate"] = info["background"] / info["livetime"]
        info["excess_rate"] = info["excess"] / info["livetime"]

        # data section
        n_bins = 0
        if self.counts is not None:
            n_bins = self.counts.data.size
        info["n_bins"] = n_bins

        n_fit_bins = 0
        if self.mask is not None:
            n_fit_bins = np.sum(self.mask.data)

        info["n_fit_bins"] = n_fit_bins
        info["stat_type"] = self.stat_type

        stat_sum = np.nan
        if self.counts is not None and self.models is not None:
            stat_sum = self.stat_sum()

        info["stat_sum"] = stat_sum

        return info

    def to_spectrum_dataset(self, on_region, containment_correction=False, name=None):
        """Return a ~gammapy.datasets.SpectrumDataset from on_region.

        Counts and background are summed in the on_region. Exposure is taken
        from the average exposure.

        The energy dispersion kernel is obtained at the on_region center.
        Only regions with centers are supported.

        The model is not exported to the ~gammapy.datasets.SpectrumDataset.
        It must be set after the dataset extraction.

        Parameters
        ----------
        on_region : `~regions.SkyRegion`
            the input ON region on which to extract the spectrum
        containment_correction : bool
            Apply containment correction for point sources and circular on regions
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `~gammapy.datasets.SpectrumDataset`
            the resulting reduced dataset
        """
        from .spectrum import SpectrumDataset

        dataset = self.to_spectrum(region=on_region, name=name)

        if containment_correction:
            if not isinstance(on_region, CircleSkyRegion):
                raise TypeError(
                    "Containment correction is only supported for"
                    " `CircleSkyRegion`."
                )
            elif self.psf is None or isinstance(self.psf, PSFKernel):
                raise ValueError("No PSFMap set. Containment correction impossible")
            else:
                geom = dataset.exposure.geom
                energy_true = geom.axes["energy_true"].center
                containment = self.psf.containment(
                    position=on_region.center,
                    energy_true=energy_true,
                    rad=on_region.radius
                )
                dataset.exposure.quantity *= containment.reshape(geom.data_shape)

        kwargs = {}

        for name in ["counts", "edisp", "mask_safe", "mask_fit", "exposure", "gti", "meta_table"]:
            kwargs[name] = getattr(dataset, name)

        if self.stat_type == "cash":
            kwargs["background"] = dataset.background

        return SpectrumDataset(**kwargs)

    def to_spectrum(self, region, name=None):
        """Return a ~gammapy.datasets.SpectrumDataset from on_region.

        The model is not exported to the ~gammapy.datasets.SpectrumDataset.
        It must be set after the dataset extraction.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region from which to extract the spectrum
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `~gammapy.datasets.MapDataset`
            the resulting reduced dataset
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}

        if self.mask_safe:
            kwargs["mask_safe"] = self.mask_safe.to_region_nd_map(region, func=np.any)

        if self.mask_fit:
            kwargs["mask_fit"] = self.mask_fit.to_region_nd_map(region, func=np.any)

        if self.counts:
            kwargs["counts"] = self.counts.to_region_nd_map(
                region, np.sum, weights=self.mask_safe
            )

        if self.stat_type == "cash" and self.background:
            kwargs["background"] = self.background.to_region_nd_map(
                region, func=np.sum, weights=self.mask_safe
            )

        if self.exposure:
            kwargs["exposure"] = self.exposure.to_region_nd_map(region, func=np.mean)

        region = region.center if region else None

        # TODO: Compute average psf in region
        if self.psf:
            kwargs["psf"] = self.psf.to_region_nd_map(region)

        # TODO: Compute average edisp in region
        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.to_region_nd_map(region)

        return self.__class__(**kwargs)

    def cutout(self, position, width, mode="trim", name=None):
        """Cutout map dataset.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : tuple of `~astropy.coordinates.Angle`
            Angular sizes of the region in (lon, lat) in that specific order.
            If only one value is passed, a square region is extracted.
        mode : {'trim', 'partial', 'strict'}
            Mode option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.
        name : str
            Name of the new dataset.

        Returns
        -------
        cutout : `MapDataset`
            Cutout map dataset.
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}
        cutout_kwargs = {"position": position, "width": width, "mode": mode}

        if self.counts is not None:
            kwargs["counts"] = self.counts.cutout(**cutout_kwargs)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.cutout(**cutout_kwargs)

        if self.background is not None and self.stat_type == "cash":
            kwargs["background"] = self.background.cutout(**cutout_kwargs)

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.cutout(**cutout_kwargs)

        if self.psf is not None:
            kwargs["psf"] = self.psf.cutout(**cutout_kwargs)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.cutout(**cutout_kwargs)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.cutout(**cutout_kwargs)

        return self.__class__(**kwargs)

    def downsample(self, factor, axis_name=None, name=None):
        """Downsample map dataset.

        The PSFMap and EDispKernelMap are not downsampled, except if
        a corresponding axis is given.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        axis_name : str
            Which non-spatial axis to downsample. By default only spatial axes are downsampled.
        name : str
            Name of the downsampled dataset.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            Downsampled map dataset.
        """
        name = make_name(name)

        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}

        if self.counts is not None:
            kwargs["counts"] = self.counts.downsample(
                factor=factor,
                preserve_counts=True,
                axis_name=axis_name,
                weights=self.mask_safe,
            )

        if self.exposure is not None:
            if axis_name is None:
                kwargs["exposure"] = self.exposure.downsample(
                    factor=factor, preserve_counts=False, axis_name=None
                )
            else:
                kwargs["exposure"] = self.exposure.copy()

        if self.background is not None and self.stat_type == "cash":
            kwargs["background"] = self.background.downsample(
                factor=factor, axis_name=axis_name, weights=self.mask_safe
            )

        if self.edisp is not None:
            if axis_name is not None:
                kwargs["edisp"] = self.edisp.downsample(
                    factor=factor, axis_name=axis_name
                )
            else:
                kwargs["edisp"] = self.edisp.copy()

        if self.psf is not None:
            kwargs["psf"] = self.psf.copy()

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.downsample(
                factor=factor, preserve_counts=False, axis_name=axis_name
            )

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.downsample(
                factor=factor, preserve_counts=False, axis_name=axis_name
            )

        return self.__class__(**kwargs)

    def pad(self, pad_width, mode="constant", name=None):
        """Pad the spatial dimensions of the dataset.

        The padding only applies to counts, masks, background and exposure.

        Counts, background and masks are padded with zeros, exposure is padded with edge value.

        Parameters
        ----------
        pad_width : {sequence, array_like, int}
            Number of pixels padded to the edges of each axis.
        name : str
            Name of the padded dataset.

        Returns
        -------
        dataset : `MapDataset`
            Padded map dataset.

        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}

        if self.counts is not None:
            kwargs["counts"] = self.counts.pad(pad_width=pad_width, mode=mode)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.pad(pad_width=pad_width, mode=mode)

        if self.background is not None:
            kwargs["background"] = self.background.pad(pad_width=pad_width, mode=mode)

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.copy()

        if self.psf is not None:
            kwargs["psf"] = self.psf.copy()

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.pad(pad_width=pad_width, mode=mode)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.pad(pad_width=pad_width, mode=mode)

        return self.__class__(**kwargs)

    def slice_by_idx(self, slices, name=None):
        """Slice sub dataset.

        The slicing only applies to the maps that define the corresponding axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.
        name : str
            Name of the sliced dataset.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            Sliced dataset
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}

        if self.counts is not None:
            kwargs["counts"] = self.counts.slice_by_idx(slices=slices)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.slice_by_idx(slices=slices)

        if self.background is not None and self.stat_type == "cash":
            kwargs["background"] = self.background.slice_by_idx(slices=slices)

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.slice_by_idx(slices=slices)

        if self.psf is not None:
            kwargs["psf"] = self.psf.slice_by_idx(slices=slices)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.slice_by_idx(slices=slices)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.slice_by_idx(slices=slices)

        return self.__class__(**kwargs)

    def slice_by_energy(self, energy_min, energy_max, name=None):
        """Select and slice datasets in energy range

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.
        name : str
            Name of the sliced dataset.

        Returns
        -------
        dataset : `MapDataset`
            Sliced Dataset

        """
        name = make_name(name)
        energy_axis = self._geom.axes["energy"]

        group = energy_axis.group_table(edges=[energy_min, energy_max])

        is_normal = group["bin_type"] == "normal   "
        group = group[is_normal]

        slices = {
            "energy": slice(int(group["idx_min"][0]), int(group["idx_max"][0]) + 1)
        }

        return self.slice_by_idx(slices, name=name)

    def reset_data_cache(self):
        """Reset data cache to free memory space"""
        for name in self._lazy_data_members:
            if self.__dict__.pop(name, False):
                log.info(f"Clearing {name} cache for dataset {self.name}")

    def resample_energy_axis(self, energy_axis, name=None):
        """Resample MapDataset over new reco energy axis.

        Counts are summed taking into account safe mask.

        Parameters
        ----------
        energy_axis : `~gammapy.maps.MapAxis`
            New reconstructed energy axis.
        name: str
            Name of the new dataset.

        Returns
        -------
        dataset: `MapDataset` or `SpectrumDataset`
            Resampled dataset.
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}

        if self.exposure:
            kwargs["exposure"] = self.exposure

        if self.psf:
            kwargs["psf"] = self.psf

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.resample_axis(
                axis=energy_axis, ufunc=np.logical_or
            )

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.resample_axis(
                axis=energy_axis, ufunc=np.logical_or
            )

        if self.counts is not None:
            kwargs["counts"] = self.counts.resample_axis(
                axis=energy_axis, weights=self.mask_safe
            )

        if self.background is not None and self.stat_type == "cash":
            kwargs["background"] = self.background.resample_axis(
                axis=energy_axis, weights=self.mask_safe
            )

        # Mask_safe or mask_irf??
        if isinstance(self.edisp, EDispKernelMap):
            kwargs["edisp"] = self.edisp.resample_energy_axis(
                energy_axis=energy_axis, weights=self.mask_safe_edisp
            )
        else:  # None or EDispMap
            kwargs["edisp"] = self.edisp

        return self.__class__(**kwargs)

    def to_image(self, name=None):
        """Create images by summing over the reconstructed energy axis.

        Parameters
        ----------
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            Dataset integrated over non-spatial axes.
        """
        energy_axis = self._geom.axes["energy"].squash()
        return self.resample_energy_axis(energy_axis=energy_axis, name=name)


class MapDatasetOnOff(MapDataset):
    """Map dataset for on-off likelihood fitting.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    counts_off : `~gammapy.maps.WcsNDMap`
        Ring-convolved counts cube
    acceptance : `~gammapy.maps.WcsNDMap`
        Acceptance from the IRFs
    acceptance_off : `~gammapy.maps.WcsNDMap`
        Acceptance off
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    mask_fit : `~gammapy.maps.WcsNDMap`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.irf.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    mask_safe : `~gammapy.maps.WcsNDMap`
        Mask defining the safe data range.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    meta_table : `~astropy.table.Table`
        Table listing informations on observations used to create the dataset.
        One line per observation for stacked datasets.
    name : str
        Name of the dataset.


    See Also
    --------
    MapDataset, SpectrumDataset, FluxPointsDataset

    """

    stat_type = "wstat"
    tag = "MapDatasetOnOff"

    def __init__(
        self,
        models=None,
        counts=None,
        counts_off=None,
        acceptance=None,
        acceptance_off=None,
        exposure=None,
        mask_fit=None,
        psf=None,
        edisp=None,
        name=None,
        mask_safe=None,
        gti=None,
        meta_table=None,
    ):
        self._name = make_name(name)
        self._evaluators = {}

        self.counts = counts
        self.counts_off = counts_off
        self.exposure = exposure
        self.acceptance = acceptance
        self.acceptance_off = acceptance_off
        self.gti = gti
        self.mask_fit = mask_fit
        self.psf = psf
        self.edisp = edisp
        self.models = models
        self.mask_safe = mask_safe
        self.meta_table = meta_table

    def __str__(self):
        str_ = super().__str__()

        counts_off = np.nan
        if self.counts_off is not None:
            counts_off = np.sum(self.counts_off.data)
        str_ += "\t{:32}: {:.0f} \n".format("Total counts_off", counts_off)

        acceptance = np.nan
        if self.acceptance is not None:
            acceptance = np.sum(self.acceptance.data)
        str_ += "\t{:32}: {:.0f} \n".format("Acceptance", acceptance)

        acceptance_off = np.nan
        if self.acceptance_off is not None:
            acceptance_off = np.sum(self.acceptance_off.data)
        str_ += "\t{:32}: {:.0f} \n".format("Acceptance off", acceptance_off)

        return str_.expandtabs(tabsize=2)

    @property
    def _geom(self):
        """Main analysis geometry"""
        if self.counts is not None:
            return self.counts.geom
        elif self.counts_off is not None:
            return self.counts_off.geom
        elif self.acceptance is not None:
            return self.acceptance.geom
        elif self.acceptance_off is not None:
            return self.acceptance_off.geom
        else:
            raise ValueError(
                "Either 'counts', 'counts_off', 'acceptance' or 'acceptance_of' must be defined."
            )

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions

        See :ref:`wstat`

        Returns
        -------
        alpha : `Map`
            Alpha map
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha = self.acceptance / self.acceptance_off

        alpha.data = np.nan_to_num(alpha.data)
        return alpha

    def npred_background(self):
        """Prediced background counts estimated from the marginalized likelihood estimate.

        See :ref:`wstat`

        Returns
        -------
        npred_background : `Map`
            Predicted background counts
        """
        mu_bkg = self.alpha.data * get_wstat_mu_bkg(
            n_on=self.counts.data,
            n_off=self.counts_off.data,
            alpha=self.alpha.data,
            mu_sig=self.npred_signal().data,
        )
        mu_bkg = np.nan_to_num(mu_bkg)
        return Map.from_geom(geom=self._geom, data=mu_bkg)

    def npred_off(self):
        """Predicted counts in the off region

        See :ref:`wstat`

        Returns
        -------
        npred_off : `Map`
            Predicted off counts
        """
        return self.npred_background() / self.alpha

    @property
    def background(self):
        """Computed as alpha * n_off

        See :ref:`wstat`

        Returns
        -------
        background : `Map`
            Background map
        """
        return self.alpha * self.counts_off

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        mu_sig = self.npred_signal().data
        on_stat_ = wstat(
            n_on=self.counts.data,
            n_off=self.counts_off.data,
            alpha=list(self.alpha.data),
            mu_sig=mu_sig,
        )
        return np.nan_to_num(on_stat_)

    @classmethod
    def from_geoms(
        cls,
        geom,
        geom_exposure,
        geom_psf=None,
        geom_edisp=None,
        reference_time="2000-01-01",
        name=None,
        **kwargs,
    ):
        """Create a MapDatasetOnOff object  swith zero filled maps according to the specified geometries

        Parameters
        ----------
        geom : `gammapy.maps.WcsGeom`
            geometry for the counts, counts_off, acceptance and acceptance_off maps
        geom_exposure : `gammapy.maps.WcsGeom`
            geometry for the exposure map
        geom_psf : `gammapy.maps.WcsGeom`
            geometry for the psf map
        geom_edisp : `gammapy.maps.WcsGeom`
            geometry for the energy dispersion kernel map.
            If geom_edisp has a migra axis, this wil create an EDispMap instead.
        reference_time : `~astropy.time.Time`
            the reference time to use in GTI definition
        name : str
            Name of the returned dataset.

        Returns
        -------
        empty_maps : `MapDatasetOnOff`
            A MapDatasetOnOff containing zero filled maps
        """
        #  TODO: it seems the super() pattern does not work here?
        dataset = MapDataset.from_geoms(
            geom=geom,
            geom_exposure=geom_exposure,
            geom_psf=geom_psf,
            geom_edisp=geom_edisp,
            name=name,
            reference_time=reference_time,
            **kwargs
        )

        off_maps = {}

        for key in ["counts_off", "acceptance", "acceptance_off"]:
            off_maps[key] = Map.from_geom(geom, unit="")

        return cls.from_map_dataset(dataset, name=name, **off_maps)

    @classmethod
    def from_map_dataset(
        cls, dataset, acceptance, acceptance_off, counts_off=None, name=None
    ):
        """Create on off dataset from a map dataset.

        Parameters
        ----------
        dataset : `MapDataset`
            Spectrum dataset defining counts, edisp, aeff, livetime etc.
        acceptance : `Map`
            Relative background efficiency in the on region.
        acceptance_off : `Map`
            Relative background efficiency in the off region.
        counts_off : `Map`
            Off counts map . If the dataset provides a background model,
            and no off counts are defined. The off counts are deferred from
            counts_off / alpha.
        name : str
            Name of the returned dataset.

        Returns
        -------
        dataset : `MapDatasetOnOff`
            Map dataset on off.

        """
        if counts_off is None and dataset.background is not None:
            alpha = acceptance / acceptance_off
            counts_off = dataset.npred_background() / alpha

        if np.isscalar(acceptance):
            acceptance = Map.from_geom(
                dataset._geom, data=acceptance
            )

        if np.isscalar(acceptance_off):
            acceptance_off = Map.from_geom(
                dataset._geom, data=acceptance_off
            )

        return cls(
            models=dataset.models,
            counts=dataset.counts,
            exposure=dataset.exposure,
            counts_off=counts_off,
            edisp=dataset.edisp,
            psf=dataset.psf,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            gti=dataset.gti,
            name=name,
            meta_table=dataset.meta_table,
        )

    def to_map_dataset(self, name=None):
        """ Convert a MapDatasetOnOff to  MapDataset
        The background model template is taken as alpha*counts_off

        Parameters
        ----------
        name: str
            Name of the new dataset

        Returns
        -------
        dataset: `MapDataset`
            Map dataset with cash statistics
        """
        name = make_name(name)

        return MapDataset(
            counts=self.counts,
            exposure=self.exposure,
            psf=self.psf,
            edisp=self.edisp,
            name=name,
            gti=self.gti,
            mask_fit=self.mask_fit,
            mask_safe=self.mask_safe,
            background=self.counts_off * self.alpha,
            meta_table=self.meta_table,
        )

    @property
    def _is_stackable(self):
        """Check if the Dataset contains enough information to be stacked"""
        incomplete = self.acceptance_off is None or self.acceptance is None or self.counts_off is None
        unmasked = np.any(self.mask_safe.data)
        if incomplete and unmasked:
            return False
        else:
            return True

    def stack(self, other):
        r"""Stack another dataset in place.

        The ``acceptance`` of the stacked dataset is normalized to 1,
        and the stacked ``acceptance_off`` is scaled so that:

        .. math::
            \alpha_\text{stacked} =
            \frac{1}{a_\text{off}} =
            \frac{\alpha_1\text{OFF}_1 + \alpha_2\text{OFF}_2}{\text{OFF}_1 + OFF_2}

        Parameters
        ----------
        other : `MapDatasetOnOff`
            Other dataset
        """
        if not isinstance(other, MapDatasetOnOff):
            raise TypeError("Incompatible types for MapDatasetOnOff stacking")

        if not self._is_stackable or not other._is_stackable:
            raise ValueError("Cannot stack incomplete MapDatsetOnOff.")

        geom = self.counts.geom
        total_off = Map.from_geom(geom)
        total_alpha = Map.from_geom(geom)

        if self.counts_off:
            total_off.stack(self.counts_off, weights=self.mask_safe)
            total_alpha.stack(self.alpha * self.counts_off, weights=self.mask_safe)
        if other.counts_off:
            total_off.stack(other.counts_off, weights=other.mask_safe)
            total_alpha.stack(other.alpha * other.counts_off, weights=other.mask_safe)

        with np.errstate(divide="ignore", invalid="ignore"):
            acceptance_off = total_off / total_alpha
            average_alpha = total_alpha.data.sum() / total_off.data.sum()

        # For the bins where the stacked OFF counts equal 0, the alpha value is performed by weighting on the total
        # OFF counts of each run
        is_zero = total_off.data == 0
        acceptance_off.data[is_zero] = 1 / average_alpha

        self.acceptance.data[...] = 1
        self.acceptance_off = acceptance_off

        self.counts_off = total_off

        super().stack(other)

    def stat_sum(self):
        """Total likelihood given the current model parameters."""
        return Dataset.stat_sum(self)

    def fake(self, npred_background, random_state="random-seed"):
        """Simulate fake counts (on and off) for the current model and reduced IRFs.

        This method overwrites the counts defined on the dataset object.

        Parameters
        ----------
        random_state : {int, 'random-seed', 'global-rng', `~numpy.random.RandomState`}
                Defines random number generator initialisation.
                Passed to `~gammapy.utils.random.get_random_state`.
        """
        random_state = get_random_state(random_state)
        npred = self.npred_signal()
        npred.data = random_state.poisson(npred.data)

        npred_bkg = random_state.poisson(npred_background.data)

        self.counts = npred + npred_bkg

        npred_off = npred_background / self.alpha
        npred_off.data = random_state.poisson(npred_off.data)
        self.counts_off = npred_off

    def to_hdulist(self):
        """Convert map dataset to list of HDUs.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
            Map dataset list of HDUs.
        """
        hdulist = super().to_hdulist()
        exclude_primary = slice(1, None)

        del hdulist["BACKGROUND"]
        del hdulist["BACKGROUND_BANDS"]

        if self.counts_off is not None:
            hdulist += self.counts_off.to_hdulist(hdu="counts_off")[exclude_primary]

        if self.acceptance is not None:
            hdulist += self.acceptance.to_hdulist(hdu="acceptance")[exclude_primary]

        if self.acceptance_off is not None:
            hdulist += self.acceptance_off.to_hdulist(hdu="acceptance_off")[
                exclude_primary
            ]

        return hdulist

    @classmethod
    def from_hdulist(cls, hdulist, name=None, format="gadf"):
        """Create map dataset from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.
        name : str
            Name of the new dataset.
        format : {"gadf"}
            Format the hdulist is given in.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        kwargs = {}
        kwargs["name"] = name

        if "COUNTS" in hdulist:
            kwargs["counts"] = Map.from_hdulist(hdulist, hdu="counts", format=format)

        if "COUNTS_OFF" in hdulist:
            kwargs["counts_off"] = Map.from_hdulist(hdulist, hdu="counts_off", format=format)

        if "ACCEPTANCE" in hdulist:
            kwargs["acceptance"] = Map.from_hdulist(hdulist, hdu="acceptance", format=format)

        if "ACCEPTANCE_OFF" in hdulist:
            kwargs["acceptance_off"] = Map.from_hdulist(hdulist, hdu="acceptance_off", format=format)

        if "EXPOSURE" in hdulist:
            kwargs["exposure"] = Map.from_hdulist(hdulist, hdu="exposure", format=format)

        # TODO: this misses the PSFMap and EDispMap

        if "MASK_SAFE" in hdulist:
            mask_safe = Map.from_hdulist(hdulist, hdu="mask_safe", format=format)
            kwargs["mask_safe"] = mask_safe

        if "MASK_FIT" in hdulist:
            mask_fit = Map.from_hdulist(hdulist, hdu="mask_fit", format=format)
            kwargs["mask_fit"] = mask_fit

        if "GTI" in hdulist:
            gti = GTI(Table.read(hdulist, hdu="GTI"))
            kwargs["gti"] = gti
        return cls(**kwargs)

    def info_dict(self, in_safe_data_range=True):
        """Basic info dict with summary statistics

        If a region is passed, then a spectrum dataset is
        extracted, and the corresponding info returned.

        Parameters
        ----------
        in_safe_data_range : bool
            Whether to sum only in the safe energy range

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        # TODO: remove code duplication with SpectrumDatasetOnOff
        info = super().info_dict(in_safe_data_range)

        if self.mask_safe and in_safe_data_range:
            mask = self.mask_safe.data.astype(bool)
        else:
            mask = slice(None)

        counts_off = np.nan
        if self.counts_off is not None:
            counts_off = self.counts_off.data[mask].sum()

        info["counts_off"] = counts_off

        acceptance = 1
        if self.acceptance:
            # TODO: handle energy dependent a_on / a_off
            acceptance = self.acceptance.data[mask].sum()

        info["acceptance"] = acceptance

        acceptance_off = np.nan
        if self.acceptance_off:
            acceptance_off = acceptance * counts_off / info["background"]

        info["acceptance_off"] = acceptance_off

        alpha = np.nan
        if self.acceptance_off and self.acceptance:
            alpha = np.mean(self.alpha.data[mask])

        info["alpha"] = alpha

        info["sqrt_ts"] = WStatCountsStatistic(
            info["counts"], info["counts_off"], acceptance / acceptance_off,
        ).sqrt_ts
        info["stat_sum"] = self.stat_sum()
        return info

    def to_spectrum_dataset(self, on_region, containment_correction=False, name=None):
        """Return a ~gammapy.datasets.SpectrumDatasetOnOff from on_region.

        Counts and OFF counts are summed in the on_region.

        Acceptance is the average of all acceptances while acceptance OFF
        is taken such that number of excess is preserved in the on_region.

        Effective area is taken from the average exposure divided by the livetime.
        Here we assume it is the sum of the GTIs.

        The energy dispersion kernel is obtained at the on_region center.
        Only regions with centers are supported.

        The model is not exported to the ~gammapy.dataset.SpectrumDataset.
        It must be set after the dataset extraction.

        Parameters
        ----------
        on_region : `~regions.SkyRegion`
            the input ON region on which to extract the spectrum
        containment_correction : bool
            Apply containment correction for point sources and circular on regions
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `~gammapy.datasets.SpectrumDatasetOnOff`
            the resulting reduced dataset
        """
        from .spectrum import SpectrumDatasetOnOff

        dataset = super().to_spectrum_dataset(on_region, containment_correction, name)

        kwargs = {}
        if self.counts_off is not None:
            kwargs["counts_off"] = self.counts_off.get_spectrum(
                on_region, np.sum, weights=self.mask_safe
            )

        if self.acceptance is not None:
            kwargs["acceptance"] = self.acceptance.get_spectrum(
                on_region, np.mean, weights=self.mask_safe
            )
            norm = self.background.get_spectrum(
                on_region, np.sum, weights=self.mask_safe
            )
            acceptance_off = kwargs["acceptance"] * kwargs["counts_off"] / norm
            np.nan_to_num(acceptance_off.data, copy=False)
            kwargs["acceptance_off"] = acceptance_off

        return SpectrumDatasetOnOff.from_spectrum_dataset(dataset=dataset, **kwargs)

    def cutout(self, position, width, mode="trim", name=None):
        """Cutout map dataset.

        Parameters
        ----------
        position : `~astropy.coordinates.SkyCoord`
            Center position of the cutout region.
        width : tuple of `~astropy.coordinates.Angle`
            Angular sizes of the region in (lon, lat) in that specific order.
            If only one value is passed, a square region is extracted.
        mode : {'trim', 'partial', 'strict'}
            Mode option for Cutout2D, for details see `~astropy.nddata.utils.Cutout2D`.
        name : str
            Name of the new dataset.

        Returns
        -------
        cutout : `MapDatasetOnOff`
            Cutout map dataset.
        """
        cutout_kwargs = {
            "position": position,
            "width": width,
            "mode": mode,
            "name": name,
        }

        cutout_dataset = super().cutout(**cutout_kwargs)

        del cutout_kwargs["name"]

        if self.counts_off is not None:
            cutout_dataset.counts_off = self.counts_off.cutout(**cutout_kwargs)

        if self.acceptance is not None:
            cutout_dataset.acceptance = self.acceptance.cutout(**cutout_kwargs)

        if self.acceptance_off is not None:
            cutout_dataset.acceptance_off = self.acceptance_off.cutout(**cutout_kwargs)

        return cutout_dataset

    def downsample(self, factor, axis_name=None, name=None):
        """Downsample map dataset.

        The PSFMap and EDispKernelMap are not downsampled, except if
        a corresponding axis is given.

        Parameters
        ----------
        factor : int
            Downsampling factor.
        axis_name : str
            Which non-spatial axis to downsample. By default only spatial axes are downsampled.
        name : str
            Name of the downsampled dataset.

        Returns
        -------
        dataset : `MapDatasetOnOff`
            Downsampled map dataset.
        """

        dataset = super().downsample(factor, axis_name, name)

        counts_off = None
        if self.counts_off is not None:
            counts_off = self.counts_off.downsample(
                factor=factor,
                preserve_counts=True,
                axis_name=axis_name,
                weights=self.mask_safe,
            )

        acceptance, acceptance_off = None, None
        if self.acceptance_off is not None:
            acceptance = self.acceptance.downsample(
                factor=factor, preserve_counts=False, axis_name=axis_name
            )
            factor = self.background.downsample(
                factor=factor,
                preserve_counts=True,
                axis_name=axis_name,
                weights=self.mask_safe,
            )
            acceptance_off = acceptance * counts_off / factor

        return self.__class__.from_map_dataset(
            dataset,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            counts_off=counts_off,
        )

    def pad(self):
        raise NotImplementedError

    def slice_by_idx(self, slices, name=None):
        """Slice sub dataset.

        The slicing only applies to the maps that define the corresponding axes.

        Parameters
        ----------
        slices : dict
            Dict of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.
        name : str
            Name of the sliced dataset.

        Returns
        -------
        map_out : `Map`
            Sliced map object.
        """
        kwargs = {"name": name}
        dataset = super().slice_by_idx(slices, name)

        if self.counts_off is not None:
            kwargs["counts_off"] = self.counts_off.slice_by_idx(slices=slices)

        if self.acceptance is not None:
            kwargs["acceptance"] = self.acceptance.slice_by_idx(slices=slices)

        if self.acceptance_off is not None:
            kwargs["acceptance_off"] = self.acceptance_off.slice_by_idx(slices=slices)

        return self.from_map_dataset(dataset, **kwargs)

    def resample_energy_axis(self, energy_axis, name=None):
        """Resample MapDatasetOnOff over reconstructed energy edges.

        Counts are summed taking into account safe mask.

        Parameters
        ----------
        energy_axis : `~gammapy.maps.MapAxis`
            New reco energy axis.
        name: str
            Name of the new dataset.

        Returns
        -------
        dataset: `SpectrumDataset`
            Resampled spectrum dataset .
        """
        dataset = super().resample_energy_axis(energy_axis, name)

        counts_off = None
        if self.counts_off is not None:
            counts_off = self.counts_off
            counts_off = counts_off.resample_axis(
                axis=energy_axis, weights=self.mask_safe
            )

        acceptance = 1
        acceptance_off = None
        if self.acceptance is not None:
            acceptance = self.acceptance
            acceptance = acceptance.resample_axis(
                axis=energy_axis, weights=self.mask_safe
            )

            norm_factor = self.background.resample_axis(
                axis=energy_axis, weights=self.mask_safe
            )

            acceptance_off = acceptance * counts_off / norm_factor

        return self.__class__.from_map_dataset(
            dataset,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            counts_off=counts_off,
        )


class MapEvaluator:
    """Sky model evaluation on maps.

    This evaluates a sky model on a 3D map and convolves with the IRFs,
    and returns a map of the predicted counts.
    Note that background counts are not added.

    For now, we only make it work for 3D WCS maps with an energy axis.
    No HPX, no other axes, those can be added later here or via new
    separate model evaluator classes.

    Parameters
    ----------
    model : `~gammapy.modeling.models.SkyModel`
        Sky model
    exposure : `~gammapy.maps.Map`
        Exposure map
    psf : `~gammapy.irf.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    mask : `~gammapy.maps.Map`
        Mask to apply to the likelihood for fitting.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    evaluation_mode : {"local", "global"}
        Model evaluation mode.
        The "local" mode evaluates the model components on smaller grids to save computation time.
        This mode is recommended for local optimization algorithms.
        The "global" evaluation mode evaluates the model components on the full map.
        This mode is recommended for global optimization algorithms.
    use_cache : bool
        Use npred caching.
    """

    def __init__(
        self,
        model=None,
        exposure=None,
        psf=None,
        edisp=None,
        gti=None,
        mask=None,
        evaluation_mode="local",
        use_cache=True,
    ):

        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp
        self.mask = mask
        self.gti = gti
        self.use_cache = use_cache
        self._init_position = None
        self.contributes = True
        self.psf_containment = None

        if evaluation_mode not in {"local", "global"}:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode!r}")

        self.evaluation_mode = evaluation_mode

        # TODO: this is preliminary solution until we have further unified the model handling
        if isinstance(self.model, BackgroundModel) or self.model.spatial_model is None or self.model.evaluation_radius is None:
            self.evaluation_mode = "global"

        # define cached computations
        self._compute_npred = lru_cache()(self._compute_npred)
        self._compute_npred_psf_after_edisp = lru_cache()(
            self._compute_npred_psf_after_edisp
        )
        self._compute_flux_spatial = lru_cache()(self._compute_flux_spatial)
        self._cached_parameter_values = None
        self._cached_parameter_values_spatial = None

    # workaround for the lru_cache pickle issue
    # see e.g. https://github.com/cloudpipe/cloudpickle/issues/178
    def __getstate__(self):
        state = self.__dict__.copy()
        for key, value in state.items():
            func = getattr(value, "__wrapped__", None)
            if func is not None:
                state[key] = func

        return state

    def __setstate__(self, state):
        for key, value in state.items():
            if key in ["_compute_npred", "_compute_flux_spatial", "_compute_npred_psf_after_edisp"]:
                state[key] = lru_cache()(value)

        self.__dict__ = state

    @property
    def geom(self):
        """True energy map geometry (`~gammapy.maps.Geom`)"""
        return self.exposure.geom

    @property
    def needs_update(self):
        """Check whether the model component has drifted away from its support."""
        # TODO: simplify and clean up
        if isinstance(self.model, BackgroundModel):
            return False
        elif self.exposure is None:
            return True
        elif self.geom.is_region:
            return False
        elif self.evaluation_mode == "global" or self.model.evaluation_radius is None:
            return False
        else:
            position = self.model.position
            separation = self._init_position.separation(position)
            update = separation > (self.model.evaluation_radius + CUTOUT_MARGIN)

        return update

    @property
    def psf_width(self):
        """Width of the PSF"""
        if self.psf is not None:
            psf_width = np.max(self.psf.psf_kernel_map.geom.width)
        else:
            psf_width = 0 * u.deg
        return psf_width

    def use_psf_containment(self, geom):
        """Use psf containment for point sources and circular regions"""
        if not geom.is_region:
            return False

        is_point_model = isinstance(self.model.spatial_model, PointSpatialModel)
        is_circle_region = isinstance(geom.region, CircleSkyRegion)
        return is_point_model & is_circle_region

    @property
    def cutout_width(self):
        """Cutout width for the model component"""
        return self.psf_width + 2 * (self.model.evaluation_radius + CUTOUT_MARGIN)

    def update(self, exposure, psf, edisp, geom, mask):
        """Update MapEvaluator, based on the current position of the model component.

        Parameters
        ----------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        psf : `gammapy.irf.PSFMap`
            PSF map.
        edisp : `gammapy.irf.EDispMap`
            Edisp map.
        geom : `WcsGeom`
            Counts geom
        mask : `~gammapy.maps.Map`
            Mask to apply to the likelihood for fitting.
        """
        # TODO: simplify and clean up
        log.debug("Updating model evaluator")

        # lookup edisp
        if edisp:
            energy_axis = geom.axes["energy"]
            self.edisp = edisp.get_edisp_kernel(
                self.model.position, energy_axis=energy_axis
            )

        # lookup psf
        if psf and self.model.spatial_model:
            if self.apply_psf_after_edisp:
                geom = geom.as_energy_true
            else:
                geom = exposure.geom

            if self.use_psf_containment(geom=geom):
                energy_true = geom.axes["energy_true"].center.reshape((-1, 1, 1))
                self.psf_containment = psf.containment(
                    energy_true=energy_true, rad=geom.region.radius
                )
            else:
                if geom.is_region:
                    # here we just need to choose a large value, the size will be the rad max
                    geom = geom.to_wcs_geom(width_min="15 deg")

                self.psf = psf.get_psf_kernel(position=self.model.position, geom=geom)

        if self.evaluation_mode == "local":
            self._init_position = self.model.position
            self.contributes = self.model.contributes(
                mask=mask, margin=self.psf_width
            )

            if self.contributes:
                self.exposure = exposure.cutout(
                    position=self.model.position, width=self.cutout_width
                )
        else:
            self.exposure = exposure

        self._compute_npred.cache_clear()
        self._compute_flux_spatial.cache_clear()
        self._compute_npred_psf_after_edisp.cache_clear()

    def compute_dnde(self):
        """Compute model differential flux at map pixel centers.

        Returns
        -------
        model_map : `~gammapy.maps.Map`
            Sky cube with data filled with evaluated model values.
            Units: ``cm-2 s-1 TeV-1 deg-2``
        """
        return self.model.evaluate_geom(self.geom, self.gti)

    def compute_flux(self):
        """Compute flux"""
        return self.model.integrate_geom(self.geom, self.gti)

    def compute_flux_psf_convolved(self):
        """Compute psf convolved and temporal model corrected flux."""
        value = self.compute_flux_spectral()

        if self.model.spatial_model:
            if self.psf_containment is not None:
                value = value * self.psf_containment
            else:
                value = value * self.compute_flux_spatial()

        if self.model.temporal_model:
            value *= self.compute_temporal_norm()

        return Map.from_geom(geom=self.geom, data=value.value, unit=value.unit)

    def _compute_flux_spatial(self):
        """Compute spatial flux

        Returns
        ----------
        value: `~astropy.units.Quantity`
            Psf-corrected, integrated flux over a given region.
        """
        if self.geom.is_region:
            if self.geom.region is None:
                return 1

            wcs_geom = self.geom.to_wcs_geom(width_min=self.cutout_width).to_image()
            values = self.model.spatial_model.integrate_geom(wcs_geom)

            if self.psf and self.model.apply_irf["psf"]:
                values = self.apply_psf(values)

            weights = wcs_geom.region_weights(regions=[self.geom.region])
            value = (values.quantity * weights).sum(axis=(1, 2), keepdims=True)

        else:
            value = self.model.spatial_model.integrate_geom(self.geom)
            if self.psf and self.model.apply_irf["psf"]:
                value = self.apply_psf(value)

        return value

    def compute_flux_spatial(self):
        """Compute spatial flux using caching"""
        if self.parameters_spatial_changed or not self.use_cache:
            self._compute_flux_spatial.cache_clear()
        return self._compute_flux_spatial()

    def compute_flux_spectral(self):
        """Compute spectral flux"""
        energy = self.geom.axes["energy_true"].edges
        value = self.model.spectral_model.integral(energy[:-1], energy[1:],)
        return value.reshape((-1, 1, 1))

    def compute_temporal_norm(self):
        """Compute temporal norm """
        integral = self.model.temporal_model.integral(
            self.gti.time_start, self.gti.time_stop
        )
        return np.sum(integral)

    def apply_exposure(self, flux):
        """Compute npred cube

        For now just divide flux cube by exposure
        """
        npred = (flux.quantity * self.exposure.quantity).to_value("")
        return Map.from_geom(self.geom, data=npred, unit="")

    def apply_psf(self, npred):
        """Convolve npred cube with PSF"""
        tmp = npred.convolve(self.psf)
        tmp.data[tmp.data < 0.0] = 0
        return tmp

    def apply_edisp(self, npred):
        """Convolve map data with energy dispersion.

        Parameters
        ----------
        npred : `~gammapy.maps.Map`
            Predicted counts in true energy bins

        Returns
        -------
        npred_reco : `~gammapy.maps.Map`
            Predicted counts in reco energy bins
        """
        return npred.apply_edisp(self.edisp)

    def _compute_npred(self):
        """Compute npred"""
        if isinstance(self.model, BackgroundModel):
            npred = self.model.evaluate()
        else:
            npred = self.compute_flux_psf_convolved()

            if self.model.apply_irf["exposure"]:
                npred = self.apply_exposure(npred)

            if self.model.apply_irf["edisp"]:
                npred = self.apply_edisp(npred)

        return npred

    @property
    def apply_psf_after_edisp(self):
        """"""
        if not isinstance(self.model, BackgroundModel):
            return self.model.apply_irf.get("psf_after_edisp")

    # TODO: remove again if possible...
    def _compute_npred_psf_after_edisp(self):
        if isinstance(self.model, BackgroundModel):
            return self.model.evaluate()

        npred = self.compute_flux()

        if self.model.apply_irf["exposure"]:
            npred = self.apply_exposure(npred)

        if self.model.apply_irf["edisp"]:
            npred = self.apply_edisp(npred)

        if self.model.apply_irf["psf"]:
            npred = self.apply_psf(npred)

        return npred

    def compute_npred(self):
        """Evaluate model predicted counts.

        Returns
        -------
        npred : `~gammapy.maps.Map`
            Predicted counts on the map (in reco energy bins)
        """
        if self.apply_psf_after_edisp:
            if self.parameters_changed or not self.use_cache:
                self._compute_npred_psf_after_edisp.cache_clear()

            return self._compute_npred_psf_after_edisp()

        if self.parameters_changed or not self.use_cache:
            self._compute_npred.cache_clear()

        return self._compute_npred()

    @property
    def parameters_changed(self):
        """Parameters changed"""
        values = self.model.parameters.value

        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._cached_parameter_values == values)

        if changed:
            self._cached_parameter_values = values

        return changed

    @property
    def parameters_spatial_changed(self):
        """Parameters changed"""
        values = self.model.spatial_model.parameters.value

        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._cached_parameter_values_spatial == values)

        if changed:
            self._cached_parameter_values_spatial = values

        return changed
