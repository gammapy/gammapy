# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from functools import lru_cache
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.nddata.utils import NoOverlapError
from astropy.table import Table
from astropy.utils import lazyproperty
from regions import CircleSkyRegion, RectangleSkyRegion
from gammapy.data import GTI
from gammapy.irf import EDispKernel
from gammapy.irf.edisp_map import EDispMap, EDispKernelMap
from gammapy.irf.psf_kernel import PSFKernel
from gammapy.irf.psf_map import PSFMap
from gammapy.maps import Map, MapAxis, RegionGeom
from gammapy.modeling.models import (
    BackgroundModel,
    Models,
    ProperModels,
)
from gammapy.stats import cash, cash_sum_cython, wstat, get_wstat_mu_bkg
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name, make_path
from gammapy.utils.fits import LazyFitsData, HDULocation
from gammapy.utils.table import hstack_columns
from .core import Dataset

__all__ = ["MapDataset", "MapDatasetOnOff", "create_map_dataset_geoms"]

log = logging.getLogger(__name__)

CUTOUT_MARGIN = 0.1 * u.deg
RAD_MAX = 0.66
RAD_AXIS_DEFAULT = MapAxis.from_bounds(
    0, RAD_MAX, nbin=66, node_type="edges", name="theta", unit="deg"
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
        if energy_axis_true.name != "energy_true":
            raise ValueError("True enery axis name must be 'energy_true'")
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
        geom_edisp = geom_irf.to_cube(
            [geom.axes["energy"], energy_axis_true]
        )

    return {
        "geom": geom,
        "geom_exposure": geom_exposure,
        "geom_psf": geom_psf,
        "geom_edisp": geom_edisp,
    }


class MapDataset(Dataset):
    """Perform sky model likelihood fit on maps.

    Parameters
    ----------
    models : `~gammapy.modeling.models.Models`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    mask_fit : `~gammapy.maps.WcsNDMap`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.irf.PSFKernel` or `~gammapy.irf.PSFMap`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel` or `~gammapy.irf.EDispMap`
        Energy dispersion kernel
    mask_safe : `~gammapy.maps.WcsNDMap`
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
    psf = LazyFitsData(cache=True)
    mask_fit = LazyFitsData(cache=True)
    mask_safe = LazyFitsData(cache=True)

    _lazy_data_members = ["counts", "exposure", "edisp", "psf", "mask_fit", "mask_safe"]

    def __init__(
        self,
        models=None,
        counts=None,
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
        self._background_model = None
        self.counts = counts
        self.exposure = exposure
        self.mask_fit = mask_fit
        self.psf = psf

        if isinstance(edisp, EDispKernel):
            edisp = EDispKernelMap.from_edisp_kernel(edisp=edisp)

        self.edisp = edisp
        self.mask_safe = mask_safe
        self.models = models
        self.gti = gti
        self.meta_table = meta_table

    @property
    def name(self):
        return self._name

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "-" * len(self.__class__.__name__) + "\n"
        str_ += "\n"

        str_ += "\t{:32}: {} \n\n".format("Name", self.name)

        counts = np.nan
        if self.counts is not None:
            counts = np.sum(self.counts.data)
        str_ += "\t{:32}: {:.0f} \n".format("Total counts", counts)

        npred = np.nan
        if self.models is not None:
            npred = np.sum(self.npred().data)
        str_ += "\t{:32}: {:.2f}\n".format("Total predicted counts", npred)

        background = np.nan
        if self.background_model is not None:
            background = np.sum(self.background_model.evaluate().data)
        str_ += "\t{:32}: {:.2f}\n\n".format("Total background counts", background)

        exposure_min, exposure_max, exposure_unit = np.nan, np.nan, ""
        if self.exposure is not None:
            if self.mask_safe is not None:
                mask = self.mask_safe.reduce_over_axes(np.logical_or).data
                if not mask.any():
                    mask = None
            else:
                mask = None
            exposure_min = np.min(self.exposure.data[..., mask])
            exposure_max = np.max(self.exposure.data[..., mask])
            exposure_unit = self.exposure.unit

        str_ += "\t{:32}: {:.2e} {}\n".format(
            "Exposure min", exposure_min, exposure_unit
        )
        str_ += "\t{:32}: {:.2e} {}\n\n".format(
            "Exposure max", exposure_max, exposure_unit
        )

        # data section
        n_bins = 0
        if self.counts is not None:
            n_bins = self.counts.data.size
        str_ += "\t{:32}: {} \n".format("Number of total bins", n_bins)

        n_fit_bins = 0
        if self.mask is not None:
            n_fit_bins = np.sum(self.mask.data)
        str_ += "\t{:32}: {} \n\n".format("Number of fit bins", n_fit_bins)

        # likelihood section
        str_ += "\t{:32}: {}\n".format("Fit statistic type", self.stat_type)

        stat = np.nan
        if self.counts is not None and self.models is not None:
            stat = self.stat_sum()
        str_ += "\t{:32}: {:.2f}\n\n".format("Fit statistic value (-2 log(L))", stat)

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
    def models(self):
        """Models (`~gammapy.modeling.models.Models`)."""
        return ProperModels(self)

    @property
    def background_model(self):
        return self._background_model

    @models.setter
    def models(self, models):
        if models is None:
            self._models = None
        else:
            self._models = Models(models)

        # TODO: clean this up (probably by removing)
        for model in self.models:
            if isinstance(model, BackgroundModel):
                if model.datasets_names is not None:
                    if self.name in model.datasets_names:
                        self._background_model = model
                        break
        else:
            if not isinstance(self, MapDatasetOnOff):
                log.warning(f"No background model defined for dataset {self.name}")
        self._evaluators = {}

    @property
    def evaluators(self):
        """Model evaluators"""

        models = self.models
        if models:
            keys = list(self._evaluators.keys())
            for key in keys:
                if key not in models:
                    del self._evaluators[key]

            for model in models:
                evaluator = self._evaluators.get(model)

                if evaluator is None:
                    evaluator = MapEvaluator(
                        model=model,
                        evaluation_mode=EVALUATION_MODE,
                        gti=self.gti,
                        use_cache=USE_NPRED_CACHE,
                    )
                    self._evaluators[model] = evaluator

                # if the model component drifts out of its support the evaluator has
                # has to be updated
                if evaluator.needs_update:
                    evaluator.update(self.exposure, self.psf, self.edisp, self._geom)

        return self._evaluators

    @property
    def _geom(self):
        """Main analysis geometry"""
        if self.counts is not None:
            return self.counts.geom
        elif self.background_model is not None:
            return self.background_model.map.geom
        elif self.mask_safe is not None:
            return self.mask_safe.geom
        elif self.mask_fit is not None:
            return self.mask_fit.geom
        else:
            raise ValueError(
                "Either 'counts', 'background_model', 'mask_fit'"
                " or 'mask_safe' must be defined."
            )

    @property
    def data_shape(self):
        """Shape of the counts or background data (tuple)"""
        return self._geom.data_shape

    def npred(self):
        """Predicted source and background counts (`~gammapy.maps.Map`)."""
        npred_total = Map.from_geom(self._geom, dtype=float)

        for evaluator in self.evaluators.values():
            if evaluator.contributes:
                npred = evaluator.compute_npred()
                npred_total.stack(npred)
        return npred_total

    @classmethod
    def from_geoms(
        cls,
        geom,
        geom_exposure,
        geom_psf,
        geom_edisp,
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
        empty_maps : `MapDataset`
            A MapDataset containing zero filled maps
        """
        name = make_name(name)
        kwargs = kwargs.copy()
        kwargs["name"] = name
        kwargs["counts"] = Map.from_geom(geom, unit="")

        background = Map.from_geom(geom, unit="")
        kwargs["models"] = Models(
            [BackgroundModel(background, name=name + "-bkg", datasets_names=[name])]
        )
        kwargs["exposure"] = Map.from_geom(geom_exposure, unit="m2 s")

        if geom_edisp.axes[0].name.lower() == "energy":
            kwargs["edisp"] = EDispKernelMap.from_geom(geom_edisp)
        else:
            kwargs["edisp"] = EDispMap.from_geom(geom_edisp)

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

    def stack(self, other):
        """Stack another dataset in place.

        Parameters
        ----------
        other: `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            Map dataset to be stacked with this one. If other is an on-off
            dataset alpha * counts_off is used as a background model.
        """
        if self.mask_safe is None:
            self.mask_safe = Map.from_geom(
                self._geom, data=np.ones_like(self.data_shape)
            )

        if other.mask_safe is None:
            other_mask_safe = Map.from_geom(
                other._geom, data=np.ones_like(other.data_shape)
            )
        else:
            other_mask_safe = other.mask_safe

        if self.counts and other.counts:
            self.counts *= self.mask_safe
            self.counts.stack(other.counts, weights=other_mask_safe)

        if self.exposure and other.exposure:
            mask_exposure = self._mask_safe_irf(self.exposure, self.mask_safe)
            self.exposure *= mask_exposure.data

            mask_exposure_other = self._mask_safe_irf(other.exposure, other_mask_safe)
            self.exposure.stack(other.exposure, weights=mask_exposure_other)

        # TODO: unify background model handling
        if other.stat_type == "wstat":
            background_model = BackgroundModel(other.background)
        else:
            background_model = other.background_model

        if self.background_model and background_model:
            self._background_model.map *= self.mask_safe
            self._background_model.stack(background_model, other_mask_safe)
            self.models = Models([self.background_model])
        else:
            self.models = None

        if self.psf and other.psf:
            if isinstance(self.psf, PSFMap) and isinstance(other.psf, PSFMap):
                mask_irf = self._mask_safe_irf(
                    self.psf.exposure_map, self.mask_safe, drop="theta"
                )
                self.psf.psf_map.data *= mask_irf.data
                self.psf.exposure_map.data *= mask_irf.data

                mask_irf_other = self._mask_safe_irf(
                    other.psf.exposure_map, other_mask_safe, drop="theta"
                )
                self.psf.stack(other.psf, weights=mask_irf_other)
            else:
                raise ValueError("Stacking of PSF kernels not supported")

        if self.edisp and other.edisp:
            if isinstance(self.edisp, EDispKernelMap) and isinstance(
                other.edisp, EDispKernelMap
            ):
                mask_irf = self._mask_safe_irf(
                    self.edisp.edisp_map, self.mask_safe, drop="energy_true"
                )
                mask_irf_other = self._mask_safe_irf(
                    other.edisp.edisp_map, other_mask_safe, drop="energy_true"
                )

            if isinstance(self.edisp, EDispMap) and isinstance(other.edisp, EDispMap):
                mask_irf = self._mask_safe_irf(
                    self.edisp.exposure_map.sum_over_axes(),
                    self.mask_safe.reduce_over_axes(func=np.logical_or, keepdims=True),
                )
                mask_irf_other = self._mask_safe_irf(
                    other.edisp.exposure_map.sum_over_axes(),
                    other_mask_safe.reduce_over_axes(func=np.logical_or, keepdims=True),
                )

            self.edisp.edisp_map.data *= mask_irf.data
            # Question: Should mask be applied on exposure map as well?
            # Mask here is on the reco energy.
            # self.edisp.exposure_map.data *= mask_irf.data

            self.edisp.stack(other.edisp, weights=mask_irf_other)

        self.mask_safe.stack(other_mask_safe)

        if self.gti and other.gti:
            self.gti.stack(other.gti)
            self.gti = self.gti.union()

        if self.meta_table and other.meta_table:
            self.meta_table = hstack_columns(self.meta_table, other.meta_table)
        elif other.meta_table:
            self.meta_table = other.meta_table.copy()

    @staticmethod
    def _mask_safe_irf(irf_map, mask, drop=None):
        if mask is None:
            return None

        geom = irf_map.geom
        geom_squash = irf_map.geom
        if drop:
            geom = geom.drop(drop)
            geom_squash = geom_squash.squash(drop)

        if "energy_true" in geom.axes.names:
            ax = geom.axes["energy_true"].copy(name="energy")
            geom = geom.to_image().to_cube([ax])

        coords = geom.get_coord()
        data = mask.get_by_coord(coords).astype(bool)
        return Map.from_geom(geom=geom_squash, data=data[:, np.newaxis])

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def residuals(self, method="diff"):
        """Compute residuals map.

        Parameters
        ----------
        method: {"diff", "diff/model", "diff/sqrt(model)"}
            Method used to compute the residuals. Available options are:
                - "diff" (default): data - model
                - "diff/model": (data - model) / model
                - "diff/sqrt(model)": (data - model) / sqrt(model)

        Returns
        -------
        residuals : `gammapy.maps.WcsNDMap`
            Residual map.
        """
        npred = self.npred()
        if isinstance(self, MapDatasetOnOff):
            npred += self.background
        return self._compute_residuals(self.counts, npred, method=method)

    def plot_residuals(
        self,
        method="diff",
        smooth_kernel="gauss",
        smooth_radius="0.1 deg",
        region=None,
        figsize=(12, 4),
        **kwargs,
    ):
        """
        Plot spatial and spectral residuals.

        The spectral residuals are extracted from the provided region, and the
        normalization used for the residuals computation can be controlled using
        the method parameter. If no region is passed, only the spatial
        residuals are shown.

        Parameters
        ----------
        method : {"diff", "diff/model", "diff/sqrt(model)"}
            Method used to compute the residuals, see `MapDataset.residuals()`
        smooth_kernel : {'gauss', 'box'}
            Kernel shape.
        smooth_radius: `~astropy.units.Quantity`, str or float
            Smoothing width given as quantity or float. If a float is given it
            is interpreted as smoothing width in pixels.
        region: `~regions.Region`
            Region (pixel or sky regions accepted)
        figsize : tuple
            Figure size used for the plotting.
        **kwargs : dict
            Keyword arguments passed to `~matplotlib.pyplot.imshow`.

        Returns
        -------
        ax_image, ax_spec : `~matplotlib.pyplot.Axes`,
            Image and spectrum axes.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize)

        counts, npred = self.counts, self.npred()

        if isinstance(self, MapDatasetOnOff):
            npred += self.background

        if self.mask is not None:
            counts = counts * self.mask
            npred = npred * self.mask

        counts_spatial = counts.sum_over_axes().smooth(
            width=smooth_radius, kernel=smooth_kernel
        )
        npred_spatial = npred.sum_over_axes().smooth(
            width=smooth_radius, kernel=smooth_kernel
        )
        spatial_residuals = self._compute_residuals(
            counts_spatial, npred_spatial, method
        )

        if self.mask_safe is not None:
            mask = self.mask_safe.reduce_over_axes(func=np.logical_or, keepdims=True)
            spatial_residuals.data[~mask.data] = np.nan

        # If no region is provided, skip spectral residuals
        ncols = 2 if region is not None else 1
        ax_image = fig.add_subplot(1, ncols, 1, projection=spatial_residuals.geom.wcs)
        ax_spec = None

        kwargs.setdefault("cmap", "coolwarm")
        kwargs.setdefault("stretch", "linear")
        kwargs.setdefault("vmin", -5)
        kwargs.setdefault("vmax", 5)
        spatial_residuals.plot(ax=ax_image, add_cbar=True, **kwargs)

        # Spectral residuals
        if region:
            ax_spec = fig.add_subplot(1, 2, 2)
            counts_spec = counts.get_spectrum(region=region)
            npred_spec = npred.get_spectrum(region=region)
            residuals = self._compute_residuals(counts_spec, npred_spec, method)
            if method == "diff":
                yerr = np.sqrt((counts_spec.data + npred_spec.data).flatten())
            else:
                yerr = np.ones_like(residuals.data.flatten())
            ax = residuals.plot(color="black", yerr=yerr, fmt=".", capsize=2, lw=1)
            ax.set_yscale("linear")
            ax.axhline(0, color="black", lw=0.5)
            ymax = 1.05 * np.nanmax(residuals.data + yerr.data)
            ymin = 1.05 * np.nanmin(residuals.data - yerr.data)
            plt.ylim(ymin, ymax)
            label = self._residuals_labels[method]
            plt.ylabel(f"Residuals ({label})")

            # Overlay spectral extraction region on the spatial residuals
            pix_region = region.to_pixel(wcs=spatial_residuals.geom.wcs)
            pix_region.plot(ax=ax_image)

        return ax_image, ax_spec

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

        if self.background_model is not None:
            hdulist += self.background_model.map.to_hdulist(hdu="background")[
                exclude_primary
            ]

        if self.edisp is not None:
            if isinstance(self.edisp, EDispKernel):
                hdus = self.edisp.to_hdulist()
                hdus["MATRIX"].name = "edisp_matrix"
                hdus["EBOUNDS"].name = "edisp_matrix_ebounds"
                hdulist.append(hdus["EDISP_MATRIX"])
                hdulist.append(hdus["EDISP_MATRIX_EBOUNDS"])
            else:
                hdulist += self.edisp.edisp_map.to_hdulist(hdu="EDISP")[exclude_primary]
                hdulist += self.edisp.exposure_map.to_hdulist(hdu="edisp_exposure")[
                    exclude_primary
                ]

        if self.psf is not None:
            if isinstance(self.psf, PSFKernel):
                hdulist += self.psf.psf_kernel_map.to_hdulist(hdu="psf_kernel")[
                    exclude_primary
                ]
            else:
                hdulist += self.psf.psf_map.to_hdulist(hdu="psf")[exclude_primary]
                hdulist += self.psf.exposure_map.to_hdulist(hdu="psf_exposure")[
                    exclude_primary
                ]

        if self.mask_safe is not None:
            mask_safe_int = self.mask_safe.copy()
            mask_safe_int.data = mask_safe_int.data.astype(int)
            hdulist += mask_safe_int.to_hdulist(hdu="mask_safe")[exclude_primary]

        if self.mask_fit is not None:
            mask_fit_int = self.mask_fit.copy()
            mask_fit_int.data = mask_fit_int.data.astype(int)
            hdulist += mask_fit_int.to_hdulist(hdu="mask_fit")[exclude_primary]

        if self.gti is not None:
            hdulist.append(fits.BinTableHDU(self.gti.table, name="GTI"))

        return hdulist

    @classmethod
    def from_hdulist(cls, hdulist, name=None, lazy=False):
        """Create map dataset from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        name = make_name(name)
        kwargs = {"name": name}

        if "COUNTS" in hdulist:
            kwargs["counts"] = Map.from_hdulist(hdulist, hdu="counts")

        if "EXPOSURE" in hdulist:
            exposure = Map.from_hdulist(hdulist, hdu="exposure")
            if exposure.geom.axes[0].name == "energy":
                exposure.geom.axes[0].name = "energy_true"
            kwargs["exposure"] = exposure

        if "BACKGROUND" in hdulist:
            background_map = Map.from_hdulist(hdulist, hdu="background")
            kwargs["models"] = Models(
                [
                    BackgroundModel(
                        background_map, datasets_names=[name], name=name + "-bkg"
                    )
                ]
            )

        if "EDISP_MATRIX" in hdulist:
            kwargs["edisp"] = EDispKernel.from_hdulist(
                hdulist, hdu1="EDISP_MATRIX", hdu2="EDISP_MATRIX_EBOUNDS"
            )
        if "EDISP" in hdulist:
            edisp_map = Map.from_hdulist(hdulist, hdu="edisp")
            try:
                exposure_map = Map.from_hdulist(hdulist, hdu="edisp_exposure")
            except KeyError:
                exposure_map = None
            if edisp_map.geom.axes[0].name == "energy":
                kwargs["edisp"] = EDispKernelMap(edisp_map, exposure_map)
            else:
                kwargs["edisp"] = EDispMap(edisp_map, exposure_map)

        if "PSF_KERNEL" in hdulist:
            psf_map = Map.from_hdulist(hdulist, hdu="psf_kernel")
            kwargs["psf"] = PSFKernel(psf_map)

        if "PSF" in hdulist:
            psf_map = Map.from_hdulist(hdulist, hdu="psf")
            try:
                exposure_map = Map.from_hdulist(hdulist, hdu="psf_exposure")
            except KeyError:
                exposure_map = None
            kwargs["psf"] = PSFMap(psf_map, exposure_map)

        if "MASK_SAFE" in hdulist:
            mask_safe = Map.from_hdulist(hdulist, hdu="mask_safe")
            mask_safe.data = mask_safe.data.astype(bool)
            kwargs["mask_safe"] = mask_safe

        if "MASK_FIT" in hdulist:
            mask_fit = Map.from_hdulist(hdulist, hdu="mask_fit")
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
    def _read_lazy(cls, name, filename, cache):
        kwargs = {"name": name}
        try:
            kwargs["gti"] = GTI.read(filename)
        except KeyError:
            pass

        path = make_path(filename)
        for hdu_name in ["counts", "exposure", "mask_fit", "mask_safe"]:
            kwargs[hdu_name] = HDULocation(
                hdu_class="map",
                file_dir=path.parent,
                file_name=path.name,
                hdu_name=hdu_name.upper(),
                cache=cache,
            )

        kwargs["edisp"] = HDULocation(
            hdu_class="edisp_kernel_map",
            file_dir=path.parent,
            file_name=path.name,
            hdu_name="EDISP",
            cache=cache,
        )

        kwargs["psf"] = HDULocation(
            hdu_class="psf_map",
            file_dir=path.parent,
            file_name=path.name,
            hdu_name="PSF",
            cache=cache,
        )

        hduloc = HDULocation(
            hdu_class="map",
            file_dir=path.parent,
            file_name=path.name,
            hdu_name="BACKGROUND",
            cache=cache,
        )

        kwargs["models"] = [
            BackgroundModel(hduloc, datasets_names=[name], name=name + "-bkg")
        ]

        return cls(**kwargs)

    @classmethod
    def read(cls, filename, name=None, lazy=False, cache=True):
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

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        name = make_name(name)

        if lazy:
            return cls._read_lazy(name=name, filename=filename, cache=cache)
        else:
            with fits.open(str(make_path(filename)), memmap=False) as hdulist:
                return cls.from_hdulist(hdulist, name=name)

    @classmethod
    def from_dict(cls, data, models, lazy=False, cache=True):
        """Create from dicts and models list generated from YAML serialization."""

        # TODO: remove handling models here
        filename = make_path(data["filename"])
        dataset = cls.read(filename, name=data["name"], lazy=lazy, cache=cache)

        for model in models:
            if (
                isinstance(model, BackgroundModel)
                and model.filename is None
                and dataset.name == model.datasets_names[0]
            ):
                model.map = dataset.background_model.map

        dataset.models = models
        return dataset

    def to_dict(self, filename=""):
        """Convert to dict for YAML serialization."""
        return {"name": self.name, "type": self.tag, "filename": str(filename)}

    def info_dict(self, region=None):
        """Basic info dict with summary statistics

        If a region is passed, then a spectrum dataset is
        extracted, and the corresponding info returned.

        Parameters
        ----------
        region : `~regions.SkyRegion`, optional
            the input ON region on which to extract the spectrum

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        if self.gti is not None:
            if region is None:
                region = RectangleSkyRegion(
                    center=self._geom.center_skydir,
                    width=self._geom.width[0][0],
                    height=self._geom.width[1][0],
                )
            info = self.to_spectrum_dataset(on_region=region).info_dict()
        else:
            info = dict()
            if self.counts:
                info["counts"] = np.sum(self.counts.data)
            if self.background_model:
                info["background"] = np.sum(self.background_model.evaluate().data)
                info["excess"] = info["counts"] - info["background"]

            info["npred"] = np.sum(self.npred())
            if self.mask_safe is not None:
                mask = self.mask_safe.reduce_over_axes(np.logical_or).data
                if not mask.any():
                    mask = None
            else:
                mask = None
            if self.exposure:
                exposure_min = np.min(self.exposure.data[..., mask])
                exposure_max = np.max(self.exposure.data[..., mask])
                info["aeff_min"] = exposure_min * self.exposure.unit
                info["aeff_max"] = exposure_max * self.exposure.unit

        info["name"] = self.name

        return info

    def to_spectrum_dataset(self, on_region, containment_correction=False, name=None):
        """Return a ~gammapy.datasets.SpectrumDataset from on_region.

        Counts and background are summed in the on_region.

        Effective area is taken from the average exposure divided by the livetime.
        Here we assume it is the sum of the GTIs.

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

        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name}

        if self.gti is not None:
            kwargs["livetime"] = self.gti.time_sum
        else:
            raise ValueError("No GTI in `MapDataset`, cannot compute livetime")

        if self.counts is not None:
            kwargs["counts"] = self.counts.get_spectrum(on_region, np.sum)

        if self.background_model is not None:
            bkg = self.background_model.evaluate().get_spectrum(on_region, np.sum)
            bkg_model = BackgroundModel(bkg, name=name + "-bkg", datasets_names=[name])
            bkg_model.spectral_model.norm.frozen = True
            kwargs["models"] = Models([bkg_model])

        if self.exposure is not None:
            kwargs["aeff"] = (
                self.exposure.get_spectrum(on_region, np.mean) / kwargs["livetime"]
            )

        if containment_correction:
            if not isinstance(on_region, CircleSkyRegion):
                raise TypeError(
                    "Containement correction is only supported for"
                    " `CircleSkyRegion`."
                )
            elif self.psf is None or isinstance(self.psf, PSFKernel):
                raise ValueError("No PSFMap set. Containement correction impossible")
            else:
                psf = self.psf.get_energy_dependent_table_psf(on_region.center)
                energy = kwargs["aeff"].geom.axes["energy_true"].center
                containment = psf.containment(energy, on_region.radius)
                kwargs["aeff"].data *= containment[:, np.newaxis]

        if self.edisp is not None:
            energy_axis = self._geom.axes["energy"]
            edisp = self.edisp.get_edisp_kernel(
                on_region.center, energy_axis=energy_axis
            )

            edisp = EDispKernelMap.from_edisp_kernel(
                edisp=edisp, geom=RegionGeom(on_region)
            )
            kwargs["edisp"] = edisp

        return SpectrumDataset(**kwargs)

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
        kwargs = {"gti": self.gti, "name": name}
        cutout_kwargs = {"position": position, "width": width, "mode": mode}

        if self.counts is not None:
            kwargs["counts"] = self.counts.cutout(**cutout_kwargs)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.cutout(**cutout_kwargs)

        if self.background_model is not None:
            model = self.background_model.cutout(**cutout_kwargs, name=name + "-bkg")
            model.datasets_names = [name]
            kwargs["models"] = model

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
        dataset : `MapDataset`
            Downsampled map dataset.
        """
        name = make_name(name)

        kwargs = {"gti": self.gti, "name": name}

        if self.counts is not None:
            kwargs["counts"] = self.counts.downsample(
                factor=factor, preserve_counts=True, axis_name=axis_name, weights=self.mask_safe
            )

        if self.exposure is not None:
            if axis_name is None:
                kwargs["exposure"] = self.exposure.downsample(
                    factor=factor, preserve_counts=False
                )
            else:
                kwargs["exposure"] = self.exposure.copy()

        if self.background_model is not None:
            m = self.background_model.evaluate().downsample(
                factor=factor, axis_name=axis_name, weights=self.mask_safe
            )
            kwargs["models"] = BackgroundModel(map=m, datasets_names=[name])

        if self.edisp is not None:
            if axis_name is not None:
                kwargs["edisp"] = self.edisp.downsample(factor=factor, axis_name=axis_name)
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
        map : `Map`
            Padded map.

        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name}

        if self.counts is not None:
            kwargs["counts"] = self.counts.pad(pad_width=pad_width, mode=mode)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.pad(pad_width=pad_width, mode=mode)

        if self.background_model is not None:
            m = self.background_model.evaluate().pad(pad_width=pad_width, mode=mode)
            kwargs["models"] = BackgroundModel(map=m, datasets_names=[name])

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
        map_out : `Map`
            Sliced map object.
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name}

        if self.counts is not None:
            kwargs["counts"] = self.counts.slice_by_idx(slices=slices)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.slice_by_idx(slices=slices)

        if self.background_model is not None:
            m = self.background_model.evaluate().slice_by_idx(slices=slices)
            kwargs["models"] = BackgroundModel(map=m, datasets_names=[name])

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.slice_by_idx(slices=slices)

        if self.psf is not None:
            kwargs["psf"] = self.psf.slice_by_idx(slices=slices)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.slice_by_idx(slices=slices)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.slice_by_idx(slices=slices)

        return self.__class__(**kwargs)

    def reset_data_cache(self):
        """Reset data cache to free memory space"""
        for name in self._lazy_data_members:
            if self.__dict__.pop(name, False):
                log.info(f"Clearing {name} cache for dataset {self.name}")

    def resample_energy_axis(self, axis=None, name=None):
        """Resample MapDataset over new reco energy axis.

        Counts are summed taking into account safe mask.

        Parameters
        ----------
        axis : `~gammapy.maps.MapAxis`
            the new reco energy axis.
        name: str
            Name of the new dataset.

        Returns
        -------
        dataset: `MapDataset`
            Resampled dataset .
        """
        if axis is None:
            e_axis = self._geom.axes["energy"]
            e_edges = u.Quantity([e_axis.edges[0], e_axis.edges[-1]])
            axis = MapAxis.from_edges(e_edges, name="energy", interp=self._geom.axes[0].interp)

        name = make_name(name)
        kwargs = {}
        kwargs["name"] = name
        kwargs["gti"] = self.gti
        kwargs["exposure"] = self.exposure
        kwargs["psf"] = self.psf

        if self.mask_safe is not None:
            weights = self.mask_safe
            kwargs["mask_safe"] = self.mask_safe.resample_axis(axis=axis, ufunc=np.logical_or)
        else:
            weights = None

        if self.counts is not None:
            kwargs["counts"] = self.counts.resample_axis(axis=axis, weights=weights)

        if self.background_model is not None:
            background = self.background_model.evaluate()
            background = background.resample_axis(axis=axis, weights=weights)
            model = BackgroundModel(
                background, datasets_names=[name], name=f"{name}-bkg"
            )
            kwargs["models"] = [model]

        # Mask_safe or mask_irf??
        if isinstance(self.edisp, EDispKernelMap):
            mask_irf = self._mask_safe_irf(
                self.edisp.edisp_map, self.mask_safe, drop="energy_true"
            )
            kwargs["edisp"] = self.edisp.resample_axis(axis=axis, weights=mask_irf)
        else:  # None or EDispMap
            kwargs["edisp"] = self.edisp

        return self.__class__(**kwargs)

    def to_image(self, name=None):
        """Create images by summing over the reconstructed-energy axis.

        Parameters
        ----------
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset containing images.
        """
        return self.resample_energy_axis(axis=None, name=name)

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
    MapDatasetOn, SpectrumDataset, FluxPointsDataset

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
        self.counts = counts
        self.counts_off = counts_off
        self.exposure = exposure

        if np.isscalar(acceptance):
            acceptance = Map.from_geom(
                self._geom, data=np.ones(self.data_shape) * acceptance
            )

        if np.isscalar(acceptance_off):
            acceptance_off = Map.from_geom(
                self._geom, data=np.ones(self.data_shape) * acceptance_off
            )

        self.acceptance = acceptance
        self.acceptance_off = acceptance_off
        self._background_model = None
        self.mask_fit = mask_fit
        self.psf = psf
        self.edisp = edisp
        self._name = make_name(name)
        self.models = models
        self.mask_safe = mask_safe
        self.gti = gti
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

        return str_.expandtabs(tabsize=4)

    @property
    def alpha(self):
        """Exposure ratio between signal and background regions"""
        alpha = self.acceptance / self.acceptance_off
        alpha.data = np.nan_to_num(alpha.data)
        return alpha

    @property
    def background(self):
        """
        Background counts estimated from the marginalized likelihood estimate.
        See :ref:wstat.
        """
        mu_bkg = self.alpha.data * get_wstat_mu_bkg(
            n_on=self.counts.data,
            n_off=self.counts_off.data,
            alpha=self.alpha.data,
            mu_sig=self.npred().data,
        )
        mu_bkg = np.nan_to_num(mu_bkg)
        return Map.from_geom(geom=self._geom, data=mu_bkg)

    @property
    def counts_off_normalised(self):
        """ alpha * n_off"""
        return self.alpha * self.counts_off

    @property
    def excess(self):
        """Excess (counts - alpha * counts_off)"""
        return self.counts - self.counts_off_normalised

    def stat_array(self):
        """Likelihood per bin given the current model parameters"""
        mu_sig = self.npred().data
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
        geom_psf,
        geom_edisp,
        reference_time="2000-01-01",
        name=None,
        **kwargs,
    ):
        """
        Create a MapDatasetOnOff object with zero filled maps according to the specified geometries

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
        kwargs = kwargs.copy()
        kwargs["name"] = name

        for key in ["counts", "counts_off", "acceptance", "acceptance_off"]:
            kwargs[key] = Map.from_geom(geom, unit="")

        kwargs["exposure"] = Map.from_geom(geom_exposure, unit="m2 s")
        if geom_edisp.axes[0].name.lower() == "energy":
            kwargs["edisp"] = EDispKernelMap.from_geom(geom_edisp)
        else:
            kwargs["edisp"] = EDispMap.from_geom(geom_edisp)

        kwargs["psf"] = PSFMap.from_geom(geom_psf)
        kwargs["gti"] = GTI.create([] * u.s, [] * u.s, reference_time=reference_time)
        kwargs["mask_safe"] = Map.from_geom(geom, dtype=bool)

        return cls(**kwargs)

    @classmethod
    def from_map_dataset(
        cls, dataset, acceptance, acceptance_off, counts_off=None, name=None
    ):
        """Create map dataseton off from another dataset.

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

        if counts_off is None and dataset.background_model is not None:
            alpha = acceptance / acceptance_off
            counts_off = dataset.background_model.evaluate() / alpha

        return cls(
            counts=dataset.counts,
            exposure=dataset.exposure,
            counts_off=counts_off,
            edisp=dataset.edisp,
            gti=dataset.gti,
            mask_safe=dataset.mask_safe,
            mask_fit=dataset.mask_fit,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            name=dataset.name,
            psf=dataset.psf,
        )

    def to_map_dataset(self, name=None):
        """ Convert a MapDatasetOnOff to  MapDataset
        The background model template is taken as alpha*counts_off

        Parameters:
        -----------
        name: str
            Name of the new dataset

        Returns:
        -------
        dataset: `MapDataset`
            MapDatset with cash statistics
        """

        name = make_name(name)

        background_model = BackgroundModel(self.counts_off * self.alpha)
        background_model.datasets_names = [name]
        return MapDataset(
            counts=self.counts,
            exposure=self.exposure,
            psf=self.psf,
            edisp=self.edisp,
            name=name,
            gti=self.gti,
            mask_fit=self.mask_fit,
            mask_safe=self.mask_safe,
            models=background_model,
            meta_table=self.meta_table,
        )

    @property
    def _is_stackable(self):
        """Check if the Dataset contains enough information to be stacked"""
        if (
            self.acceptance_off is None
            or self.acceptance is None
            or self.counts_off is None
        ):
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

        # Factor containing: self.alpha * self.counts_off + other.alpha * other.counts_off
        tmp_factor = self.counts_off_normalised * self.mask_safe
        tmp_factor.stack(other.counts_off_normalised, weights=other.mask_safe)

        # Stack the off counts (in place)
        self.counts_off.data[~self.mask_safe.data] = 0
        self.counts_off.stack(other.counts_off, weights=other.mask_safe)

        self.acceptance_off = self.counts_off / tmp_factor
        self.acceptance.data = np.ones(self.data_shape)

        super().stack(other)

    def stat_sum(self):
        """Total likelihood given the current model parameters."""
        return Dataset.stat_sum(self)

    def fake(self, background_model, random_state="random-seed"):
        """Simulate fake counts (on and off) for the current model and reduced IRFs.

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

        npred_bkg = background_model.copy()
        npred_bkg.data = random_state.poisson(npred_bkg.data)

        self.counts = npred + npred_bkg

        npred_off = background_model / self.alpha
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
    def from_hdulist(cls, hdulist, name=None):
        """Create map dataset from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        kwargs = {}
        kwargs["name"] = name
        if "COUNTS" in hdulist:
            kwargs["counts"] = Map.from_hdulist(hdulist, hdu="counts")

        if "COUNTS_OFF" in hdulist:
            kwargs["counts_off"] = Map.from_hdulist(hdulist, hdu="counts_off")

        if "ACCEPTANCE" in hdulist:
            kwargs["acceptance"] = Map.from_hdulist(hdulist, hdu="acceptance")

        if "ACCEPTANCE_OFF" in hdulist:
            kwargs["acceptance_off"] = Map.from_hdulist(hdulist, hdu="acceptance_off")

        if "EXPOSURE" in hdulist:
            kwargs["exposure"] = Map.from_hdulist(hdulist, hdu="exposure")

        if "EDISP_MATRIX" in hdulist:
            kwargs["edisp"] = EDispKernel.from_hdulist(
                hdulist, hdu1="EDISP_MATRIX", hdu2="EDISP_MATRIX_EBOUNDS"
            )

        if "PSF_KERNEL" in hdulist:
            psf_map = Map.from_hdulist(hdulist, hdu="psf_kernel")
            kwargs["psf"] = PSFKernel(psf_map)

        if "MASK_SAFE" in hdulist:
            mask_safe = Map.from_hdulist(hdulist, hdu="mask_safe")
            kwargs["mask_safe"] = mask_safe

        if "MASK_FIT" in hdulist:
            mask_fit = Map.from_hdulist(hdulist, hdu="mask_fit")
            kwargs["mask_fit"] = mask_fit

        if "GTI" in hdulist:
            gti = GTI(Table.read(hdulist, hdu="GTI"))
            kwargs["gti"] = gti
        return cls(**kwargs)

    def info_dict(self, region=None):
        """Basic info dict with summary statistics

        If a region is passed, then a spectrum dataset is
        extracted, and the corresponding info returned.

        Parameters
        ----------
        region : `~regions.SkyRegion`, optional
            the input ON region on which to extract the spectrum

        Returns
        -------
        info_dict : dict
            Dictionary with summary info.
        """
        info = super().info_dict(region)
        info["name"] = self.name
        if self.gti is None:
            if self.counts_off is not None:
                info["counts_off"] = np.sum(self.counts_off.data)

            if self.acceptance is not None:
                info["acceptance"] = np.sum(self.acceptance.data)

            if self.acceptance_off is not None:
                info["acceptance_off"] = np.sum(self.acceptance_off.data)

            info["excess"] = np.sum(self.excess.data)
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
            kwargs["counts_off"] = self.counts_off.get_spectrum(on_region, np.sum)

        if self.acceptance is not None:
            kwargs["acceptance"] = self.acceptance.get_spectrum(on_region, np.mean)
            norm = self.counts_off_normalised.get_spectrum(on_region, np.sum)
            kwargs["acceptance_off"] = (
                kwargs["acceptance"] * kwargs["counts_off"] / norm
            )

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
                factor=factor, preserve_counts=True, axis_name=axis_name, weights=self.mask_safe
            )

        acceptance, acceptance_off = None, None
        if self.acceptance_off is not None:
            acceptance = self.acceptance.downsample(
                factor=factor, preserve_counts=False, axis_name=axis_name
            )
            factor = self.counts_off_normalised.downsample(
                factor=factor, preserve_counts=True, axis_name=axis_name, weights=self.mask_safe
            )
            acceptance_off = acceptance * counts_off / factor

        return self.__class__.from_map_dataset(
            dataset,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            counts_off=counts_off
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

    def resample_energy_axis(self, axis=None, name=None):
        """Resample MapDatasetOnOff over reco energy edges.

        Counts are summed taking into account safe mask.

        Parameters
        ----------
        axis : `~gammapy.maps.MapAxis`
            the new reco energy axis.
        name: str
            Name of the new dataset.

        Returns
        -------
        dataset: `SpectrumDataset`
            Resampled spectrum dataset .
        """
        dataset = super().resample_energy_axis(axis,name)

        axis = dataset.counts.geom.axes["energy"]

        if self.mask_safe is not None:
            weights = self.mask_safe
        else:
            weights = None

        counts_off = None
        if self.counts_off is not None:
            counts_off = self.counts_off
            counts_off = counts_off.resample_axis(axis=axis, weights=weights)

        acceptance = 1
        acceptance_off = None
        if self.acceptance is not None:
            acceptance = self.acceptance
            acceptance = acceptance.resample_axis(axis=axis, weights=weights)

            norm_factor = self.counts_off_normalised.resample_axis(axis=axis, weights=weights)

            acceptance_off = acceptance * counts_off / norm_factor

        return self.__class__.from_map_dataset(
            dataset,
            acceptance=acceptance,
            acceptance_off=acceptance_off,
            counts_off=counts_off
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
        evaluation_mode="local",
        use_cache=True,
    ):

        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp
        self.gti = gti
        self.contributes = True
        self.use_cache = use_cache

        if evaluation_mode not in {"local", "global"}:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode!r}")

        self.evaluation_mode = evaluation_mode

        # TODO: this is preliminary solution until we have further unified the model handling
        if isinstance(self.model, BackgroundModel):
            self.evaluation_mode = "global"

        # define cached computations
        self._compute_npred = lru_cache()(self._compute_npred)
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
            if key in ["_compute_npred", "_compute_flux_spatial"]:
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
        elif self.evaluation_mode == "global" or self.model.evaluation_radius is None:
            return False
        else:
            position = self.model.position
            separation = self._init_position.separation(position)
            update = separation > (self.model.evaluation_radius + CUTOUT_MARGIN)
        return update

    def update(self, exposure, psf, edisp, geom):
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
        """
        # TODO: simplify and clean up
        log.debug("Updating model evaluator")
        # cache current position of the model component

        # lookup edisp
        if edisp:
            energy_axis = geom.axes["energy"]
            self.edisp = edisp.get_edisp_kernel(
                self.model.position, energy_axis=energy_axis
            )

        if isinstance(psf, PSFMap):
            # lookup psf
            self.psf = psf.get_psf_kernel(self.model.position, geom=exposure.geom)
        else:
            self.psf = psf

        if self.evaluation_mode == "local" and self.model.evaluation_radius is not None:
            self._init_position = self.model.position
            if self.psf is not None:
                psf_width = np.max(self.psf.psf_kernel_map.geom.width)
            else:
                psf_width = 0 * u.deg

            width = psf_width + 2 * (self.model.evaluation_radius + CUTOUT_MARGIN)
            try:
                self.exposure = exposure.cutout(
                    position=self.model.position, width=width
                )
                self.contributes = True
            except (NoOverlapError, ValueError):
                self.contributes = False
        else:
            self.exposure = exposure

        self._compute_npred.cache_clear()
        self._compute_flux_spatial.cache_clear()

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

        if self.model.spatial_model and not isinstance(self.geom, RegionGeom):
            value = value * self.compute_flux_spatial().quantity

        if self.model.temporal_model:
            value *= self.compute_temporal_norm()

        return Map.from_geom(geom=self.geom, data=value.value, unit=value.unit)

    def _compute_flux_spatial(self):
        """Compute spatial flux"""
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
        value = self.model.spectral_model.integral(
            energy[:-1], energy[1:], intervals=True
        )
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
            flux_conv = self.compute_flux_psf_convolved()

            if self.model.apply_irf["exposure"]:
                npred = self.apply_exposure(flux_conv)

            if self.model.apply_irf["edisp"]:
                npred = self.apply_edisp(npred)

        return npred

    def compute_npred(self):
        """Evaluate model predicted counts.

        Returns
        -------
        npred : `~gammapy.maps.Map`
            Predicted counts on the map (in reco energy bins)
        """
        if self.parameters_changed or not self.use_cache:
            self._compute_npred.cache_clear()

        return self._compute_npred()

    @property
    def parameters_changed(self):
        """Parameters changed"""
        values = self.model.parameters.values

        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._cached_parameter_values == values)

        if changed:
            self._cached_parameter_values = values

        return changed

    @property
    def parameters_spatial_changed(self):
        """Parameters changed"""
        values = self.model.spatial_model.parameters.values

        # TODO: possibly allow for a tolerance here?
        changed = ~np.all(self._cached_parameter_values_spatial == values)

        if changed:
            self._cached_parameter_values_spatial = values

        return changed
