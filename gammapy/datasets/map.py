# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.nddata.utils import NoOverlapError
from astropy.table import Table
from astropy.utils import lazyproperty
from regions import CircleSkyRegion
from gammapy.data import GTI
from gammapy.irf import EDispKernel, EffectiveAreaTable
from gammapy.irf.edisp_map import EDispMap
from gammapy.irf.psf_kernel import PSFKernel
from gammapy.irf.psf_map import PSFMap
from gammapy.maps import Map, MapAxis
from gammapy.modeling.models import BackgroundModel, Models
from gammapy.stats import cash, cash_sum_cython, wstat
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_name, make_path
from .core import Dataset


__all__ = ["MapDataset", "MapDatasetOnOff"]

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
    psf : `~gammapy.cube.PSFKernel` or `~gammapy.cube.PSFMap`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel` or `~gammapy.cube.EDispMap`
        Energy dispersion kernel
    evaluation_mode : {"local", "global"}
        Model evaluation mode.
        The "local" mode evaluates the model components on smaller grids to save computation time.
        This mode is recommended for local optimization algorithms.
        The "global" evaluation mode evaluates the model components on the full map.
        This mode is recommended for global optimization algorithms.
    mask_safe : `~gammapy.maps.WcsNDMap`
        Mask defining the safe data range.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    """

    stat_type = "cash"
    tag = "MapDataset"

    def __init__(
        self,
        models=None,
        counts=None,
        exposure=None,
        mask_fit=None,
        psf=None,
        edisp=None,
        name=None,
        evaluation_mode="local",
        mask_safe=None,
        gti=None,
    ):
        if mask_fit is not None and mask_fit.data.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        if mask_safe is not None and mask_safe.data.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self._name = make_name(name)
        self.background_model = None
        self.evaluation_mode = evaluation_mode
        self.counts = counts
        self.exposure = exposure
        self.mask_fit = mask_fit
        self.psf = psf
        self.edisp = edisp
        self.mask_safe = mask_safe
        self.models = models
        self.gti = gti

        # check whether a reference geom is defined
        _ = self._geom

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
        return self._models

    @models.setter
    def models(self, models):
        if models is None:
            self._models = None
        else:
            self._models = Models(models)

        # TODO: clean this up (probably by removing)
        if self.models is not None:
            for model in self.models:
                if isinstance(model, BackgroundModel):
                    if model.datasets_names is not None:
                        if self.name in model.datasets_names:
                            self.background_model = model
                            break
            else:
                log.warning(f"No background model defined for dataset {self.name}")
        self._evaluators = {}

    @property
    def evaluators(self):
        """Model evaluators"""
        # this call is needed to trigger the setup of the evaluators
        if not self._evaluators:
            self.npred()

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

        if self.models:
            for model in self.models:
                if model.datasets_names is not None:
                    if self.name not in model.datasets_names:
                        continue

                evaluator = self._evaluators.get(model.name)

                if evaluator is None:
                    evaluator = MapEvaluator(
                        model=model, evaluation_mode=self.evaluation_mode, gti=self.gti
                    )
                    self._evaluators[model.name] = evaluator

                # if the model component drifts out of its support the evaluator has
                # has to be updated

                if evaluator.needs_update:
                    evaluator.update(self.exposure, self.psf, self.edisp, self._geom)

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
            geometry for the energy dispersion map
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
        kwargs["edisp"] = EDispMap.from_geom(geom_edisp)
        kwargs["psf"] = PSFMap.from_geom(geom_psf)

        kwargs.setdefault("gti", GTI.create([] * u.s, [] * u.s, reference_time=reference_time))
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
            Migration axis for the energy dispersion map
        rad_axis : `~gammapy.maps.MapAxis`
            Rad axis for the psf map
        binsz_irf : float
            IRF Map pixel size in degrees.
        reference_time : `~astropy.time.Time`
            the reference time to use in GTI definition
        name : str
            Name of the returned dataset.

        Returns
        -------
        empty_maps : `MapDataset`
            A MapDataset containing zero filled maps
        """
        migra_axis = migra_axis or MIGRA_AXIS_DEFAULT
        rad_axis = rad_axis or RAD_AXIS_DEFAULT

        if energy_axis_true is not None:
            if energy_axis_true.name != "energy_true":
                raise ValueError("True enery axis name must be 'energy_true'")
        else:
            energy_axis_true = geom.get_axis_by_name("energy").copy(name="energy_true")

        binsz_irf = binsz_irf or BINSZ_IRF_DEFAULT
        geom_image = geom.to_image()
        geom_exposure = geom_image.to_cube([energy_axis_true])
        geom_irf = geom_image.to_binsz(binsz=binsz_irf)
        geom_psf = geom_irf.to_cube([rad_axis, energy_axis_true])
        geom_edisp = geom_irf.to_cube([migra_axis, energy_axis_true])

        return cls.from_geoms(
            geom,
            geom_exposure,
            geom_psf,
            geom_edisp,
            reference_time=reference_time,
            name=name,
            **kwargs,
        )

    def stack(self, other):
        """Stack another dataset in place.

        Parameters
        ----------
        other: `~gammapy.cube.MapDataset` or `~gammapy.cube.MapDatasetOnOff`
            Map dataset to be stacked with this one. If other is an on-off
            dataset alpha * counts_off is used as a background model.
        """

        if self.counts and other.counts:
            self.counts *= self.mask_safe
            self.counts.stack(other.counts, weights=other.mask_safe)

        if self.exposure and other.exposure:
            mask_image = self.mask_safe.reduce_over_axes(func=np.logical_or)
            self.exposure *= mask_image.data
            # TODO: apply energy dependent mask to exposure. Does this require
            #  a mask_safe in true energy?
            mask_image_other = other.mask_safe.reduce_over_axes(func=np.logical_or)
            self.exposure.stack(other.exposure, weights=mask_image_other)

        # TODO: unify background model handling
        if other.stat_type == "wstat":
            background_model = BackgroundModel(other.background)
        else:
            background_model = other.background_model

        if self.background_model and background_model:
            self.background_model.map *= self.mask_safe
            self.background_model.stack(background_model, other.mask_safe)

        if self.mask_safe is not None and other.mask_safe is not None:
            self.mask_safe.stack(other.mask_safe)

        if self.psf and other.psf:
            if isinstance(self.psf, PSFMap) and isinstance(other.psf, PSFMap):
                mask_irf = self._mask_safe_irf(self.psf.psf_map, mask_image)
                self.psf.psf_map *= mask_irf.data
                self.psf.exposure_map *= mask_irf.data

                mask_image_other = other.mask_safe.reduce_over_axes(func=np.logical_or)
                mask_irf_other = self._mask_safe_irf(
                    other.psf.psf_map, mask_image_other
                )
                self.psf.stack(other.psf, weights=mask_irf_other)
            else:
                raise ValueError("Stacking of PSF kernels not supported")

        if self.edisp and other.edisp:
            if isinstance(self.edisp, EDispMap) and isinstance(other.edisp, EDispMap):
                mask_irf = self._mask_safe_irf(self.edisp.edisp_map, mask_image)
                self.edisp.edisp_map *= mask_irf.data
                self.edisp.exposure_map *= mask_irf.data

                mask_image_other = other.mask_safe.reduce_over_axes(func=np.logical_or)
                mask_irf_other = self._mask_safe_irf(
                    other.edisp.edisp_map, mask_image_other
                )
                self.edisp.stack(other.edisp, weights=mask_irf_other)
            else:
                raise ValueError("Stacking of edisp kernels not supported")

        if self.gti and other.gti:
            self.gti = self.gti.stack(other.gti).union()

    @staticmethod
    def _mask_safe_irf(irf_map, mask):
        geom = irf_map.geom.to_image()
        coords = geom.get_coord()
        data = mask.get_by_coord(coords).astype(bool)
        return Map.from_geom(geom=geom, data=data)

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
        return self._compute_residuals(self.counts, self.npred(), method=method)

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
            mask = self.mask_safe.reduce_over_axes(func=np.logical_or)
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
            ax = residuals.plot()
            ax.set_yscale("linear")
            ax.axhline(0, color="black", lw=0.5)

            y_max = 2 * np.nanmax(residuals.data)
            plt.ylim(-y_max, y_max)
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
                [BackgroundModel(background_map, datasets_names=[name], name=name + "-bkg")]
            )

        if "EDISP_MATRIX" in hdulist:
            kwargs["edisp"] = EDispKernel.from_hdulist(
                hdulist, hdu1="EDISP_MATRIX", hdu2="EDISP_MATRIX_EBOUNDS"
            )
        if "EDISP" in hdulist:
            edisp_map = Map.from_hdulist(hdulist, hdu="edisp")
            exposure_map = Map.from_hdulist(hdulist, hdu="edisp_exposure")
            kwargs["edisp"] = EDispMap(edisp_map, exposure_map)

        if "PSF_KERNEL" in hdulist:
            psf_map = Map.from_hdulist(hdulist, hdu="psf_kernel")
            kwargs["psf"] = PSFKernel(psf_map)
        if "PSF" in hdulist:
            psf_map = Map.from_hdulist(hdulist, hdu="psf")
            exposure_map = Map.from_hdulist(hdulist, hdu="psf_exposure")
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
        self.to_hdulist().writeto(make_path(filename), overwrite=overwrite)

    @classmethod
    def read(cls, filename, name=None):
        """Read map dataset from file.

        Parameters
        ----------
        filename : str
            Filename to read from.
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        with fits.open(make_path(filename), memmap=False) as hdulist:
            return cls.from_hdulist(hdulist, name=name)

    @classmethod
    def from_dict(cls, data, models):
        """Create from dicts and models list generated from YAML serialization."""
        dataset = cls.read(data["filename"], name=data["name"])

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
        return {
            "name": self.name,
            "type": self.tag,
            "filename": str(filename),
        }

    def to_spectrum_dataset(self, on_region, containment_correction=False, name=None):
        """Return a ~gammapy.spectrum.SpectrumDataset from on_region.

        Counts and background are summed in the on_region.

        Effective area is taken from the average exposure divided by the livetime.
        Here we assume it is the sum of the GTIs.

        The energy dispersion kernel is obtained at the on_region center.
        Only regions with centers are supported.

        The model is not exported to the ~gammapy.spectrum.SpectrumDataset.
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
        dataset : `~gammapy.spectrum.SpectrumDataset`
            the resulting reduced dataset
        """
        from .spectrum import SpectrumDataset

        kwargs = {"gti": self.gti, "name": name}

        if self.gti is not None:
            kwargs["livetime"] = self.gti.time_sum
        else:
            raise ValueError("No GTI in `MapDataset`, cannot compute livetime")

        if self.counts is not None:
            kwargs["counts"] = self.counts.get_spectrum(on_region, np.sum)

        if self.background_model is not None:
            kwargs["background"] = self.background_model.evaluate().get_spectrum(
                on_region, np.sum
            )

        if self.exposure is not None:
            exposure = self.exposure.get_spectrum(on_region, np.mean)
            energy = exposure.geom.axes[0].edges
            kwargs["aeff"] = EffectiveAreaTable(
                energy_lo=energy[:-1],
                energy_hi=energy[1:],
                data=exposure.quantity[:, 0, 0] / kwargs["livetime"],
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
                containment = psf.containment(
                    kwargs["aeff"].energy.center, on_region.radius
                )
                kwargs["aeff"].data.data *= containment.squeeze()

        if self.edisp is not None:
            if isinstance(self.edisp, EDispKernel):
                edisp = self.edisp
            else:
                axis = self._geom.get_axis_by_name("energy")
                edisp = self.edisp.get_edisp_kernel(on_region.center, e_reco=axis.edges)
            kwargs["edisp"] = edisp

        return SpectrumDataset(**kwargs)

    def to_image(self, spectrum=None, name=None):
        """Create images by summing over the energy axis.

        Exposure is weighted with an assumed spectrum,
        resulting in a weighted mean exposure image.

        Currently the PSFMap and EdispMap are dropped from the
        resulting image dataset.

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset containing images.
        """
        from gammapy.makers.utils import _map_spectrum_weight

        name = make_name(name)
        kwargs = {}
        kwargs["name"] = name
        kwargs["gti"] = self.gti

        if self.mask_safe is not None:
            mask_safe = self.mask_safe
            kwargs["mask_safe"] = mask_safe.reduce_over_axes(
                func=np.logical_or, keepdims=True
            )
        else:
            mask_safe = 1

        if self.counts is not None:
            counts = self.counts * mask_safe
            kwargs["counts"] = counts.sum_over_axes(keepdims=True)

        if self.exposure is not None:
            exposure = _map_spectrum_weight(self.exposure, spectrum)
            kwargs["exposure"] = exposure.sum_over_axes(keepdims=True)

        if self.background_model is not None:
            background = self.background_model.evaluate() * mask_safe
            background = background.sum_over_axes(keepdims=True)
            kwargs["models"] = Models(
                [BackgroundModel(background, datasets_names=[name])]
            )

        if self.psf is not None:
            # TODO: implement PSFKernel.to_image()
            if not isinstance(self.psf, PSFKernel):
                kwargs["psf"] = self.psf.to_image(spectrum=spectrum, keepdims=True)
            else:
                # assume exposure at center position
                kwargs["psf"] = None

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
    mask_fit : `~numpy.ndarray`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    evaluation_mode : {"local", "global"}
        Model evaluation mode.
        The "local" mode evaluates the model components on smaller grids to save computation time.
        This mode is recommended for local optimization algorithms.
        The "global" evaluation mode evaluates the model components on the full map.
        This mode is recommended for global optimization algorithms.
    mask_safe : `~numpy.ndarray`
        Mask defining the safe data range.
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    name : str
        Name of the dataset.

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
        evaluation_mode="local",
        mask_safe=None,
        gti=None,
    ):
        if mask_fit is not None and mask_fit.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.evaluation_mode = evaluation_mode
        self.counts = counts
        self.counts_off = counts_off

        if np.isscalar(acceptance):
            acceptance = np.ones(self.data_shape) * acceptance

        if np.isscalar(acceptance_off):
            acceptance_off = np.ones(self.data_shape) * acceptance_off

        self.acceptance = acceptance
        self.acceptance_off = acceptance_off
        self.exposure = exposure
        self.background_model = None
        self.mask_fit = mask_fit
        self.psf = psf
        self.edisp = edisp
        self.models = models
        self._name = make_name(name)
        self.mask_safe = mask_safe
        self.gti = gti

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
        """Predicted background in the on region.

        Notice that this definition is valid under the assumption of cash statistic.
        """
        return self.alpha * self.counts_off

    @property
    def excess(self):
        """Excess (counts - alpha * counts_off)"""
        return self.counts.data - self.background.data

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
            geometry for the energy dispersion map
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
        kwargs["edisp"] = EDispMap.from_geom(geom_edisp)
        kwargs["psf"] = PSFMap.from_geom(geom_psf)
        kwargs["gti"] = GTI.create([] * u.s, [] * u.s, reference_time=reference_time)
        kwargs["mask_safe"] = Map.from_geom(geom, dtype=bool)

        return cls(**kwargs)

    @classmethod
    def from_map_dataset(
        cls, dataset, acceptance, acceptance_off, counts_off=None, name=None
    ):
        """Create spectrum dataseton off from another dataset.

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
        kwargs = {"name": name}

        if counts_off is None and dataset.background_model is not None:
            alpha = acceptance / acceptance_off
            kwargs["counts_off"] = dataset.background_model.evaluate() / alpha

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
            evaluation_mode=dataset.evaluation_mode,
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
        tmp_factor = (self.alpha * self.counts_off).copy()
        tmp_factor.data[~self.mask_safe.data] = 0
        tmp_factor.stack(other.alpha * other.counts_off, weights=other.mask_safe)

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
            mask_safe_map = Map.from_hdulist(hdulist, hdu="mask_safe")
            kwargs["mask_safe"] = mask_safe_map.data.astype(bool)

        if "MASK_FIT" in hdulist:
            mask_fit_map = Map.from_hdulist(hdulist, hdu="mask_fit")
            kwargs["mask_fit"] = mask_fit_map.data.astype(bool)

        if "GTI" in hdulist:
            gti = GTI(Table.read(hdulist, hdu="GTI"))
            kwargs["gti"] = gti
        return cls(**kwargs)

    def to_spectrum_dataset(self, on_region, containment_correction=False, name=None):
        """Return a ~gammapy.spectrum.SpectrumDatasetOnOff from on_region.

        Counts and OFF counts are summed in the on_region.

        Acceptance is the average of all acceptances while acceptance OFF
        is taken such that number of excess is preserved in the on_region.

        Effective area is taken from the average exposure divided by the livetime.
        Here we assume it is the sum of the GTIs.

        The energy dispersion kernel is obtained at the on_region center.
        Only regions with centers are supported.

        The model is not exported to the ~gammapy.spectrum.SpectrumDataset.
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
        dataset : `~gammapy.spectrum.SpectrumDatasetOnOff`
            the resulting reduced dataset
        """
        from .spectrum import SpectrumDatasetOnOff

        dataset = super().to_spectrum_dataset(on_region, containment_correction, name)

        kwargs = {}
        if self.counts_off is not None:
            kwargs["counts_off"] = self.counts_off.get_spectrum(on_region, np.sum)

        if self.acceptance is not None:
            kwargs["acceptance"] = self.acceptance.get_spectrum(on_region, np.mean)
            background = self.background.get_spectrum(on_region, np.sum)
            kwargs["acceptance_off"] = (
                kwargs["acceptance"] * kwargs["counts_off"] / background
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

    def to_image(self, spectrum=None, name=None):
        """Create images by summing over the energy axis.

        Exposure is weighted with an assumed spectrum,
        resulting in a weighted mean exposure image.

        Currently the PSFMap and EdispMap are dropped from the
        resulting image dataset.

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        name : str
            Name of the new dataset.

        Returns
        -------
        dataset : `MapDatasetOnOff`
            Map dataset containing images.
        """
        kwargs = {"name": name}
        dataset = super().to_image(spectrum, name)

        if self.mask_safe is not None:
            mask_safe = self.mask_safe
        else:
            mask_safe = 1

        if self.counts_off is not None:
            counts_off = self.counts_off * mask_safe
            kwargs["counts_off"] = counts_off.sum_over_axes(keepdims=True)

        if self.acceptance is not None:
            acceptance = self.acceptance * mask_safe
            kwargs["acceptance"] = acceptance.sum_over_axes(keepdims=True)

            background = self.background * mask_safe
            background = background.sum_over_axes(keepdims=True)
            kwargs["acceptance_off"] = (
                kwargs["acceptance"] * kwargs["counts_off"] / background
            )

        return self.from_map_dataset(dataset, **kwargs)


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
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EDispKernel`
        Energy dispersion
    gti : `~gammapy.data.GTI`
        GTI of the observation or union of GTI if it is a stacked observation
    evaluation_mode : {"local", "global"}
        Model evaluation mode.
    """

    def __init__(
        self,
        model=None,
        exposure=None,
        psf=None,
        edisp=None,
        gti=None,
        evaluation_mode="local",
    ):
        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp
        self.gti = gti
        self.contributes = True

        if evaluation_mode not in {"local", "global"}:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode!r}")

        self.evaluation_mode = evaluation_mode

        # TODO: this is preliminary solution until we have further unified the model handling
        if isinstance(model, BackgroundModel):
            self.compute_npred = model.evaluate
            self.evaluation_mode = "global"

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
        psf : `gammapy.cube.PSFMap`
            PSF map.
        edisp : `gammapy.cube.EDispMap`
            Edisp map.
        geom : `WcsGeom`
            Counts geom
        """
        # TODO: simplify and clean up
        log.debug("Updating model evaluator")
        # cache current position of the model component

        if isinstance(edisp, EDispMap):
            e_reco = geom.get_axis_by_name("energy").edges
            self.edisp = edisp.get_edisp_kernel(self.model.position, e_reco=e_reco)
        else:
            self.edisp = edisp

        if isinstance(psf, PSFMap):
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
        """Compute model integral flux over map pixel volumes.

        For now, we simply multiply dnde with bin volume.
        """
        return self.model.integrate_geom(self.geom, self.gti)

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

    def compute_npred(self):
        """
        Evaluate model predicted counts.

        Returns
        -------
        npred : `~gammapy.maps.Map`
            Predicted counts on the map (in reco energy bins)
        """
        flux = self.compute_flux()

        if self.model.apply_irf["exposure"]:
            npred = self.apply_exposure(flux)

        if self.psf and self.model.apply_irf["psf"]:
            npred = self.apply_psf(npred)

        if self.model.apply_irf["edisp"]:
            npred = self.apply_edisp(npred)

        return npred
