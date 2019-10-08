# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.nddata.utils import NoOverlapError
from astropy.utils import lazyproperty
from regions import CircleSkyRegion
from gammapy.cube.edisp_map import EDispMap
from gammapy.cube.psf_kernel import PSFKernel
from gammapy.cube.psf_map import PSFMap
from gammapy.data import GTI
from gammapy.irf import EffectiveAreaTable, EnergyDispersion, apply_containment_fraction
from gammapy.maps import Map, MapAxis
from gammapy.modeling import Dataset, Parameters
from gammapy.modeling.models import BackgroundModel, SkyModel, SkyModels
from gammapy.spectrum import SpectrumDataset
from gammapy.stats import cash, cash_sum_cython, cstat, cstat_sum_cython
from gammapy.utils.random import get_random_state
from gammapy.utils.scripts import make_path
from .exposure import _map_spectrum_weight

__all__ = ["MapEvaluator", "MapDataset"]

log = logging.getLogger(__name__)

CUTOUT_MARGIN = 0.1 * u.deg
RAD_MAX = 0.66
RAD_AXIS_DEFAULT = MapAxis.from_bounds(
    0, RAD_MAX, nbin=66, node_type="edges", name="theta", unit="deg"
)
MIGRA_AXIS_DEFAULT = MapAxis.from_bounds(
    0.2, 5, nbin=48, node_type="edges", name="migra"
)
BINSZ_IRF = 0.2
# TODO: Choose optimal binnings depending on IRFs


class MapDataset(Dataset):
    """Perform sky model likelihood fit on maps.

    Parameters
    ----------
    model : `~gammapy.modeling.models.SkyModel` or `~gammapy.modeling.models.SkyModels`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    mask_fit : `~numpy.ndarray`
        Mask to apply to the likelihood for fitting.
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    background_model : `~gammapy.modeling.models.BackgroundModel`
        Background model to use for the fit.
    likelihood : {"cash", "cstat"}
        Likelihood function to use for the fit.
    evaluation_mode : {"local", "global"}
        Model evaluation mode.

        The "local" mode evaluates the model components on smaller grids to save computation time.
        This mode is recommended for local optimization algorithms.
        The "global" evaluation mode evaluates the model components on the full map.
        This mode is recommended for global optimization algorithms.
    mask_safe : `~numpy.ndarray`
        Mask defining the safe data range.
    gti : '~gammapy.data.GTI'
        GTI of the observation or union of GTI if it is a stacked observation
    """

    def __init__(
        self,
        model=None,
        counts=None,
        exposure=None,
        mask_fit=None,
        psf=None,
        edisp=None,
        background_model=None,
        name="",
        likelihood="cash",
        evaluation_mode="local",
        mask_safe=None,
        gti=None,
    ):
        if mask_fit is not None and mask_fit.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.evaluation_mode = evaluation_mode
        self.likelihood_type = likelihood
        self.counts = counts
        self.exposure = exposure
        self.mask_fit = mask_fit
        self.psf = psf
        self.edisp = edisp
        self.background_model = background_model
        self.model = model
        self.name = name
        self.mask_safe = mask_safe
        self.gti = gti
        if likelihood == "cash":
            self._stat = cash
            self._stat_sum = cash_sum_cython
        elif likelihood == "cstat":
            self._stat = cstat
            self._stat_sum = cstat_sum_cython
        else:
            raise ValueError(f"Invalid likelihood: {likelihood!r}")

    def __str__(self):
        str_ = f"{self.__class__.__name__}\n"
        str_ += "\n"

        str_ += "\t{:32}: {} \n\n".format("Name", self.name)

        counts = np.nan
        if self.counts is not None:
            counts = np.sum(self.counts.data)
        str_ += "\t{:32}: {:.0f} \n".format("Total counts", counts)

        npred = np.nan
        if self.model is not None or self.background_model is not None:
            npred = np.sum(self.npred().data)
        str_ += "\t{:32}: {:.2f}\n".format("Total predicted counts", npred)

        background = np.nan
        if self.background_model is not None:
            background = np.sum(self.background_model.evaluate().data)
        str_ += "\t{:32}: {:.2f}\n\n".format("Total background counts", background)

        exposure_min, exposure_max, exposure_unit = np.nan, np.nan, ""
        if self.exposure is not None:
            exposure_min = np.min(self.exposure.data[self.exposure.data > 0])
            exposure_max = np.max(self.exposure.data)
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
            n_fit_bins = np.sum(self.mask)
        str_ += "\t{:32}: {} \n\n".format("Number of fit bins", n_fit_bins)

        # likelihood section
        str_ += "\t{:32}: {}\n".format("Fit statistic type", self.likelihood_type)

        stat = np.nan
        if self.model is not None or self.background_model is not None:
            stat = self.likelihood()
        str_ += "\t{:32}: {:.2f}\n\n".format("Fit statistic value (-2 log(L))", stat)

        # model section
        n_models = 0
        if self.model is not None:
            n_models = len(self.model.skymodels)
        str_ += "\t{:32}: {} \n".format("Number of models", n_models)

        str_ += "\t{:32}: {}\n".format(
            "Number of parameters", len(self.parameters.parameters)
        )
        str_ += "\t{:32}: {}\n\n".format(
            "Number of free parameters", len(self.parameters.free_parameters)
        )

        components = []

        if self.model is not None:
            components += self.model.skymodels

        if self.background_model is not None:
            components += [self.background_model]

        for idx, model in enumerate(components):
            str_ += f"\tComponent {idx}: \n"
            str_ += "\t\t{:28}: {}\n".format("Name", model.name)
            str_ += "\t\t{:28}: {}\n".format("Type", model.__class__.__name__)

            if isinstance(model, SkyModel):
                str_ += "\t\t{:28}: {}\n".format(
                    "Spatial  model type", model.spatial_model.__class__.__name__
                )
                str_ += "\t\t{:28}: {}\n".format(
                    "Spectral model type", model.spectral_model.__class__.__name__
                )

            str_ += "\t\tParameters:\n"

            info = str(model.parameters)
            lines = info.split("\n")
            str_ += "\t\t" + "\n\t\t".join(lines[2:-1])

            str_ += "\n\n"

        return str_.expandtabs(tabsize=4)

    @property
    def model(self):
        """Sky model to fit (`~gammapy.cube.SkyModel` or `~gammapy.cube.SkyModels`)"""
        return self._model

    @model.setter
    def model(self, model):
        if isinstance(model, SkyModel):
            model = SkyModels([model])

        self._model = model

        if model is not None:
            evaluators = []

            for component in model.skymodels:
                evaluator = MapEvaluator(
                    component, evaluation_mode=self.evaluation_mode
                )
                evaluator.update(self.exposure, self.psf, self.edisp)
                evaluators.append(evaluator)

            self._evaluators = evaluators

    @property
    def parameters(self):
        """List of parameters (`~gammapy.modeling.Parameters`)"""
        parameters = []

        if self.model:
            parameters += self.model.parameters.parameters

        if self.background_model:
            parameters += self.background_model.parameters.parameters

        return Parameters(parameters)

    @property
    def _geom(self):
        if self.counts is not None:
            return self.counts.geom
        elif self.background_model is not None:
            return self.background_model.map.geom
        else:
            return self.exposure.geom

    @property
    def _energy_axis(self):
        return self._geom.get_axis_by_name("energy")

    @property
    def data_shape(self):
        """Shape of the counts data (tuple)"""
        return self.counts.data.shape

    def npred(self):
        """Predicted source and background counts (`~gammapy.maps.Map`)."""
        npred_total = Map.from_geom(self._geom, dtype=float)

        if self.background_model:
            npred_total += self.background_model.evaluate()

        if self.model:
            for evaluator in self._evaluators:
                # if the model component drifts out of its support the evaluator has
                # has to be updated
                if evaluator.needs_update:
                    evaluator.update(self.exposure, self.psf, self.edisp, self._geom)

                npred = evaluator.compute_npred()
                npred_total.stack(npred, check=False)

        return npred_total

    @classmethod
    def create(
        cls,
        geom,
        geom_irf=None,
        migra_axis=None,
        rad_axis=None,
        reference_time="2000-01-01",
        name="",
        **kwargs
    ):
        """Creates a MapDataset object with zero filled maps

        Parameters
        ----------
        geom: `~gammapy.maps.WcsGeom`
            Reference target geometry in reco energy, used for counts and background maps
        geom_irf: `~gammapy.maps.WcsGeom`
            Reference image geometry in true energy, used for IRF maps.
        migra_axis: `~gammapy.maps.MapAxis`
            Migration axis for the energy dispersion map
        rad_axis: `~gammapy.maps.MapAxis`
            Rad axis for the psf map
        name : str
            Name of the dataset.
        """
        geom_irf = geom_irf or geom.to_binsz(BINSZ_IRF)
        migra_axis = migra_axis or MIGRA_AXIS_DEFAULT
        rad_axis = rad_axis or RAD_AXIS_DEFAULT

        counts = Map.from_geom(geom, unit="")

        background = Map.from_geom(geom, unit="")
        background_model = BackgroundModel(background)

        energy_axis = geom_irf.get_axis_by_name("ENERGY")

        exposure_geom = geom.to_image().to_cube([energy_axis])
        exposure = Map.from_geom(exposure_geom, unit="m2 s")
        exposure_irf = Map.from_geom(geom_irf, unit="m2 s")

        mask_safe = np.zeros(geom.data_shape, dtype=bool)

        gti = GTI.create([] * u.s, [] * u.s, reference_time=reference_time)

        geom_migra = geom_irf.to_image().to_cube([migra_axis, energy_axis])
        edisp_map = Map.from_geom(geom_migra, unit="")
        loc = migra_axis.edges.searchsorted(1.0)
        edisp_map.data[:, loc, :, :] = 1.0
        edisp = EDispMap(edisp_map, exposure_irf)

        geom_rad = geom_irf.to_image().to_cube([rad_axis, energy_axis])
        psf_map = Map.from_geom(geom_rad, unit="sr-1")
        psf = PSFMap(psf_map, exposure_irf)

        return cls(
            counts=counts,
            exposure=exposure,
            psf=psf,
            edisp=edisp,
            background_model=background_model,
            gti=gti,
            mask_safe=mask_safe,
            name=name,
            **kwargs
        )

    def stack(self, other):
        """Stack another dataset in place.

        Parameters
        ----------
        other: `~gammapy.cube.MapDataset`
            Map dataset to be stacked with this one.
        """
        if self.counts and other.counts:
            self.counts.data[~self.mask_safe] = 0
            self.counts.stack(other.counts, weights=other.mask_safe)

        if self.exposure and other.exposure:
            self.exposure.stack(other.exposure)

        if self.background_model and other.background_model:
            bkg = self.background_model.evaluate()
            bkg.data[~self.mask_safe] = 0
            other_bkg = other.background_model.evaluate()
            other_bkg.data[~other.mask_safe] = 0
            bkg.stack(other_bkg)
            self.background_model = BackgroundModel(bkg, name=self.background_model.name)

        if self.mask_safe is not None and other.mask_safe is not None:
            # TODO: make mask_safe a Map object
            mask_safe = Map.from_geom(self.counts.geom, data=self.mask_safe)
            mask_safe_other = Map.from_geom(other.counts.geom, data=other.mask_safe)
            mask_safe.stack(mask_safe_other)
            self.mask_safe = mask_safe.data

        if self.psf and other.psf:
            if isinstance(self.psf, PSFMap) and isinstance(other.psf, PSFMap):
                self.psf.stack(other.psf)
            else:
                raise ValueError("Stacking of PSF kernels not supported")

        if self.edisp and other.edisp:
            if isinstance(self.edisp, EDispMap) and isinstance(other.edisp, EDispMap):
                self.edisp.stack(other.edisp)
            else:
                raise ValueError("Stacking of edisp kernels not supported")

        if self.gti and other.gti:
            self.gti = self.gti.stack(other.gti).union()

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        return self._stat(n_on=self.counts.data, mu_on=self.npred().data)

    def residuals(self, method="diff"):
        """Compute residuals map.

        Parameters
        ----------
        method: {"diff", "diff/model", "diff/sqrt(model)"}
            Method used to compute the residuals. Available options are:
                - `diff` (default): data - model
                - `diff/model`: (data - model) / model
                - `diff/sqrt(model)`: (data - model) / sqrt(model)

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

        The spectral residuals are extracted from the provided `region`, and the
        normalization used for the residuals computation can be controlled using
        the `norm` parameter. If no `region` is passed, only the spatial
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

        spatial_residuals.data[self.exposure.data[0] == 0] = np.nan

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

    def likelihood(self):
        """Total likelihood given the current model parameters."""
        counts, npred = self._counts_data, self.npred().data

        if self.mask is not None:
            return self._stat_sum(counts[self.mask], npred[self.mask])
        else:
            return self._stat_sum(counts.ravel(), npred.ravel())

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
            if isinstance(self.edisp, EnergyDispersion):
                hdus = self.edisp.to_hdulist()
                hdus["MATRIX"].name = "edisp_matrix"
                hdus["EBOUNDS"].name = "edisp_matrix_ebounds"
                hdulist.append(hdus["EDISP_MATRIX"])
                hdulist.append(hdus["EDISP_MATRIX_EBOUNDS"])
            else:
                hdulist += self.edisp.edisp_map.to_hdulist(hdu="EDISP")[exclude_primary]

        if self.psf is not None:
            if isinstance(self.psf, PSFKernel):
                hdulist += self.psf.psf_kernel_map.to_hdulist(hdu="psf_kernel")[
                    exclude_primary
                ]
            else:
                hdulist += self.psf.psf_map.to_hdulist(hdu="psf")[exclude_primary]

        if self.mask_safe is not None:
            mask_safe_map = Map.from_geom(
                self.counts.geom, data=self.mask_safe.astype(int)
            )
            hdulist += mask_safe_map.to_hdulist(hdu="mask_safe")[exclude_primary]

        if self.mask_fit is not None:
            mask_fit_map = Map.from_geom(
                self.counts.geom, data=self.mask_fit.astype(int)
            )
            hdulist += mask_fit_map.to_hdulist(hdu="mask_fit")[exclude_primary]

        if self.gti is not None:
            hdulist += self.gti.to_hdulist()

        return hdulist

    @classmethod
    def from_hdulist(cls, hdulist, name=""):
        """Create map dataset from list of HDUs.

        Parameters
        ----------
        hdulist : `~astropy.io.fits.HDUList`
            List of HDUs.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        init_kwargs = {}
        init_kwargs["name"] = name
        if "COUNTS" in hdulist:
            init_kwargs["counts"] = Map.from_hdulist(hdulist, hdu="counts")

        if "EXPOSURE" in hdulist:
            init_kwargs["exposure"] = Map.from_hdulist(hdulist, hdu="exposure")

        if "BACKGROUND" in hdulist:
            background_map = Map.from_hdulist(hdulist, hdu="background")
            init_kwargs["background_model"] = BackgroundModel(background_map)

        if "EDISP_MATRIX" in hdulist:
            init_kwargs["edisp"] = EnergyDispersion.from_hdulist(
                hdulist, hdu1="EDISP_MATRIX", hdu2="EDISP_MATRIX_EBOUNDS"
            )

        if "PSF_KERNEL" in hdulist:
            psf_map = Map.from_hdulist(hdulist, hdu="psf_kernel")
            init_kwargs["psf"] = PSFKernel(psf_map)

        if "MASK_SAFE" in hdulist:
            mask_safe_map = Map.from_hdulist(hdulist, hdu="mask_safe")
            init_kwargs["mask_safe"] = mask_safe_map.data.astype(bool)

        if "MASK_FIT" in hdulist:
            mask_fit_map = Map.from_hdulist(hdulist, hdu="mask_fit")
            init_kwargs["mask_fit"] = mask_fit_map.data.astype(bool)

        if "GTI" in hdulist:
            gti = GTI.from_hdulist(hdulist, hdu="GTI")
            init_kwargs["gti"] = gti
        return cls(**init_kwargs)

    def write(self, filename, overwrite=False):
        """Write map dataset to file.

        Parameters
        ----------
        filename : str
            Filename to write to.
        overwrite : bool
            Overwrite file if it exists.
        """
        filename = make_path(filename)
        hdulist = self.to_hdulist()
        hdulist.writeto(str(filename), overwrite=overwrite)

    @classmethod
    def read(cls, filename, name=""):
        """Read map dataset from file.

        Parameters
        ----------
        filename : str
            Filename to read from.

        Returns
        -------
        dataset : `MapDataset`
            Map dataset.
        """
        filename = make_path(filename)
        hdulist = fits.open(str(filename))
        return cls.from_hdulist(hdulist, name=name)

    def to_dict(self, filename=""):
        """Convert to dict for YAML serialization."""
        return {
            "name": self.name,
            "models": self.model.names,
            "background": self.background_model.name,
            "filename": filename,
        }

    def to_spectrum_dataset(self, on_region, containment_correction=False):
        """Return a ~gammapy.spectrum.SpectrumDataset from on_region.

        Counts and background are summed in the on_region.

        Effective area is taken from the average exposure divided by the livetime.
        Here we assume it is the sum of the GTIs.

        EnergyDispersion is obtained at the on_region center.
        Only regions with centers are supported.

        Parameters
        ----------
        on_region : `~regions.SkyRegion`
            the input ON region on which to extract the spectrum
        containment_correction : bool
            Apply containment correction for point sources and circular on regions

        Returns
        -------
        dataset : `~gammapy.spectrum.SpectrumDataset`
            the resulting reduced dataset
        """
        if self.gti is not None:
            livetime = self.gti.time_sum
        else:
            raise ValueError("No GTI in `MapDataset`, cannot compute livetime")

        if self.counts is not None:
            counts = self.counts.get_spectrum(on_region, np.sum)
        else:
            counts = None

        if self.background_model is not None:
            background = self.background_model.evaluate().get_spectrum(
                on_region, np.sum
            )
        else:
            background = None

        if self.exposure is not None:
            exposure = self.exposure.get_spectrum(on_region, np.mean)
            aeff = EffectiveAreaTable(
                energy_lo=exposure.energy.edges[:-1],
                energy_hi=exposure.energy.edges[1:],
                data=exposure.data / livetime,
            )
        else:
            aeff = None

        if containment_correction:
            if not isinstance(on_region, CircleSkyRegion):
                raise TypeError(
                    "Containement correction is only supported for"
                    " `CircleSkyRegion`."
                )
            elif self.psf is None or isinstance(self.psf, PSFKernel):
                raise ValueError("No PSFMap set. Containement correction impossible")
            else:
                psf_table = self.psf.get_energy_dependent_table_psf(on_region.center)
                aeff = apply_containment_fraction(aeff, psf_table, on_region.radius)

        if self.edisp is not None:
            if isinstance(self.edisp, EnergyDispersion):
                edisp = self.edisp
            else:
                self.edisp.get_energy_dispersion(on_region.center, self._energy_axis)
        else:
            edisp = None

        return SpectrumDataset(
            counts=counts,
            background=background,
            aeff=aeff,
            edisp=edisp,
            livetime=livetime,
            gti=self.gti,
            name=self.name,
        )

    def to_image(self, spectrum=None, keepdims=True):
        """Create images by summing over the energy axis.

        Exposure is weighted with an assumed spectrum,
        resulting in a weighted mean exposure image.

        Parameters
        ----------
        spectrum : `~gammapy.modeling.models.SpectralModel`
            Spectral model to compute the weights.
            Default is power-law with spectral index of 2.
        keepdims : bool, optional
            If this is set to True, the energy axes is kept with a single bin.
            If False, the energy axes is removed

        Returns
        -------
        dataset : `MapDataset`
            Map dataset containing images.
        """
        counts = self.counts.sum_over_axes(keepdims=keepdims)
        exposure = _map_spectrum_weight(self.exposure, spectrum)
        exposure = exposure.sum_over_axes(keepdims=keepdims)
        background = self.background_model.evaluate().sum_over_axes(keepdims=keepdims)

        # TODO: add edisp and psf
        edisp = None
        psf = None

        return self.__class__(
            counts=counts,
            exposure=exposure,
            background_model=BackgroundModel(background),
            edisp=edisp,
            psf=psf,
            gti=self.gti,
            name=self.name,
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
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    evaluation_mode : {"local", "global"}
        Model evaluation mode.
    """

    def __init__(
        self, model=None, exposure=None, psf=None, edisp=None, evaluation_mode="local"
    ):
        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp

        if evaluation_mode not in {"local", "global"}:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode!r}")

        self.evaluation_mode = evaluation_mode

    @property
    def geom(self):
        """True energy map geometry (`~gammapy.maps.Geom`)"""
        return self.exposure.geom

    @property
    def needs_update(self):
        """Check whether the model component has drifted away from its support."""
        if self.evaluation_mode == "global" or self.model.evaluation_radius is None:
            return False
        else:
            position = self.model.position
            separation = self._init_position.separation(position)
            update = separation > (self.model.evaluation_radius + CUTOUT_MARGIN)
        return update

    def update(self, exposure, psf, edisp):
        """Update MapEvaluator, based on the current position of the model component.

        Parameters
        ----------
        exposure : `~gammapy.maps.Map`
            Exposure map.
        psf : `gammapy.cube.PSFMap`
            PSF map.
        edisp : `gammapy.cube.EDispMap`
            Edisp map.
        """
        log.debug("Updating model evaluator")
        # cache current position of the model component

        # TODO: lookup correct Edisp for this component
        self.edisp = edisp

        # TODO: lookup correct PSF for this component
        self.psf = psf

        if self.evaluation_mode == "local" and self.model.evaluation_radius is not None:
            self._init_position = self.model.position
            if psf is not None:
                psf_width = np.max(psf.psf_kernel_map.geom.width)
            else:
                psf_width = 0 * u.deg

            width = psf_width + 2 * (self.model.evaluation_radius + CUTOUT_MARGIN)
            try:
                self.exposure = exposure.cutout(
                    position=self.model.position, width=width
                )
            except NoOverlapError:
                raise ValueError(
                    f"Position {self.model.position!r} of model component is outside the image boundaries."
                    " Please check the starting values or position parameter boundaries of the model."
                )
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
        return self.model.evaluate_geom(self.geom)

    def compute_flux(self):
        """Compute model integral flux over map pixel volumes.

        For now, we simply multiply dnde with bin volume.
        """
        dnde = self.compute_dnde()
        volume = self.geom.bin_volume()
        return dnde * volume

    def apply_exposure(self, flux):
        """Compute npred cube

        For now just divide flux cube by exposure
        """
        npred = (flux * self.exposure.quantity).to_value("")
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
        npred = self.apply_exposure(flux)
        if self.psf is not None:
            npred = self.apply_psf(npred)
        if self.edisp is not None:
            npred = self.apply_edisp(npred)

        return npred
