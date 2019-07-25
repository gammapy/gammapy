# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u
from astropy.nddata.utils import NoOverlapError
from astropy.io import fits
from ..utils.random import get_random_state
from ..utils.scripts import make_path
from ..utils.fitting import Parameters, Dataset
from ..stats import cash, cstat, cash_sum_cython, cstat_sum_cython
from ..maps import Map
from ..irf import EnergyDispersion
from .models import SkyModel, SkyModels, BackgroundModel
from .psf_kernel import PSFKernel

__all__ = ["MapEvaluator", "MapDataset"]

log = logging.getLogger(__name__)

CUTOUT_MARGIN = 0.1 * u.deg


class MapDataset(Dataset):
    """Perform sky model likelihood fit on maps.

    Parameters
    ----------
    model : `~gammapy.cube.models.SkyModel` or `~gammapy.cube.models.SkyModels`
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
    background_model : `~gammapy.cube.models.BackgroundModel` or `~gammapy.cube.models.BackgroundModels`
        Background models to use for the fit.
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
    gti : '~gammapy.data.gti.GTI'
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
        likelihood="cash",
        evaluation_mode="local",
        mask_safe=None,
        gti=None,
    ):
        if mask_fit is not None and mask_fit.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.evaluation_mode = evaluation_mode
        self.likelihood_type = likelihood
        self.model = model
        self.counts = counts
        self.exposure = exposure
        self.mask_fit = mask_fit
        self.psf = psf
        self.edisp = edisp
        self.background_model = background_model
        self.mask_safe = mask_safe
        self.gti = gti
        if likelihood == "cash":
            self._stat = cash
            self._stat_sum = cash_sum_cython
        elif likelihood == "cstat":
            self._stat = cstat
            self._stat_sum = cstat_sum_cython
        else:
            raise ValueError("Invalid likelihood: {!r}".format(likelihood))

    def __repr__(self):
        str_ = self.__class__.__name__
        return str_

    def __str__(self):
        str_ = "{}\n".format(self.__class__.__name__)
        str_ += "\n"

        counts = np.nan
        if self.counts is not None:
            counts = np.sum(self.counts.data)
        str_ += "\t{:32}: {:.0f} \n".format("Total counts", counts)

        npred = np.nan
        if self.model is not None:
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
        if self.model is not None:
            stat = self.likelihood()
        str_ += "\t{:32}: {:.2f}\n\n".format("Fit statistic value (-2 log(L))", stat)

        # model section
        n_models = 0
        if self.model is not None:
            n_models = len(self.model.skymodels)
        str_ += "\t{:32}: {} \n".format("Number of models", n_models)

        n_bkg_models = 0
        if self.background_model is not None:
            try:
                n_bkg_models = len(self.background_model.models)
            except AttributeError:
                n_bkg_models = 1
        str_ += "\t{:32}: {} \n".format("Number of background models", n_bkg_models)

        str_ += "\t{:32}: {}\n".format(
            "Number of parameters", len(self.parameters.parameters)
        )
        str_ += "\t{:32}: {}\n\n".format(
            "Number of free parameters", len(self.parameters.free_parameters)
        )

        if self.model is not None:
            for idx, model in enumerate(self.model.skymodels):
                str_ += "\tSource {}: \n".format(idx)
                str_ += "\t\t{:28}: {}\n".format("Name", model.name)
                str_ += "\t\t{:28}: {}\n".format(
                    "Spatial model type", model.spatial_model.__class__.__name__
                )
                info = str(model.spatial_model.parameters)
                lines = info.split("\n")
                str_ += "\t\t" + "\n\t\t".join(lines[2:-1])

                str_ += "\n\t\t{:28}: {}\n".format(
                    "Spectral model type", model.spectral_model.__class__.__name__
                )
                info = str(model.spectral_model.parameters)
                lines = info.split("\n")
                str_ += "\t\t" + "\n\t\t".join(lines[2:-1])

        if self.background_model is not None:
            try:
                background_models = self.background_model.models
            except AttributeError:
                background_models = [self.background_model]

            for idx, model in enumerate(background_models):
                str_ += "\n\n\tBackground {}: \n".format(idx)
                str_ += "\t\t{:28}: {}\n".format(
                    "Model type", self.background_model.__class__.__name__
                )
                info = str(self.background_model.parameters)
                lines = info.split("\n")
                str_ += "\t\t" + "\n\t\t".join(lines[2:-1])

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
                evaluators.append(evaluator)

            self._evaluators = evaluators

    @property
    def parameters(self):
        """List of parameters (`~gammapy.utils.fitting.Parameters`)"""
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
        else:
            return self.background_model.map.geom

    @property
    def data_shape(self):
        """Shape of the counts data (tuple)"""
        return self.counts.data.shape

    def npred(self):
        """Predicted source and background counts (`~gammapy.maps.Map`)."""
        npred_total = Map.from_geom(self._geom)

        if self.background_model:
            npred_total += self.background_model.evaluate()

        if self.model:
            for evaluator in self._evaluators:
                # if the model component drifts out of its support the evaluator has
                # has to be updated
                if evaluator.needs_update:
                    evaluator.update(self.exposure, self.psf, self.edisp, self._geom)

                npred = evaluator.compute_npred()

                # avoid slow fancy indexing, when the shape is equivalent
                if npred.data.shape == npred_total.data.shape:
                    npred_total += npred.data
                else:
                    npred_total.data[evaluator.coords_idx] += npred.data

        return npred_total

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
        residuals = self._compute_residuals(self.counts, self.npred(), method=method)
        return residuals

    def plot_residuals(
        self,
        method="diff",
        smooth_kernel="gauss",
        smooth_radius="0.1 deg",
        region=None,
        figsize=(12, 4),
        **kwargs
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
            plt.ylabel("Residuals ({})".format(label))

            # Overlay spectral extraction region on the spatial residuals
            pix_region = region.to_pixel(wcs=spatial_residuals.geom.wcs)
            pix_region.plot(ax=ax_image)

        return ax_image, ax_spec

    @lazyproperty
    def _counts_data(self):
        return self.counts.data.astype(float)

    def likelihood(self):
        """Total likelihood given the current model parameters.

        """
        counts, npred = self._counts_data, self.npred().data

        if self.mask is not None:
            stat = self._stat_sum(counts[self.mask], npred[self.mask])
        else:
            stat = self._stat_sum(counts.ravel(), npred.ravel())

        return stat

    def fake(self, random_state="random-seed"):
        """
        Simulate fake counts for the current model and reduced irfs.

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
        hdulist += self.counts.to_hdulist(hdu="counts")[exclude_primary]
        hdulist += self.exposure.to_hdulist(hdu="exposure")[exclude_primary]
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
                hdulist += self.edisp.edisp_map.to_hdulist(hdu="EDISP")

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

        return hdulist

    @classmethod
    def from_hdulist(cls, hdulist):
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
        init_kwargs["counts"] = Map.from_hdulist(hdulist, hdu="counts")
        init_kwargs["exposure"] = Map.from_hdulist(hdulist, hdu="exposure")

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
    def read(cls, filename):
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
        return cls.from_hdulist(hdulist)


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
    model : `~gammapy.cube.models.SkyModel`
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

    _cached_properties = [
        "lon_lat",
        "solid_angle",
        "bin_volume",
        "geom_reco",
        "energy_bin_width",
        "energy_edges",
        "energy_center",
    ]

    def __init__(
        self, model=None, exposure=None, psf=None, edisp=None, evaluation_mode="local"
    ):
        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp

        if evaluation_mode not in {"local", "global"}:
            raise ValueError("Invalid evaluation_mode: {!r}".format(evaluation_mode))

        self.evaluation_mode = evaluation_mode

    @property
    def geom(self):
        """True energy map geometry (`~gammapy.maps.MapGeom`)"""
        return self.exposure.geom

    @lazyproperty
    def geom_reco(self):
        """Reco energy map geometry (`~gammapy.maps.MapGeom`)"""
        e_reco_axis = self.edisp.e_reco.copy(name="energy")
        return self.geom_image.to_cube(axes=[e_reco_axis])

    @property
    def geom_image(self):
        """Image map geometry (`~gammapy.maps.MapGeom`)"""
        return self.geom.to_image()

    @lazyproperty
    def energy_center(self):
        """True energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.get_axis_by_name("energy")
        return energy_axis.center[:, np.newaxis, np.newaxis]

    @lazyproperty
    def energy_edges(self):
        """True energy axis bin edges (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.get_axis_by_name("energy")
        return energy_axis.edges[:, np.newaxis, np.newaxis]

    @lazyproperty
    def energy_bin_width(self):
        """Energy axis bin widths (`astropy.units.Quantity`)"""
        return np.diff(self.energy_edges, axis=0)

    @lazyproperty
    def lon_lat(self):
        """Spatial coordinate pixel centers (``lon, lat`` tuple of `~astropy.units.Quantity`).
        """
        coord = self.geom_image.get_coord()
        frame = self.model.frame

        if frame is not None:
            coordsys = "CEL" if frame == "icrs" else "GAL"

            if not coord.coordsys == coordsys:
                coord = coord.to_coordsys(coordsys)

        return (
            u.Quantity(coord.lon, "deg", copy=False),
            u.Quantity(coord.lat, "deg", copy=False),
        )

    @property
    def lon(self):
        return self.lon_lat[0]

    @property
    def lat(self):
        return self.lon_lat[1]

    @lazyproperty
    def solid_angle(self):
        """Solid angle per pixel"""
        return self.geom.solid_angle()

    @lazyproperty
    def bin_volume(self):
        """Map pixel bin volume (solid angle times energy bin width)."""
        omega = self.solid_angle
        de = self.energy_bin_width
        return omega * de

    @property
    def coords(self):
        """Return evaluator coords"""
        lon, lat = self.lon_lat
        if self.edisp:
            energy = self.edisp.e_reco.center[:, np.newaxis, np.newaxis]
        else:
            energy = self.energy_center

        return {"lon": lon.value, "lat": lat.value, "energy": energy}

    @property
    def needs_update(self):
        """Check whether the model component has drifted away from its support."""
        if self.exposure is None:
            update = True
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
        geom : `gammapy.maps.MapGeom`
            Reference geometry of the data.
        """
        log.debug("Updating model evaluator")
        # cache current position of the model component
        self._init_position = self.model.position

        # TODO: lookup correct Edisp for this component
        self.edisp = edisp

        # TODO: lookup correct PSF for this component
        self.psf = psf

        if self.evaluation_mode == "local":
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
                    "Position {} of model component is outside the image boundaries."
                    " Please check the starting values or position parameter boundaries of the model.".format(
                        self.model.position
                    )
                )

            # Reset cached quantities
            for cached_property in self._cached_properties:
                self.__dict__.pop(cached_property, None)

            self.coords_idx = geom.coord_to_idx(self.coords)[::-1]

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
        coord = (self.lon, self.lat, self.energy_center)
        dnde = self.model.evaluate(*coord)
        return dnde

    def compute_flux(self):
        """Compute model integral flux over map pixel volumes.

        For now, we simply multiply dnde with bin volume.
        """
        dnde = self.compute_dnde()
        volume = self.bin_volume
        flux = dnde * volume
        return flux

    def apply_exposure(self, flux):
        """Compute npred cube

        For now just divide flux cube by exposure
        """
        npred = (flux * self.exposure.quantity).to_value("")
        return Map.from_geom(self.geom, data=npred, unit="")

    def apply_psf(self, npred):
        """Convolve npred cube with PSF"""
        return npred.convolve(self.psf)

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
        loc = npred.geom.get_axis_index_by_name("energy")
        data = np.rollaxis(npred.data, loc, len(npred.data.shape))
        data = np.dot(data, self.edisp.pdf_matrix)
        data = np.rollaxis(data, -1, loc)
        return Map.from_geom(self.geom_reco, data=data, unit="")

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
