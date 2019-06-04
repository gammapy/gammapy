# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u
from astropy.nddata.utils import NoOverlapError
from ..utils.fitting import Parameters, Dataset
from ..stats import cash, cstat, cash_sum_cython, cstat_sum_cython
from ..maps import Map
from .models import SkyModel, SkyModels

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
    """

    def __init__(
        self,
        model,
        counts=None,
        exposure=None,
        mask_fit=None,
        psf=None,
        edisp=None,
        background_model=None,
        likelihood="cash",
        evaluation_mode="local",
        mask_safe=None,
    ):
        if mask_fit is not None and mask_fit.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.evaluation_mode = evaluation_mode
        self.model = model
        self.counts = counts
        self.exposure = exposure
        self.mask_fit = mask_fit
        self.psf = psf
        self.edisp = edisp
        self.background_model = background_model
        self.mask_safe = mask_safe

        if likelihood == "cash":
            self._stat = cash
            self._stat_sum = cash_sum_cython
        elif likelihood == "cstat":
            self._stat = cstat
            self._stat_sum = cstat_sum_cython
        else:
            raise ValueError("Invalid likelihood: {!r}".format(likelihood))

    @property
    def model(self):
        """Sky model to fit (`~gammapy.cube.SkyModel` or `~gammapy.cube.SkyModels`)"""
        return self._model

    @model.setter
    def model(self, model):
        if isinstance(model, SkyModel):
            model = SkyModels([model])

        self._model = model

        evaluators = []

        for component in model.skymodels:
            evaluator = MapEvaluator(component, evaluation_mode=self.evaluation_mode)
            evaluators.append(evaluator)

        self._evaluators = evaluators

    @property
    def parameters(self):
        """List of parameters (`~gammapy.utils.fitting.Parameters`)"""
        if self.background_model:
            parameters = Parameters(
                self.model.parameters.parameters
                + self.background_model.parameters.parameters
            )
        else:
            parameters = Parameters(self.model.parameters.parameters)
        return parameters

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
        if self.background_model:
            npred_total = self.background_model.evaluate()
        else:
            npred_total = Map.from_geom(self._geom)

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

        return {"lon": lon.value, "lat": lat.value, "energy": energy.value}

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
