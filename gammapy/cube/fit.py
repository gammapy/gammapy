# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u
from ..utils.fitting import Fit, Parameters
from ..stats import cash
from ..maps import Map, MapAxis

__all__ = ["MapEvaluator", "MapDataset"]


class MapDataset:
    """Perform sky model likelihood fit on maps.

    Parameters
    ----------
    model : `~gammapy.cube.models.SkyModel` or `~gammapy.cube.models.SkyModels`
        Source sky models.
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    mask : `~gammapy.maps.WcsNDMap`
        Mask to apply to the likelihood.
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    background_model: `~gammapy.cube.models.BackgroundModel` or `~gammapy.cube.models.BackgroundModel`
        Background models to use for the fit.
    likelihood : {"cash"}
	    Likelihood function to use for the fit.
    """

    def __init__(
        self,
        model,
        counts=None,
        exposure=None,
        mask=None,
        psf=None,
        edisp=None,
        background_model=None,
    ):
        if mask is not None and mask.data.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        self.model = model
        self.counts = counts
        self.exposure = exposure
        self.mask = mask
        self.psf = psf
        self.edisp = edisp
        self.background_model = background_model
        if background_model:
            self.parameters = Parameters(
                self.model.parameters.parameters +
                self.background_model.parameters.parameters
            )
        else:
            self.parameters = Parameters(self.model.parameters.parameters)

        self.evaluator = MapEvaluator(
            model=self.model, exposure=exposure, psf=self.psf, edisp=self.edisp
        )

    @property
    def data_shape(self):
        """Shape of the counts data"""
        return self.counts.data.shape

    def npred(self):
        """Returns npred map (model + background)"""
        model_npred = self.evaluator.compute_npred()
        back_npred = self.background_model.evaluate()
        total_npred = model_npred.data + back_npred.data
        return back_npred.copy(data=total_npred)
        # TODO: return model_npred + back_npred
        # There is some bug: edisp.e_reco.unit is dimensionless
        # thus map arithmetic does not work.

    def likelihood_per_bin(self):
        """Likelihood per bin given the current model parameters"""
        return cash(n_on=self.counts.data, mu_on=self.npred().data)

    def likelihood(self, parameters, mask=None):
        """Total likelihood given the current model parameters.

        Parameters
        ----------
        mask : `~numpy.ndarray`
            Mask to be combined with the dataset mask.
        """
        if self.mask is None and mask is None:
            stat = self.likelihood_per_bin()
        elif self.mask is None:
            stat = self.likelihood_per_bin()[mask]
        elif mask is None:
            stat = self.likelihood_per_bin()[self.mask.data]
        else:
            stat = self.likelihood_per_bin()[mask & self.mask.data]
        return np.sum(stat, dtype=np.float64)


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
    background : `~gammapy.maps.Map`
        Background map
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(self, model=None, exposure=None, psf=None, edisp=None):
        self.model = model
        self.exposure = exposure
        self.psf = psf
        self.edisp = edisp

        self.parameters = Parameters(self.model.parameters.parameters)

    @lazyproperty
    def geom(self):
        """This will give the energy axes in e_true"""
        return self.exposure.geom

    @lazyproperty
    def geom_reco(self):
        edges = self.edisp.e_reco.bins
        e_reco_axis = MapAxis.from_edges(
            edges=edges, name="energy",
            unit=self.edisp.e_reco.unit,
            interp=self.edisp.e_reco.interpolation_mode)
        return self.geom_image.to_cube(axes=[e_reco_axis])

    @lazyproperty
    def geom_image(self):
        return self.geom.to_image()

    @lazyproperty
    def energy_center(self):
        """True energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.get_axis_by_name("energy")
        energy = energy_axis.center * energy_axis.unit
        return energy[:, np.newaxis, np.newaxis]

    @lazyproperty
    def energy_edges(self):
        """Energy axis bin edges (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.get_axis_by_name("energy")
        energy = energy_axis.edges * energy_axis.unit
        return energy[:, np.newaxis, np.newaxis]

    @lazyproperty
    def energy_bin_width(self):
        """Energy axis bin widths (`astropy.units.Quantity`)"""
        return np.diff(self.energy_edges, axis=0)

    @lazyproperty
    def lon_lat(self):
        """Spatial coordinate pixel centers.

        Returns ``lon, lat`` tuple of `~astropy.units.Quantity`.
        """
        coord = self.geom_image.get_coord()
        frame = self.model.frame

        if frame is not None:
            coordsys = "CEL" if frame == "icrs" else "GAL"

            if not coord.coordsys == coordsys:
                coord = coord.to_coordsys(coordsys)

        return (u.Quantity(coord.lon, "deg", copy=False), u.Quantity(coord.lat, "deg", copy=False))

    @lazyproperty
    def lon(self):
        return self.lon_lat[0]

    @lazyproperty
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
        ---------
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
