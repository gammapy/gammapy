# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u
from ..utils.fitting import Fit
from ..stats import cash
from ..maps import Map

__all__ = ["MapFit", "MapEvaluator"]


class MapFit(Fit):
    """Perform sky model likelihood fit on maps.

    This is the first go at such a class. It's geared to the
    `~gammapy.spectrum.SpectrumFit` class which does the 1D spectrum fit.

    Parameters
    ----------
    model : `~gammapy.cube.models.SkyModel`
        Fit model
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    background : `~gammapy.maps.WcsNDMap`
        Background Cube
    mask : `~gammapy.maps.WcsNDMap`
        Mask to apply for the fit. All the pixels that contain 1 or True are included
        in the fit, all others are ignored.
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(
        self, model, counts, exposure, background=None, mask=None, psf=None, edisp=None
    ):
        if mask is not None and mask.data.dtype != np.dtype("bool"):
            raise ValueError("mask data must have dtype bool")

        # Create a copy of the input model that is owned and modified by MapFit
        self._model = model.copy()
        self.counts = counts
        self.exposure = exposure
        self.background = background
        self.mask = mask
        self.psf = psf
        self.edisp = edisp

        self.evaluator = MapEvaluator(
            model=self._model,
            exposure=exposure,
            background=self.background,
            psf=self.psf,
            edisp=self.edisp,
        )

    @property
    def stat(self):
        """Likelihood per bin given the current model parameters"""
        npred = self.evaluator.compute_npred()
        return cash(n_on=self.counts.data, mu_on=npred)

    def total_stat(self, parameters):
        """Total likelihood given the current model parameters"""
        self._model.parameters = parameters
        if self.mask:
            stat = self.stat[self.mask.data]
        else:
            stat = self.stat
        return np.sum(stat, dtype=np.float64)


class MapEvaluator(object):
    """Sky model evaluation on maps.

    This is a first attempt to compute flux as well as predicted counts maps.

    The basic idea is that this evaluator is created once at the start
    of the analysis, and pre-computes some things.
    It it then evaluated many times during likelihood fit when model parameters
    change, re-using pre-computed quantities each time.
    At the moment it does some things, e.g. cache and re-use energy and coordinate grids,
    but overall it is not an efficient implementation yet.

    For now, we only make it work for 3D WCS maps with an energy axis.
    No HPX, no other axes, those can be added later here or via new
    separate model evaluator classes.

    We should discuss how to organise the model and IRF evaluation code,
    and things like integrations and convolutions in a good way.

    Parameters
    ----------
    model : `~gammapy.cube.models.SkyModel`
        Sky model
    exposure : `~gammapy.maps.Map`
        Exposure map
    background : `~gammapy.maps.Map`
        background map
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(
        self, model=None, exposure=None, background=None, psf=None, edisp=None
    ):
        self.model = model
        self.exposure = exposure
        self.background = background
        self.psf = psf
        self.edisp = edisp

    @lazyproperty
    def geom(self):
        return self.exposure.geom

    @lazyproperty
    def geom_image(self):
        return self.geom.to_image()

    @lazyproperty
    def energy_center(self):
        """Energy axis bin centers (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.axes[0]
        energy = energy_axis.center * energy_axis.unit
        return energy[:, np.newaxis, np.newaxis]

    @lazyproperty
    def energy_edges(self):
        """Energy axis bin edges (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.axes[0]
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
        lon, lat = self.geom_image.get_coord()
        return lon * u.deg, lat * u.deg

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
        return flux.to("cm-2 s-1")

    def apply_exposure(self, flux):
        """Compute npred cube

        For now just divide flux cube by exposure
        """
        npred = Map.from_geom(self.geom, unit="")
        npred.data = (flux * self.exposure.quantity).to("").value
        return npred

    def apply_psf(self, npred):
        """Convolve npred cube with PSF"""
        return npred.convolve(self.psf)

    def apply_edisp(self, data):
        """Convolve map data with energy dispersion."""
        data = np.rollaxis(data, 0, 3)
        data = np.dot(data, self.edisp.pdf_matrix)
        return np.rollaxis(data, 2, 0)

    def compute_npred(self):
        """Evaluate model predicted counts.
        """
        flux = self.compute_flux()
        npred = self.apply_exposure(flux)
        if self.psf is not None:
            npred = self.apply_psf(npred)
        # TODO: discuss and decide whether we need to make map objects in `apply_aeff` and `apply_psf`.
        if self.edisp is not None:
            npred.data = self.apply_edisp(npred.data)
        if self.background:
            npred.data += self.background.data
        return npred.data
