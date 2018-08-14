# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.utils import lazyproperty
import astropy.units as u
from ..utils.fitting import fit_iminuit
from ..stats import cash
from ..maps import Map

__all__ = [
    'MapFit',
    'MapEvaluator',
]


class MapFit(object):
    """Perform sky model likelihood fit on maps.

    This is the first go at such a class. It's geared to the
    `~gammapy.spectrum.SpectrumFit` class which does the 1D spectrum fit.

    Parameters
    ----------
    model : `~gammapy.cube.SkyModel`
        Fit model
    counts : `~gammapy.maps.WcsNDMap`
        Counts cube
    exposure : `~gammapy.maps.WcsNDMap`
        Exposure cube
    background : `~gammapy.maps.WcsNDMap`
        Background Cube
    background : `~gammapy.maps.WcsNDMap`
        Exclusion mask.
    psf : `~gammapy.cube.PSFKernel`
        PSF kernel
    edisp : `~gammapy.irf.EnergyDispersion`
        Energy dispersion
    """

    def __init__(self, model, counts, exposure, background=None, exclusion_mask=None, psf=None, edisp=None):
        self.model = model
        self.counts = counts
        self.exposure = exposure
        self.background = background
        self.exclusion_mask = exclusion_mask
        self.psf = psf
        self.edisp = edisp

        self._npred = None
        self._stat = None
        self._minuit = None

        # applying the exclusion mask to the exposure is currently the fastest
        # way of applying a mask for fitting
        if exclusion_mask:
            data = exposure.data * (1 - exclusion_mask.data.astype(int))
            exposure = exposure.copy(data=data)

        self.evaluator = MapEvaluator(
            model=self.model,
            exposure=exposure,
            background=self.background,
            psf=self.psf,
            edisp=self.edisp,
        )

    @property
    def npred(self):
        """Predicted counts cube"""
        return self._npred

    @property
    def stat(self):
        """Fit statistic per bin"""
        return self._stat

    @property
    def minuit(self):
        """`~iminuit.Minuit` object"""
        return self._minuit

    def compute_npred(self):
        """Compute predicted counts"""
        self._npred = self.evaluator.compute_npred()

    def compute_stat(self):
        """Compute fit statistic per bin"""
        self._stat = cash(
            n_on=self.counts.data,
            mu_on=self.npred
        )

    def total_stat(self, parameters):
        """Likelihood for a given set of model parameters"""
        self.model.parameters = parameters
        self.compute_npred()
        self.compute_stat()
        return np.sum(self.stat, dtype=np.float64)

    def fit(self, opts_minuit=None):
        """Run the fit

        Parameters
        ----------
        opts_minuit : dict (optional)
            Options passed to `iminuit.Minuit` constructor
        """
        minuit = fit_iminuit(parameters=self.model.parameters,
                             function=self.total_stat,
                             opts_minuit=opts_minuit)
        self._minuit = minuit


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

    def __init__(self, model=None, exposure=None, background=None, psf=None, edisp=None):
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
        return energy

    @lazyproperty
    def energy_edges(self):
        """Energy axis bin edges (`~astropy.units.Quantity`)"""
        energy_axis = self.geom.axes[0]
        energy = energy_axis.edges * energy_axis.unit
        return energy

    @lazyproperty
    def energy_bin_width(self):
        """Energy axis bin widths (`astropy.units.Quantity`)"""
        return np.diff(self.energy_edges)

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
        de = de[:, np.newaxis, np.newaxis]
        return omega * de

    def compute_dnde(self):
        """Compute model differential flux at map pixel centers.

        Returns
        -------
        model_map : `~gammapy.map.Map`
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
        return flux.to('cm-2 s-1')

    def apply_exposure(self, flux):
        """Compute npred cube

        For now just divide flux cube by exposure
        """
        npred = Map.from_geom(self.geom, unit='')
        npred.data = (flux * self.exposure.quantity).to('').value
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
