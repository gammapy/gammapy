# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import astropy.units as u
from astropy.utils import lazyproperty

__all__ = [
    'SkyModel',
    'SkyModelMapEvaluator',
]


class SkyModel(object):
    """Sky model component.

    This model represents a factorised sky model.

    TODO: add possibility to have a temporal model component also.

    Parameters
    ----------
    spatial_model : `~gammapy.image.models.SpatialModel`
        Spatial model (must be normalised to integrate to 1)
    spectral_model : `~gammapy.spectrum.models.SpectralModel`
        Spectral model
    """

    def __init__(self, spatial_model, spectral_model):
        self.spatial_model = spatial_model
        self.spectral_model = spectral_model

    def __repr__(self):
        fmt = '{}(spatial_model={!r}, spectral_model={!r})'
        return fmt.format(self.__class__.__name__,
                          self.spatial_model, self.spectral_model)

    def __str__(self):
        ss = '{}\n\n'.format(self.__class__.__name__)
        ss += 'spatial_model = {}\n\n'.format(self.spatial_model)
        ss += 'spectral_model = {}\n'.format(self.spectral_model)
        return ss

    def evaluate(self, lon, lat, energy):
        """Evaluate the model at given points.

        Return differential surface brightness cube.
        At the moment in units: ``cm-2 s-1 TeV-1 deg-2``

        Parameters
        ----------
        lon, lat : `~astropy.units.Quantity`
            Spatial coordinates
        energy : `~astropy.units.Quantity`
            Energy coordinate

        Returns
        -------
        value : `~astropy.units.Quantity`
            Model value at the given point.
        """
        val_spatial = self.spatial_model(lon, lat)

        val_spectral = self.spectral_model(energy)
        val_spectral = np.atleast_1d(val_spectral)[:, np.newaxis, np.newaxis]

        val = val_spatial * val_spectral

        return val.to('cm-2 s-1 TeV-1 deg-2')


class SkyModelMapEvaluator(object):
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
    sky_model : `~gammapy.cube.models.SkyModel`
        Sky model
    exposure : `~gammapy.maps.Map`
        Exposure map
    psf : TODO
        PSF or PSF kernel
    """

    def __init__(self, sky_model=None, exposure=None, psf=None):
        self.sky_model = sky_model
        self.exposure = exposure
        self.psf = psf

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
        dnde = self.sky_model.evaluate(*coord)
        return dnde

    def compute_flux(self):
        """Compute model integral flux over map pixel volumes.

        For now, we simply multiply dnde with bin volume.
        """
        dnde = self.compute_dnde()
        volume = self.bin_volume
        flux = dnde * volume
        return flux.to('cm-2 s-1')

    def compute_npred(self):
        """Evaluate model predicted counts.

        For now, we simply multiply flux with exposure.
        """
        flux = self.compute_flux()
        exposure = self.exposure.quantity
        npred = flux * exposure
        return npred.to('').value
