# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Morphological models for astrophysical gamma-ray sources - new implementation
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from ...utils.modeling import Parameter, ParameterList
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import Angle, Longitude, Latitude
import numpy as np
import astropy.units as u

__all__ = [
    'SkySpatialModel',
    'SkyGaussian2D',
]


class SkySpatialModel(object):
    """SkySpatial model base class.
    """

    def __str__(self):
        ss = self.__class__.__name__
        ss += '\n\nParameters: \n\n\t'

        table = self.parameters.to_table()
        ss += '\n\t'.join(table.pformat())

        if self.parameters.covariance is not None:
            ss += '\n\nCovariance: \n\n\t'
            covar = self.parameters.covariance_to_table()
            ss += '\n\t'.join(covar.pformat())
        return ss

    def __call__(self, lon, lat):
        """Call evaluate method of derived classes"""
        kwargs = dict()
        for par in self.parameters.parameters:
            kwargs[par.name] = par.quantity

        return self.evaluate(lon, lat, **kwargs)

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)


class SkyGaussian2D(SkySpatialModel):
    r"""Two-dimensional symmetric Gaussian model.

    .. math::

        \phi(x, y) = \frac{1}{2\pi\sigma^2} \exp{\left(-\frac{1}{2}
            \frac{\theta^2}{\sigma^2}\right)}

    where :math:`\theta` is the sky separation

    Parameters
    ----------
    lon_mean : `~astropy.coordiantes.Longitude`
        :math:`x_0`
    lat_mean : `~astropy.coordinates.Latitude`
        :math:`y_0`
    sigma : `~astropy.coordinates.Angle`
        :math:`\sigma`
    """

    def __init__(self, lon_mean, lat_mean, sigma):
        self.parameters = ParameterList([
            Parameter('lon_mean', Longitude(lon_mean)),
            Parameter('lat_mean', Latitude(lat_mean)),
            Parameter('sigma', Angle(sigma))
        ])

    @staticmethod
    def evaluate(lon, lat, lon_mean, lat_mean, sigma):
        """Evaluate the model (static function)."""
        sep = angular_separation(lon, lat, lon_mean, lat_mean)
        fact = (sep / sigma).to('').value
        val = np.exp(-0.5 * fact **2) / (2 * np.pi * sigma**2)
        return val
