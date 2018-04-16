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
    'SkyPointSource',
    'SkyDisk2D',
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
    lon_0 : `~astropy.coordiantes.Longitude`
        :math:`lon_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`lat_0`
    sigma : `~astropy.coordinates.Angle`
        :math:`\sigma`
    """

    def __init__(self, lon_0, lat_0, sigma):
        self.parameters = ParameterList([
            Parameter('lon_0', Longitude(lon_0)),
            Parameter('lat_0', Latitude(lat_0)),
            Parameter('sigma', Angle(sigma))
        ])

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, sigma):
        """Evaluate the model (static function)."""
        sep = angular_separation(lon, lat, lon_0, lat_0)
        fact = (sep / sigma).to('').value
        val = np.exp(-0.5 * fact ** 2) / (2 * np.pi * sigma**2)
        return val


class SkyPointSource(SkySpatialModel):
    r"""Point Source.

    .. math::

        \phi(x, y) = \delta{(lon - lon_0, lat - lat_0)}

    A tolerance of 1 arcsecond is accepted for numerical stability

    Parameters
    ----------
    lon_0: `~astropy.coordiantes.Longitude`
        : math: `lon_0`
    lat_0: `~astropy.coordinates.Latitude`
        : math: `lat_0`
    """

    def __init__(self, lon_0, lat_0):
        self.parameters = ParameterList([
            Parameter('lon_0', Longitude(lon_0)),
            Parameter('lat_0', Latitude(lat_0))
        ])

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0):
        """Evaluate the model (static function)."""
        tolerance = 1 * u.arcsec
        sep = angular_separation(lon, lat, lon_0, lat_0)
        val = 1 if sep < tolerance else 0
        return val


class SkyDisk2D(SkySpatialModel):
    r"""Constant radial disk model.

    .. math::

        f(r) = \frac{1}{2 \pi (1 - \cos{r}) } \cdot
                \begin{cases}
                    1 & \text{for} \theta \leq r_0 \\
                    0 & \text{else}
                \end{cases}

    where :math:`\theta` is the sky separation

    Parameters
    ----------
    lon_0: `~astropy.coordiantes.Longitude`
        : math: `lon_0`
    lat_0: `~astropy.coordinates.Latitude`
        : math: `lat_0`
    r_0: `~astropy.coordinates.Angle`
        : math: `r_0`
    """

    def __init__(self, lon_0, lat_0, r_0):
        self.parameters = ParameterList([
            Parameter('lon_0', Longitude(lon_0)),
            Parameter('lat_0', Latitude(lat_0)),
            Parameter('r_0', Angle(r_0))
        ])

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, r_0):
        """Evaluate the model (static function)."""
        sep = angular_separation(lon, lat, lon_0, lat_0)
        norm = 2 * np.pi * (1 - np.cos(r_0))
        val = 1./norm if sep <= r_0 else 0
        return val / u.deg ** 2
