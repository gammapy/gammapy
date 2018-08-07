# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Spatial models for astrophysical gamma-ray sources.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import abc
import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import Angle, Longitude, Latitude
from ...extern import six
from ...utils.modeling import Parameter, ParameterList
from ...maps import Map

__all__ = [
    'SkySpatialModel',
    'SkyPointSource',
    'SkyGaussian',
    'SkyDisk',
    'SkyShell',
    'SkyDiffuseConstant',
    'SkyDiffuseMap',
]


@six.add_metaclass(abc.ABCMeta)
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
        """Call evaluate method"""
        kwargs = dict()
        for par in self.parameters.parameters:
            kwargs[par.name] = par.quantity

        return self.evaluate(lon, lat, **kwargs)

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)


class SkyPointSource(SkySpatialModel):
    r"""Point Source.

    .. math::

        \phi(lon, lat) = \delta{(lon - lon_0, lat - lat_0)}

    A tolerance of 1 arcsecond is accepted for numerical stability

    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`lon_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`lat_0`
    """

    def __init__(self, lon_0, lat_0):
        self.parameters = ParameterList([
            Parameter('lon_0', Longitude(lon_0)),
            Parameter('lat_0', Latitude(lat_0))
        ])

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0):
        """Evaluate the model (static function)."""

        wrapval = lon_0 + 180 * u.deg
        lon = Angle(lon).wrap_at(wrapval)

        _, grad_lon = np.gradient(lon)
        grad_lat, _ = np.gradient(lat)
        lon_diff = np.abs((lon - lon_0) / grad_lon)
        lat_diff = np.abs((lat - lat_0) / grad_lat)

        lon_val = np.select([lon_diff < 1], [1 - lon_diff], 0) / np.abs(grad_lon)
        lat_val = np.select([lat_diff < 1], [1 - lat_diff], 0) / np.abs(grad_lat)
        val = lon_val * lat_val
        return val.to('sr-1')


class SkyGaussian(SkySpatialModel):
    r"""Two-dimensional symmetric Gaussian model.

    .. math::

        \phi(lon, lat) = \frac{1}{2\pi\sigma^2} \exp{\left(-\frac{1}{2}
            \frac{\theta^2}{\sigma^2}\right)}

    where :math:`\theta` is the sky separation

    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
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
        sep = sep.to('rad').value
        sigma = sigma.to('rad').value

        norm = 1 / (2 * np.pi * sigma ** 2)
        exponent = -0.5 * (sep / sigma) ** 2
        val = norm * np.exp(exponent)

        return val * u.Unit('sr-1')


class SkyDisk(SkySpatialModel):
    r"""Constant radial disk model.

    .. math::

        \phi(lon, lat) = \frac{1}{2 \pi (1 - \cos{r}) } \cdot
                \begin{cases}
                    1 & \text{for } \theta \leq r_0 \\
                    0 & \text{for } \theta < r_0
                \end{cases}

    where :math:`\theta` is the sky separation

    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`lon_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`lat_0`
    r_0 : `~astropy.coordinates.Angle`
        :math:`r_0`
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
        sep = sep.to('rad').value
        r_0 = r_0.to('rad').value

        norm = 1. / (2 * np.pi * (1 - np.cos(r_0)))
        val = np.where(sep <= r_0, norm, 0)

        return val * u.Unit('sr-1')


class SkyShell(SkySpatialModel):
    r"""Shell model

    .. math::

        \phi(lon, lat) = \frac{3}{2 \pi (r_{out}^3 - r_{in}^3)} \cdot
                \begin{cases}
                    \sqrt{r_{out}^2 - \theta^2} - \sqrt{r_{in}^2 - \theta^2} &
                                 \text{for } \theta \lt r_{in} \\
                    \sqrt{r_{out}^2 - \theta^2} &
                                 \text{for } r_{in} \leq \theta \lt r_{out} \\
                    0 & \text{for } \theta > r_{out}
                \end{cases}

    where :math:`\theta` is the sky separation and :math:`r_out = r_in` + width

    Note that the normalization is a small angle approximation,
    although that approximation is still very good even for 10 deg radius shells.

    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`lon_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`lat_0`
    radius : `~astropy.coordinates.Angle`
        Inner radius, :math:`r_{in}`
    width : `~astropy.coordinates.Angle`
        Shell width
    """

    def __init__(self, lon_0, lat_0, radius, width):
        self.parameters = ParameterList([
            Parameter('lon_0', Longitude(lon_0)),
            Parameter('lat_0', Latitude(lat_0)),
            Parameter('radius', Angle(radius)),
            Parameter('width', Angle(width))
        ])

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, radius, width):
        """Evaluate the model (static function)."""
        sep = angular_separation(lon, lat, lon_0, lat_0)
        sep = sep.to('rad').value
        r_i = radius.to('rad').value
        r_o = (radius + width).to('rad').value

        norm = 3 / (2 * np.pi * (r_o ** 3 - r_i ** 3))

        with np.errstate(invalid='ignore'):
            val_out = np.sqrt(r_o ** 2 - sep ** 2)
            val_in = val_out - np.sqrt(r_i ** 2 - sep ** 2)
            val = np.select([sep < r_i, sep < r_o], [val_in, val_out])

        return norm * val * u.Unit('sr-1')


class SkyDiffuseConstant(SkySpatialModel):
    """Spatially constant (isotropic) spatial model.

    Parameters
    ----------
    value : `~astropy.units.Quantity`
        Value
    """

    def __init__(self, value=1):
        self.parameters = ParameterList([
            Parameter('value', value),
        ])

    @staticmethod
    def evaluate(lon, lat, value):
        return value


class SkyDiffuseMap(SkySpatialModel):
    """Spatial sky map template model (2D).

    This is for a 2D image.
    The map unit is assumed to be ``sr-1``.
    Use `~gammapy.cube.SkyDiffuseCube` for 3D cubes with an energy axis.

    Parameters
    ----------
    map : `~gammapy.map.Map`
        Map template
    norm : float
        Norm parameter (multiplied with map values)
    meta : dict, optional
        Meta information, meta['filename'] will be used for serialization
    """

    def __init__(self, map, norm=1, meta=None):
        self.map = map
        self._interp_opts = {'fill_value': 0, 'interp': 'linear'}
        self.parameters = ParameterList([
            Parameter('norm', norm),
        ])
        self.meta = dict() if meta is None else meta

    @classmethod
    def read(cls, filename, **kwargs):
        """Read spatial template model from FITS image.

        Parameters
        ----------
        filename : str
            FITS image filename.
        """
        m = Map.read(filename, **kwargs)
        return cls(m)

    def evaluate(self, lon, lat, norm):
        """Evaluate model."""
        coord = {
            'lon': lon.to('deg').value,
            'lat': lat.to('deg').value,
        }
        val = self.map.interp_by_coord(coord, **self._interp_opts)
        return norm * val * u.Unit('sr-1')
