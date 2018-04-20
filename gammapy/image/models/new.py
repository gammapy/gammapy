# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Morphological models for astrophysical gamma-ray sources - new implementation
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import abc
import numpy as np
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation
from astropy.coordinates import Angle, Longitude, Latitude, SkyCoord
from astropy.utils import lazyproperty
from ...extern import six
from ...utils.modeling import Parameter, ParameterList
from ...maps import Map, MapCoord

__all__ = [
    'SkySpatialModel',
    'SkyGaussian2D',
    'SkyPointSource',
    'SkyDisk2D',
    'SkyShell2D',
    'SkyTemplate2D',
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

    @staticmethod
    @abc.abstractmethod
    def evaluate():
        """Model evaluation"""
        pass

    def __call__(self, lon, lat):
        """Call evaluate method"""
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
        fact = (sep / sigma).to('').value
        val = np.exp(-0.5 * fact ** 2) / (2 * np.pi * sigma ** 2)
        return val


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
        tolerance = 1 * u.arcsec
        sep = angular_separation(lon, lat, lon_0, lat_0)
        val = 1 if sep < tolerance else 0
        return val


class SkyDisk2D(SkySpatialModel):
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
        norm = 1. / (2 * np.pi * (1 - np.cos(r_0)))
        val = np.where(sep <= r_0, norm, 0)
        return val / u.deg ** 2


class SkyShell2D(SkySpatialModel):
    r"""Shell model

    .. math::

        \phi(lon, lat) = \frac{3}{2 \pi (r_o^3 - r_i^3)} \cdot
                \begin{cases}
                    \sqrt{r_o^2 - \theta^2} - \sqrt{r_i^2 - \theta^2} &
                                 \text{for } \theta \lt r_i \\
                    \sqrt{r_o^2 - \theta^2} & 
                                 \text{for } r_i \leq \theta \lt r_o \\
                    0 & \text{for } \theta > r_o
                \end{cases}

    where :math:`\theta` is the sky separation.

    Note that the normalization is a small angle approximation,
    although that approximation is still very good even for 10 deg radius shells.

    Parameters
    ----------
    lon_0 : `~astropy.coordinates.Longitude`
        :math:`lon_0`
    lat_0 : `~astropy.coordinates.Latitude`
        :math:`lat_0`
    r_i : `~astropy.coordinates.Angle`
        :math:`r_i`
    r_o : `~astropy.coordinates.Angle`
        :math:`r_o`
    """

    def __init__(self, lon_0, lat_0, r_i, r_o):
        self.parameters = ParameterList([
            Parameter('lon_0', Longitude(lon_0)),
            Parameter('lat_0', Latitude(lat_0)),
            Parameter('r_i', Angle(r_i)),
            Parameter('r_o', Angle(r_o))
        ])

    @staticmethod
    def evaluate(lon, lat, lon_0, lat_0, r_i, r_o):
        """Evaluate the model (static function)."""
        sep = angular_separation(lon, lat, lon_0, lat_0)
        term1 = np.sqrt(r_o ** 2 - sep ** 2)
        term2 = term1 - np.sqrt(r_i ** 2 - sep ** 2)
        norm = 3 / (2 * np.pi * (r_o ** 3 - r_i ** 3))

        val = np.where(sep < r_i,
                       term2.to('deg').value,
                       term1.to('deg').value) * u.deg
        val[sep > r_o] = 0

        return norm * val


class SkyTemplate2D(SkySpatialModel):
    """Two dimensional table model.

    Parameters
    ----------
    image : `~gammapy.image.SkyImage`
        Template
    """

    def __init__(self, image):
        self._image = self._norm_image(image)
        self.parameters = ParameterList([])

    @property
    def image(self):
        """Normalized template"""
        return self._image

    @staticmethod
    def _norm_image(image):
        """Norm image"""
        solid_angle = np.sum(image.geom.to_image().solid_angle().to('deg2'))
        image.quantity = image.data / (image.data.sum() * solid_angle)
        return image

    @classmethod
    def read(cls, filename, **kwargs):
        """Read spatial template model from FITS image.

        Parameters
        ----------
        filename : str
            Fits image filename.
        """
        template = Map.read(filename, **kwargs)
        return cls(template)

    def evaluate(self, lon, lat):
        #TODO : Don't hardcode galactic frame
        coord = MapCoord.create(
            dict(lon=lon.to('deg').value,
                 lat=lat.to('deg').value),
            coordsys = 'GAL'
        )
        values = self.image.interp_by_coord(coord)
        return values * self.image.unit
