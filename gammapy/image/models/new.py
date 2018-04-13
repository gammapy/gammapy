# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Morphological models for astrophysical gamma-ray sources - new implementation
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from ...utils.modeling import Parameter, ParameterList
import numpy as np

__all__ = [
    'SkySpatialModel',
    'SkyGaussian2D',
]


class SkySpatialModel(object):
    """SkySpatial model base class.
    """

    def __repr__(self):
        fmt = '{}()'
        return fmt.format(self.__class__.__name__)

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

        \phi(x, y) = \phi_0 \cdot \exp{\left(-\frac{1}{2\sigma^2}
            \left((x-x_0)^2+(y-y_0)^2\right)\right)}

    Parameters
    ----------
    amplitude : `~astropy.units.Quantity`
        :math:`\phi_0`
    lon_mean : `~astropy.units.Quantity`
        :math:`x_0`
    lat_mean : `~astropy.units.Quantity`
        :math:`y_0`
    sigma : `~astropy.units.Quantity`
        :math:`\sigma`
    """

    def __init__(self, amplitude, lon_mean, lat_mean, sigma):
        self.parameters = ParameterList([
            Parameter('amplitude', amplitude),
            Parameter('lon_mean', lon_mean),
            Parameter('lat_mean', lat_mean),
            Parameter('sigma', sigma)
        ])

    @staticmethod
    def evaluate(lon, lat, amplitude, lon_mean, lat_mean, sigma):
        """Evaluate the model (static function)."""
        exp_ = (lon - lon_mean) ** 2 + (lat - lat_mean) ** 2
        val = amplitude * np.exp((- 1 / (2 * sigma ** 2)) * exp_)
        return val
