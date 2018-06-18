# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.convolution.utils import discretize_oversample_2D
from astropy.coordinates import Angle
from ..image.models import Gauss2DPDF
from ..irf import TablePSF
from ..maps import WcsNDMap

__all__ = [
    'WcsNDMapPSFKernel',
]


class WcsNDMapPSFKernel(object):
    """PSF kernel for `~gammapy.maps.WcsNDMap`.

    This is a container class to store a PSF kernel
    that can be used to convolve `~gammapy.maps.WcsNDMap` objects.
    It is usually computed for a given sky position as the
    mean PSF for a list of observations.

    It is very preliminary, for now the goal is just to get the
    case of 3D maps with an energy axis working.

    For now, we store the PSF itself in a `~gammapy.maps.WcsNDMap`
    object. That should be considered an implementation detail
    that we might or might not keep later. For now it saves us the
    trouble of developing code for extra axes and FITS I/O.

    Parameters
    ----------
    psf_map : `~gammapy.maps.WcsNDMap`
        PSF data
    """

    def __init__(self, psf_map):
        self._psf_map = psf_map

    @classmethod
    def from_map(cls, psf_map):
        return cls(psf_map)

    @classmethod
    def read(cls, *args, **kwargs):
        psf_map = WcsNDMap.read(*args, **kwargs)
        return cls.from_map(psf_map)

    @classmethod
    def from_gauss(cls, geom, sigma, rad_max=None, containment_fraction=0.99):
        """Create Gaussian PSF.

        This is used for testing and examples.

        The map geometry parameters (pixel size, energy bins)
        are taken from `geom`.

        The Gaussian width ``sigma`` can be a scalar,
        or an array if it should vary along the energy axis.

        Parameters
        ----------
        geom : `~gammapy.map.WcsGeom`
            Map geometry
        sigma : `~astropy.coordinates.Angle`
            Gaussian width
        rad_max : `~astropy.coordinates.Angle`
            Desired kernel width
        """
        sigma = Angle(sigma)
        if rad_max is None:
            rad_max = Gauss2DPDF(sigma.to('deg').value).containment_radius(
                containment_fraction=containment_fraction
            )
        rad_max = Angle(rad_max)
        pixel_size = np.abs(geom.wcs.wcs.cdelt)
        psf = TablePSF.from_shape(shape='gauss', width=sigma)
        npix = int(rad_max.radian / pixel_size.radian)
        data = table_psf_to_kernel_array(psf, npix)

        m = WcsNDMap.from_geom(geom)
        m.data = data

        return cls.from_map(m)

    def to_map(self):
        return self._psf_map

    def write(self, *args, **kwargs):
        psf_map = self.to_map()
        psf_map.write(*args, **kwargs)


def table_psf_to_kernel_array(psf, npix, pixel_size, normalize=True, factor=5):
    """Compute oversampled

    Parameters
    ----------
    TODO
    """
    def f(x, y):
        rad = np.sqrt(x * x + y * y) * pixel_size
        return psf.evaluate(rad)

    pix_range = (-npix, npix + 1)

    kernel = discretize_oversample_2D(
        f, x_range=pix_range, y_range=pix_range, factor=factor
    )
    if normalize:
        kernel = kernel / kernel.sum()

    return kernel
