# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..data import SpectralCube
from astropy.coordinates import SkyCoord, Angle

__all__ = [
    'exposure_cube'
]


def exposure_cube(pointing,
                  livetime,
                  aeff2D,
                  ref_cube):
    """Calculate exposure cube

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        pointing direction
    livetime : `~astropy.units.Quantity`
        livetime
    aeff2D : `~gammapy.irf.EffectiveAreaTable2D`
        effective area table
    ref_cube : `~gammapy.data.SpectralCube`
        reference cube used to define geometry

    Returns
    -------
    expcube : `~gammapy.data.SpectralCube`
        3D exposure
    """
    ny, nx = ref_cube.data.shape[1:]
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    lon, lat, en = ref_cube.pix2world(xx, yy, 0)
    coord = SkyCoord(lon, lat, frame=ref_cube.wcs.wcs.radesys.lower())  # don't care about energy
    offset = coord.separation(pointing)
    offset = np.clip(offset, Angle(0, 'deg'), Angle(2.2, 'deg'))

    exposure = aeff2D.evaluate(offset, ref_cube.energy)
    exposure = np.rollaxis(exposure, 2)
    exposure *= livetime

    expcube = SpectralCube(data=exposure,
                           wcs=ref_cube.wcs,
                           energy=ref_cube.energy)
    return expcube
