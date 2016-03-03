# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord, Angle
from .spectral_cube import SpectralCube


__all__ = [
    'exposure_cube'
]


def exposure_cube(pointing,
                  livetime,
                  aeff2d,
                  ref_cube,
                  offset_max=None,
                  ):
    """Calculate exposure cube.

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff2d : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area table
    ref_cube : `~gammapy.data.SpectralCube`
        Reference cube used to define geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    expcube : `~gammapy.data.SpectralCube`
        Exposure cube (3D)
    """
    # Delayed import to avoid circular import issues
    # TODO: figure out if `gammapy.irf` is a good location for
    # exposure computation functionality, or if this should be
    # moved to `gammapy.data` or `gammapy.spectrum` or ...
    
    ny, nx = ref_cube.data.shape[1:]
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    lon, lat, en = ref_cube.pix2world(xx, yy, 0)
    coord = SkyCoord(lon, lat, frame=ref_cube.wcs.wcs.radesys.lower())  # don't care about energy
    offset = coord.separation(pointing)
    offset = np.clip(offset, Angle(0, 'deg'), offset_max)

    exposure = aeff2d.evaluate(offset, ref_cube.energy)
    exposure = np.rollaxis(exposure, 2)
    exposure *= livetime

    expcube = SpectralCube(data=exposure,
                           wcs=ref_cube.wcs,
                           energy=ref_cube.energy)
    return expcube


def obs_exposure_cube(obs, ref_cube=None):
    """Make exposure cube for a given observation.

    Parameters
    ----------
    obs : `gammapy.data.Observation`
        Observation
    ref_cube : `~gammapy.data.SpectralCube`
        Reference cube used to define geometry

    Returns
    -------
    expcube : `~gammapy.data.SpectralCube`
        3D exposure
    """
    # TODO: the observation class still needs to be implemented first!
    raise NotImplemented
    if not ref_cube:
        ref_cube = obs.ref_cube

    return exposure_cube(obs.pointing, obs.livetime, obs.irfs.aeff2d, ref_cube)
