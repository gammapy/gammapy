# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import SkyCoord, Angle
from .core import SkyCube
from ..utils.energy import EnergyBounds

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
    ref_cube : `~gammapy.data.SkyCube`
        Reference cube used to define geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    expcube : `~gammapy.data.SkyCube`
        Exposure cube (3D)
    """
    coordinates = ref_cube.sky_image_ref.coordinates()
    offset = coordinates.separation(pointing)
    offset = np.clip(offset, Angle(0, 'deg'), offset_max)

    energy = ref_cube.energy_axis.energy
    exposure = aeff2d.evaluate(offset=offset, energy=energy)
    exposure *= livetime

    expcube = SkyCube(data=exposure,
                      wcs=ref_cube.wcs,
                      energy=energy)
    return expcube


def obs_exposure_cube(obs, ref_cube=None):
    """Make exposure cube for a given observation.

    Parameters
    ----------
    obs : `gammapy.data.Observation`
        Observation
    ref_cube : `~gammapy.data.SkyCube`
        Reference cube used to define geometry

    Returns
    -------
    expcube : `~gammapy.data.SkyCube`
        3D exposure
    """
    # TODO: the observation class still needs to be implemented first!
    raise NotImplemented
    if not ref_cube:
        ref_cube = obs.ref_cube

    return exposure_cube(obs.pointing, obs.livetime, obs.irfs.aeff2d, ref_cube)
