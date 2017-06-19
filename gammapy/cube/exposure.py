# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .core import SkyCube

__all__ = [
    'make_exposure_cube',
]


def make_exposure_cube(pointing,
                       livetime,
                       aeff,
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
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area table
    ref_cube : `~gammapy.cube.SkyCube`
        Reference cube used to define geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    expcube : `~gammapy.cube.SkyCube`
        Exposure cube (3D)
    """
    coordinates = ref_cube.sky_image_ref.coordinates()
    offset = coordinates.separation(pointing)
    energy = ref_cube.energies()

    exposure = aeff.data.evaluate(offset=offset, energy=energy)
    exposure *= livetime
    exposure[:, offset >= offset_max] = 0

    return SkyCube(
        data=exposure,
        wcs=ref_cube.wcs,
        energy_axis=ref_cube.energy_axis,
    )


def make_exposure_cube_obs(obs, ref_cube=None):
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

    return make_exposure_cube(obs.pointing, obs.livetime, obs.irfs.aeff2d, ref_cube)
