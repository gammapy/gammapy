# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from .core import SkyCube

__all__ = [
    'make_exposure_cube',
    'make_background_cube',
]


def make_exposure_cube(pointing,
                       livetime,
                       aeff,
                       ref_cube,
                       offset_max,
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


def make_background_cube(pointing,
                         obstime,
                         bkg,
                         ref_cube,
                         offset_max,
                         ):
    """Calculate background predicted counts cube.

    This function evaluates the background rate model on
    a sky cube, and then multiplies with the cube bin size,
    computed via `gammapy.cube.SkyCube.bin_size`, resulting
    in a cube with values that contain predicted background
    counts per bin.

    Note that this method isn't very precise if the energy
    bins are large. In that case you might consider implementing
    a more precise method that integrates over energy (e.g. by
    choosing a finer energy binning here and then to group
    energy bins).

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    obstime : `~astropy.units.Quantity`
        Observation time
    bkg : `~gammapy.irf.Background3D`
        Background rate model
    ref_cube : `~gammapy.cube.SkyCube`
        Reference cube used to define geometry
    offset_max : `~astropy.coordinates.Angle`
        Maximum field of view offset.

    Returns
    -------
    background : `~gammapy.cube.SkyCube`
        Background predicted counts sky cube
    """
    coordinates = ref_cube.sky_image_ref.coordinates()
    offset = coordinates.separation(pointing)
    energy = ref_cube.energies()

    # TODO: properly transform FOV to sky coordinates
    # For now we assume the background is radially symmetric

    data = bkg.data.evaluate(detx=offset, dety='0 deg', energy=energy)
    data *= obstime * ref_cube.bin_size
    data[:, offset >= offset_max] = 0

    data = data.to('')

    return SkyCube(
        name='bkg',
        data=data,
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
