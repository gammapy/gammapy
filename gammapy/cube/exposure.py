# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from ..irf import EnergyDispersion
from ..spectrum.models import PowerLaw
from ..maps import WcsNDMap, MapAxis, Map

__all__ = [
    'make_map_exposure_true_energy',
    'weighted_exposure_image',
]


def make_map_exposure_true_energy(pointing, livetime, aeff, geom):
    """Compute exposure map.

    This map has a true energy axis, the exposure is not combined
    with energy dispersion.

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    livetime : `~astropy.units.Quantity`
        Livetime
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area
    geom : `~gammapy.maps.WcsGeom`
        Map geometry (must have an energy axis)

    Returns
    -------
    map : `~gammapy.maps.WcsNDMap`
        Exposure map
    """
    offset = geom.separation(pointing)
    energy = geom.axes[0].center * geom.axes[0].unit

    exposure = aeff.data.evaluate(offset=offset, energy=energy)
    # TODO: Improve IRF evaluate to preserve energy axis if length 1
    # For now, we handle that case via this hack:
    if len(exposure.shape) < 3:
        exposure = np.expand_dims(exposure.value, 0) * exposure.unit

    exposure = (exposure * livetime).to('m2 s')

    return WcsNDMap(geom, exposure.value, unit=exposure.unit)

def weighted_exposure_image(exposure_map, spectrum=None):
    """Create an exposure map in reco energy from an exposure map in true energy.

    Exposure in true energy is weighted with an input spectrum and redistributed in
    reco energy with the input energy dispersion.

    Parameters
    ----------
    exposure_map : `~gammapy.maps.Map`
        Input exposure map in true energy. unit should have dimension of m2s
    spectrum : `~gammapy.spectrum.models.SpectralModel`, default is None
        Spectral model to use to weight exposure in true energy
        If None is passed, a power law of photon index -2 is assumed.

    Returns
    -------
    exposure_image : `~gammapy.maps.Map`
        Resulting weighted image. The unit is the same as the input map.
    """
    expo_map = exposure_map.copy()
    energy_axis = expo_map.geom.get_axis_by_name("energy")
    energy_center = energy_axis.center * energy_axis.unit
    energy_edges = energy_axis.edges * energy_axis.unit
    binsize = np.diff(energy_edges)

    if spectrum is None:
        spectrum = PowerLaw(index=2.0)
    weights = spectrum(energy_center)*binsize
    weights /= weights.sum()

    for img, idx in expo_map.iter_by_image():
        img *= weights[idx].value

    return expo_map.sum_over_axes()
