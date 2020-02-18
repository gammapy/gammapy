# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from gammapy.maps import WcsNDMap
from gammapy.modeling.models import PowerLawSpectralModel

__all__ = ["make_map_exposure_true_energy"]


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
    energy = geom.get_axis_by_name("energy_true").center

    exposure = aeff.data.evaluate(
        offset=offset, energy_true=energy[:, np.newaxis, np.newaxis]
    )
    # TODO: Improve IRF evaluate to preserve energy axis if length 1
    # For now, we handle that case via this hack:
    if len(exposure.shape) < 3:
        exposure = np.expand_dims(exposure.value, 0) * exposure.unit

    exposure = (exposure * livetime).to("m2 s")

    return WcsNDMap(geom, exposure.value.reshape(geom.data_shape), unit=exposure.unit)


def _map_spectrum_weight(map, spectrum=None):
    """Weight a map with a spectrum.

    This requires map to have an "energy" axis.
    The weights are normalised so that they sum to 1.
    The mean and unit of the output image is the same as of the input cube.

    At the moment this is used to get a weighted exposure image.

    Parameters
    ----------
    map : `~gammapy.maps.Map`
        Input map with an "energy" axis.
    spectrum : `~gammapy.modeling.models.SpectralModel`
        Spectral model to compute the weights.
        Default is power-law with spectral index of 2.

    Returns
    -------
    map_weighted : `~gammapy.maps.Map`
        Weighted image
    """
    if spectrum is None:
        spectrum = PowerLawSpectralModel(index=2.0)

    # Compute weights vector
    energy_edges = map.geom.get_axis_by_name("energy").edges
    weights = spectrum.integral(
        emin=energy_edges[:-1], emax=energy_edges[1:], intervals=True
    )
    weights /= weights.sum()
    shape = np.ones(len(map.geom.data_shape))
    shape[0] = -1
    return map * weights.reshape(shape.astype(int))
