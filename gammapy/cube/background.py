# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from ..maps import WcsNDMap

__all__ = [
    'make_map_background_irf',
    'make_map_background_fov',
]


def make_map_background_irf(pointing, livetime, bkg, geom, n_integration_bins=1):
    """Compute background map from background IRFs.

    TODO: Call a method on bkg that returns integral over energy bin directly
    Related: https://github.com/gammapy/gammapy/pull/1342

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction
    livetime : `~astropy.units.Quantity`
        Observation livetime
    bkg : `~gammapy.irf.Background3D`
        Background rate model
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry
    n_integration_bins : int
            Number of bins used to integrate on each energy range

    Returns
    -------
    background : `~gammapy.maps.WcsNDMap`
        Background predicted counts sky cube in reco energy
    """
    # Compute the expected background
    # TODO: properly transform FOV to sky coordinates
    # For now we assume the background is radially symmetric

    energy_axis = geom.axes[0]
    # Compute offsets of all pixels
    map_coord = geom.get_coord()
    # Retrieve energies from map coordinates
    energy_reco = map_coord[energy_axis.name] * energy_axis.unit
    # TODO: go from SkyCoord to FOV coordinates. Here assume symmetric geometry for fov_lon, fov_lat
    # Compute offset at all the pixels and energy of the Map
    fov_lon = map_coord.skycoord.separation(pointing)
    fov_lat = Angle(np.zeros_like(fov_lon), fov_lon.unit)
    data_int = Quantity(np.zeros_like(fov_lat.value), "s^-1 sr^-1")
    for ie, (e_lo, e_hi) in enumerate(zip(energy_axis.edges[0:-1], energy_axis.edges[1:])):
        data_int[ie, :, :] = bkg.integrate_on_energy_range(
            fov_lon=fov_lon[0, :, :],
            fov_lat=fov_lat[0, :, :],
            energy_range=[e_lo * energy_axis.unit, e_hi * energy_axis.unit],
            n_integration_bins=n_integration_bins,
        )

    d_omega = geom.solid_angle()
    data = (data_int * d_omega * livetime).to('').value

    return WcsNDMap(geom, data=data)


def make_map_background_fov(acceptance_map, counts_map, exclusion_mask=None):
    """Build Normalized background map from a given acceptance map and counts map.

    This operation is normally performed on single observation maps.
    An exclusion map is used to avoid using regions with significant gamma-ray emission.
    All maps are assumed to follow the same WcsGeom.

    TODO: A model map could be used instead of an exclusion mask.

    Parameters
    ----------
    acceptance_map : `~gammapy.maps.WcsNDMap`
        Observation hadron acceptance map (i.e. predicted background map)
    counts_map : `~gammapy.maps.WcsNDMap`
        Observation counts map
    exclusion_mask : `~gammapy.maps.WcsNDMap`
        Exclusion mask

    Returns
    -------
    norm_bkg_map : `~gammapy.maps.WcsNDMap`
        Normalized background
    """
    if exclusion_mask is None:
        mask = np.ones_like(counts_map, dtype=bool)
    else:
        # We resize the mask
        mask = np.resize(np.squeeze(exclusion_mask.data), acceptance_map.data.shape)

    # We multiply the data with the mask to obtain normalization factors in each energy bin
    integ_acceptance = np.sum(acceptance_map.data * mask, axis=(1, 2))
    integ_counts = np.sum(counts_map.data * mask, axis=(1, 2))

    # TODO: Here we need to add a function rebin energy axis to have minimal statistics for the normalization

    # Normalize background
    norm_factor = integ_counts / integ_acceptance

    norm_bkg = norm_factor * acceptance_map.data.T

    return acceptance_map.copy(data=norm_bkg.T)
