# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.coordinates import Angle
from astropy.units import Quantity
from ..maps import WcsNDMap

__all__ = ["make_map_background_irf"]


def make_map_background_irf(pointing, livetime, bkg, geom, n_integration_bins=1):
    """Compute background map from background IRFs.

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
        Number of bins per energy bin in integration

    Returns
    -------
    background : `~gammapy.maps.WcsNDMap`
        Background predicted counts sky cube in reco energy
    """
    energy_axis = geom.axes[0]
    ebounds = energy_axis.edges * energy_axis.unit

    # Compute FOV coordinates; at the moment assume symmetric background model
    # TODO: implement FOV coordinates properly
    map_coord = geom.get_coord()
    fov_lon = map_coord.skycoord.separation(pointing)
    fov_lat = Angle(np.zeros_like(fov_lon), fov_lon.unit)

    if n_integration_bins == 0:
        energy_reco = map_coord[energy_axis.name] * energy_axis.unit
        data = bkg.evaluate(fov_lon=fov_lon, fov_lat=fov_lat, energy_reco=energy_reco)
        d_energy = np.diff(energy_axis.edges) * energy_axis.unit
        bkg_de = data * d_energy[:, np.newaxis, np.newaxis]
    else:
        bkg_de = Quantity(np.zeros_like(fov_lat.value), "s^-1 sr^-1")
        for idx in range(len(ebounds) - 1):
            energy_range = ebounds[idx], ebounds[idx + 1]
            bkg_de[idx, :, :] = bkg.integrate_on_energy_range(
                fov_lon=fov_lon[0, :, :],
                fov_lat=fov_lat[0, :, :],
                energy_range=energy_range,
                n_integration_bins=n_integration_bins,
            )

    d_omega = geom.solid_angle()
    data = (bkg_de * d_omega * livetime).to("").value

    return WcsNDMap(geom, data=data)


def _fov_background_norm(acceptance_map, counts_map, exclusion_mask=None):
    """Compute FOV background norm

    This operation is normally performed on single observation maps.
    An exclusion map is used to avoid using regions with significant gamma-ray emission.
    All maps are assumed to follow the same WcsGeom.

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
    norm_factor : array
        Background normalisation factor as function of energy (1D vector)
    """
    if exclusion_mask is None:
        mask = np.ones_like(counts_map, dtype=bool)
    else:
        # We resize the mask
        mask = np.resize(np.squeeze(exclusion_mask.data), acceptance_map.data.shape)

    # We multiply the data with the mask to obtain normalization factors in each energy bin
    integ_acceptance = np.sum(acceptance_map.data * mask, axis=(1, 2))
    integ_counts = np.sum(counts_map.data * mask, axis=(1, 2))

    norm_factor = integ_counts / integ_acceptance

    return norm_factor
