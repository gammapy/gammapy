# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from ..maps import WcsNDMap
from ..utils.coordinates import sky_to_fov

__all__ = ["make_map_background_irf"]


def make_map_background_irf(pointing, ontime, bkg, geom):
    """Compute background map from background IRFs.

    Parameters
    ----------
    pointing : `~gammapy.data.pointing.FixedPointingInfo`
        Fixed Pointing info
    ontime : `~astropy.units.Quantity`
        Observation ontime. i.e. not corrected for deadtime
        see https://gamma-astro-data-formats.readthedocs.io/en/stable/irfs/full_enclosure/bkg/index.html#notes)
    bkg : `~gammapy.irf.Background3D`
        Background rate model
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry

    Returns
    -------
    background : `~gammapy.maps.WcsNDMap`
        Background predicted counts sky cube in reco energy
    """
    # TODO:
    #  This implementation can be improved in two ways:
    #  1. Create equal time intervals between TSTART and TSTOP and sum up the
    #  background IRF for each interval. This is instead of multiplying by
    #  the total ontime. This then handles the rotation of the FoV.
    #  2. Use the pointing table (does not currently exist in CTA files) to
    #  obtain the RA DEC and time for each interval. This then considers that
    #  the pointing might change slightly over the observation duration

    # Get altaz coords for map
    map_coord = geom.to_image().get_coord()
    sky_coord = map_coord.skycoord
    altaz_coord = sky_coord.transform_to(pointing.altaz_frame)

    # Compute FOV coordinates of map relative to pointing
    fov_lon, fov_lat = sky_to_fov(
        altaz_coord.az,
        altaz_coord.alt,
        pointing.altaz.az,
        pointing.altaz.alt
    )

    energy_axis = geom.get_axis_by_name("energy")
    energies = energy_axis.edges * energy_axis.unit

    bkg_de = bkg.evaluate_integrate(
        fov_lon=fov_lon,
        fov_lat=fov_lat,
        energy_reco=energies[:, np.newaxis, np.newaxis],
    )

    d_omega = geom.solid_angle()
    data = (bkg_de * d_omega * ontime).to_value("")
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
