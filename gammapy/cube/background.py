# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.coordinates import SkyOffsetFrame
from ..data import FixedPointingInfo
from ..maps import WcsNDMap
from ..utils.coordinates import sky_to_fov

__all__ = ["make_map_background_irf"]


def make_map_background_irf(pointing, ontime, bkg, geom, nodes_per_decade=30):
    """Compute background map from background IRFs.

    If a `FixedPointingInfo` is passed the correct FoV coordinates are properly computed.
    If a simple `SkyCoord` is passed, FoV coordinates are computed without proper rotation
    of the frame.

    Parameters
    ----------
    pointing : `~gammapy.data.pointing.FixedPointingInfo` or `~astropy.coordinates.SkyCoord`
        Fixed Pointing info or coordinates of the pointing
    ontime : `~astropy.units.Quantity`
        Observation ontime. i.e. not corrected for deadtime
        see https://gamma-astro-data-formats.readthedocs.io/en/stable/irfs/full_enclosure/bkg/index.html#notes)
    bkg : `~gammapy.irf.Background3D`
        Background rate model
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry
    nodes_per_decade : int
        Minimum number of nodes per decade of energy used in the integration

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

    if isinstance(pointing, FixedPointingInfo):
        altaz_coord = sky_coord.transform_to(pointing.altaz_frame)

        # Compute FOV coordinates of map relative to pointing
        fov_lon, fov_lat = sky_to_fov(
            altaz_coord.az,
            altaz_coord.alt,
            pointing.altaz.az,
            pointing.altaz.alt
        )
    else:
        # Create OffsetFrame
        frame = SkyOffsetFrame(origin=pointing)
        pseudo_fov_coord = sky_coord.transform_to(frame)
        fov_lon = pseudo_fov_coord.lon
        fov_lat = pseudo_fov_coord.lat

    energy_axis = geom.get_axis_by_name("energy")

    # make sure the integration uses at least `nodes_per_decade` nodes
    logemin = np.log10(energy_axis.edges[0])
    logemax = np.log10(energy_axis.edges[-1])
    n_decade = logemax - logemin
    n_nodes = energy_axis.nbin
    oversample = int( np.ceil( nodes_per_decade / (n_nodes / n_decade) ) )
    energies = np.logspace(logemin, logemax, n_nodes * oversample + 1) * energy_axis.unit

    # perform integration
    bkg_de = bkg.evaluate_integrate(
        fov_lon=fov_lon,
        fov_lat=fov_lat,
        energy_reco=energies[:, np.newaxis, np.newaxis],
    )

    # combine energy bins to obtain binning of original energy axis
    indices = np.arange(0, n_nodes * oversample, oversample)
    bkg_de = np.add.reduceat(bkg_de, indices, axis=0)

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
