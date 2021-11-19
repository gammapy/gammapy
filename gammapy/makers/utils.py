# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from astropy.coordinates import Angle, SkyOffsetFrame
from astropy.table import Table
from gammapy.data import FixedPointingInfo
from gammapy.irf import EDispMap, PSFMap
from gammapy.maps import Map
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.stats import WStatCountsStatistic
from gammapy.utils.coordinates import sky_to_fov

__all__ = [
    "make_map_background_irf",
    "make_edisp_map",
    "make_edisp_kernel_map",
    "make_psf_map",
    "make_map_exposure_true_energy",
    "make_theta_squared_table",
]

log = logging.getLogger(__name__)


def make_map_exposure_true_energy(
    pointing, livetime, aeff, geom, use_region_center=True
):
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
    use_region_center: bool
        If geom is a RegionGeom, whether to just
        consider the values at the region center
        or the instead the average over the whole region

    Returns
    -------
    map : `~gammapy.maps.WcsNDMap`
        Exposure map
    """
    if not use_region_center:
        coords, weights = geom.get_wcs_coord_and_weights()
    else:
        coords, weights = geom.get_coord(sparse=True), None

    offset = coords.skycoord.separation(pointing)
    exposure = aeff.evaluate(offset=offset, energy_true=coords["energy_true"])

    data = (exposure * livetime).to("m2 s")
    meta = {"livetime": livetime, "is_pointlike": aeff.is_pointlike}

    if not use_region_center:
        data = np.average(data, axis=1, weights=weights)

    return Map.from_geom(geom=geom, data=data.value, unit=data.unit, meta=meta)


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
    energy_edges = map.geom.axes["energy_true"].edges
    weights = spectrum.integral(
        energy_min=energy_edges[:-1], energy_max=energy_edges[1:]
    )
    weights /= weights.sum()
    shape = np.ones(len(map.geom.data_shape))
    shape[0] = -1
    return map * weights.reshape(shape.astype(int))


def make_map_background_irf(
    pointing, ontime, bkg, geom, oversampling=None, use_region_center=True
):
    """Compute background map from background IRFs.

    Parameters
    ----------
    pointing : `~gammapy.data.FixedPointingInfo` or `~astropy.coordinates.SkyCoord`
        Observation pointing

        - If a ``FixedPointingInfo`` is passed, FOV coordinates are properly computed.
        - If a ``SkyCoord`` is passed, FOV frame rotation is not taken into account.
    ontime : `~astropy.units.Quantity`
        Observation ontime. i.e. not corrected for deadtime
        see https://gamma-astro-data-formats.readthedocs.io/en/stable/irfs/full_enclosure/bkg/index.html#notes)
    bkg : `~gammapy.irf.Background3D`
        Background rate model
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry
    oversampling: int
        Oversampling factor in energy, used for the background model evaluation.
    use_region_center: bool
        If geom is a RegionGeom, whether to just
        consider the values at the region center
        or the instead the sum over the whole region

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
    if oversampling is not None:
        geom = geom.upsample(factor=oversampling, axis_name="energy")

    coords = {"energy": geom.axes["energy"].edges.reshape((-1, 1, 1))}

    if not use_region_center:
        image_geom = geom.to_wcs_geom().to_image()
        region_coord, weights = geom.get_wcs_coord_and_weights()
        idx = image_geom.coord_to_idx(region_coord)
        sky_coord = region_coord.skycoord
        d_omega = image_geom.solid_angle().T[idx]
    else:
        image_geom = geom.to_image()
        map_coord = image_geom.get_coord()
        sky_coord = map_coord.skycoord
        d_omega = image_geom.solid_angle()

    if bkg.is_offset_dependent:
        coords["offset"] = sky_coord.separation(pointing)
    else:
        if isinstance(pointing, FixedPointingInfo):
            altaz_coord = sky_coord.transform_to(pointing.altaz_frame)

            # Compute FOV coordinates of map relative to pointing
            fov_lon, fov_lat = sky_to_fov(
                altaz_coord.az, altaz_coord.alt, pointing.altaz.az, pointing.altaz.alt
            )
        else:
            # Create OffsetFrame
            frame = SkyOffsetFrame(origin=pointing)
            pseudo_fov_coord = sky_coord.transform_to(frame)
            fov_lon = pseudo_fov_coord.lon
            fov_lat = pseudo_fov_coord.lat

        coords["fov_lon"] = fov_lon
        coords["fov_lat"] = fov_lat

    bkg_de = bkg.integrate_log_log(**coords, axis_name="energy")
    data = (bkg_de * d_omega * ontime).to_value("")

    if not use_region_center:
        data = np.sum(weights * data, axis=2)

    bkg_map = Map.from_geom(geom, data=data)

    if oversampling is not None:
        bkg_map = bkg_map.downsample(factor=oversampling, axis_name="energy")

    return bkg_map


def make_psf_map(psf, pointing, geom, exposure_map=None):
    """Make a psf map for a single observation

    Expected axes : rad and true energy in this specific order
    The name of the rad MapAxis is expected to be 'rad'

    Parameters
    ----------
    psf : `~gammapy.irf.PSF3D`
        the PSF IRF
    pointing : `~astropy.coordinates.SkyCoord`
        the pointing direction
    geom : `~gammapy.maps.Geom`
        the map geom to be used. It provides the target geometry.
        rad and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        the associated exposure map.
        default is None

    Returns
    -------
    psfmap : `~gammapy.irf.PSFMap`
        the resulting PSF map
    """
    coords = geom.get_coord(sparse=True)

    # Compute separations with pointing position
    offset = coords.skycoord.separation(pointing)

    # Compute PSF values
    data = psf.evaluate(
        energy_true=coords["energy_true"],
        offset=offset,
        rad=coords["rad"],
    )

    # Create Map and fill relevant entries
    psf_map = Map.from_geom(geom, data=data.value, unit=data.unit)
    psf_map.normalize(axis_name="rad")
    return PSFMap(psf_map, exposure_map)


def make_edisp_map(edisp, pointing, geom, exposure_map=None, use_region_center=True):
    """Make a edisp map for a single observation

    Expected axes : migra and true energy in this specific order
    The name of the migra MapAxis is expected to be 'migra'

    Parameters
    ----------
    edisp : `~gammapy.irf.EnergyDispersion2D`
        the 2D Energy Dispersion IRF
    pointing : `~astropy.coordinates.SkyCoord`
        the pointing direction
    geom : `~gammapy.maps.Geom`
        the map geom to be used. It provides the target geometry.
        migra and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        the associated exposure map.
        default is None
    use_region_center: Bool
        If geom is a RegionGeom, whether to just
        consider the values at the region center
        or the instead the average over the whole region

    Returns
    -------
    edispmap : `~gammapy.irf.EDispMap`
        the resulting EDisp map
    """
    # Compute separations with pointing position
    if not use_region_center:
        coords, weights = geom.get_wcs_coord_and_weights()
    else:
        coords, weights = geom.get_coord(sparse=True), None

    offset = coords.skycoord.separation(pointing)

    # Compute EDisp values
    data = edisp.evaluate(
        offset=offset,
        energy_true=coords["energy_true"],
        migra=coords["migra"],
    )

    if not use_region_center:
        data = np.average(data, axis=2, weights=weights)

    # Create Map and fill relevant entries
    edisp_map = Map.from_geom(geom, data=data.to_value(""), unit="")
    edisp_map.normalize(axis_name="migra")
    return EDispMap(edisp_map, exposure_map)


def make_edisp_kernel_map(
    edisp, pointing, geom, exposure_map=None, use_region_center=True
):
    """Make a edisp kernel map for a single observation

    Expected axes : (reco) energy and true energy in this specific order
    The name of the reco energy MapAxis is expected to be 'energy'.
    The name of the true energy MapAxis is expected to be 'energy_true'.

    Parameters
    ----------
    edisp : `~gammapy.irf.EnergyDispersion2D`
        the 2D Energy Dispersion IRF
    pointing : `~astropy.coordinates.SkyCoord`
        the pointing direction
    geom : `~gammapy.maps.Geom`
        the map geom to be used. It provides the target geometry.
        energy and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        the associated exposure map.
        default is None
    use_region_center: Bool
        If geom is a RegionGeom, whether to just
        consider the values at the region center
        or the instead the average over the whole region

    Returns
    -------
    edispmap : `~gammapy.irf.EDispKernelMap`
        the resulting EDispKernel map
    """
    # Use EnergyDispersion2D migra axis.
    migra_axis = edisp.axes["migra"]

    # Create temporary EDispMap Geom
    new_geom = geom.to_image().to_cube([migra_axis, geom.axes["energy_true"]])

    edisp_map = make_edisp_map(
        edisp, pointing, new_geom, exposure_map, use_region_center
    )

    return edisp_map.to_edisp_kernel_map(geom.axes["energy"])


def make_theta_squared_table(
    observations, theta_squared_axis, position, position_off=None
):
    """Make theta squared distribution in the same FoV for a list of `Observation`
    objects.

    The ON theta2 profile is computed from a given distribution, on_position.
    By default, the OFF theta2 profile is extracted from a mirror position
    radially symmetric in the FOV to pos_on.

    The ON and OFF regions are assumed to be of the same size, so the normalisation
    factor between both region alpha = 1.

    Parameters
    ----------
    observations: `~gammapy.data.Observations`
        List of observations
    theta_squared_axis : `~gammapy.maps.geom.MapAxis`
        Axis of edges of the theta2 bin used to compute the distribution
    position : `~astropy.coordinates.SkyCoord`
        Position from which the on theta^2 distribution is computed
    position_off : `astropy.coordinates.SkyCoord`
        Position from which the OFF theta^2 distribution is computed.
        Default: reflected position w.r.t. to the pointing position

    Returns
    -------
    table : `~astropy.table.Table`
        Table containing the on counts, the off counts, acceptance, off acceptance and alpha
        for each theta squared bin.
    """
    if not theta_squared_axis.edges.unit.is_equivalent("deg2"):
        raise ValueError("The theta2 axis should be equivalent to deg2")

    table = Table()

    table["theta2_min"] = theta_squared_axis.edges[:-1]
    table["theta2_max"] = theta_squared_axis.edges[1:]
    table["counts"] = 0
    table["counts_off"] = 0
    table["acceptance"] = 0.0
    table["acceptance_off"] = 0.0

    alpha_tot = np.zeros(len(table))
    livetime_tot = 0

    create_off = position_off is None
    for observation in observations:
        separation = position.separation(observation.events.radec)
        counts, _ = np.histogram(separation ** 2, theta_squared_axis.edges)
        table["counts"] += counts

        if create_off:
            # Estimate the position of the mirror position
            pos_angle = observation.pointing_radec.position_angle(position)
            sep_angle = observation.pointing_radec.separation(position)
            position_off = observation.pointing_radec.directional_offset_by(
                pos_angle + Angle(np.pi, "rad"), sep_angle
            )

        # Angular distance of the events from the mirror position
        separation_off = position_off.separation(observation.events.radec)

        # Extract the ON and OFF theta2 distribution from the two positions.
        counts_off, _ = np.histogram(separation_off ** 2, theta_squared_axis.edges)
        table["counts_off"] += counts_off

        # Normalisation between ON and OFF is one
        acceptance = np.ones(theta_squared_axis.nbin)
        acceptance_off = np.ones(theta_squared_axis.nbin)

        table["acceptance"] += acceptance
        table["acceptance_off"] += acceptance_off
        alpha = acceptance / acceptance_off
        alpha_tot += alpha * observation.observation_live_time_duration.to_value("s")
        livetime_tot += observation.observation_live_time_duration.to_value("s")

    alpha_tot /= livetime_tot
    table["alpha"] = alpha_tot

    stat = WStatCountsStatistic(table["counts"], table["counts_off"], table["alpha"])
    table["excess"] = stat.n_sig
    table["sqrt_ts"] = stat.sqrt_ts
    table["excess_errn"] = stat.compute_errn()
    table["excess_errp"] = stat.compute_errp()

    table.meta["ON_RA"] = position.icrs.ra
    table.meta["ON_DEC"] = position.icrs.dec
    return table
