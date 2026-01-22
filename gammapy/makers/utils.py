# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
from astropy.table import Table
from astropy.time import Time
from gammapy.data import FixedPointingInfo, PointingMode
from gammapy.irf import EDispMap, FoVAlignment, PSFMap
from gammapy.maps import Map, RegionNDMap, MapAxis
from gammapy.maps.utils import broadcast_axis_values_to_geom
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.stats import WStatCountsStatistic
from gammapy.utils.coordinates import FoVICRSFrame, FoVAltAzFrame
from gammapy.utils.regions import compound_region_to_regions
from gammapy.maps import RegionGeom

__all__ = [
    "make_mask_events",
    "make_events_off_mask",
    "make_counts_off_rad_max",
    "make_counts_rad_max",
    "make_edisp_kernel_map",
    "make_edisp_map",
    "make_map_background_irf",
    "make_map_exposure_true_energy",
    "make_psf_map",
    "make_theta_squared_table",
    "make_effective_livetime_map",
    "make_observation_time_map",
]

log = logging.getLogger(__name__)

MINIMUM_TIME_STEP = 1 * u.s  # Minimum time step used to handle FoV rotations
EARTH_ANGULAR_VELOCITY = 360 * u.deg / u.day


def _compute_rotation_time_steps(
    time_start, time_stop, fov_rotation, pointing_altaz, location
):
    """
    Compute the time intervals between start and stop times, such that the FoV associated to a fixed RaDec position rotates
    by 'fov_rotation' in AltAz frame during each time step.
    It assumes that the rotation rate at the provided pointing is a good estimate of the rotation rate over the full
    output duration (first order approximation).


    Parameters
    ----------
    time_start : `~astropy.time.Time`
        Start time
    time_stop : `~astropy.time.Time`
        Stop time
    fov_rotation : `~astropy.units.Quantity`
        Rotation angle.
    pointing_altaz : `~astropy.coordinates.SkyCoord`
        Pointing direction.
    location : `astropy.coordinates.EarthLocation`
        Observatory location
    Returns
    -------
    times : `~astropy.time.Time`
        Times associated with the requested rotation.
    """

    def _time_step(rotation, pnt_altaz):
        denom = (
            EARTH_ANGULAR_VELOCITY
            * np.cos(pnt_altaz.location.lat.rad)
            * np.abs(np.cos(pnt_altaz.az.rad))
        )
        return rotation * np.cos(pnt_altaz.alt.rad) / denom

    time = time_start
    times = [time]
    while time < time_stop:
        time_step = _time_step(fov_rotation, pointing_altaz.get_altaz(time, location))
        time_step = max(time_step, MINIMUM_TIME_STEP)
        time = min(time + time_step, time_stop)
        times.append(time)
    return Time(times)


def make_map_exposure_true_energy(
    pointing, livetime, aeff, geom, use_region_center=True
):
    """Compute exposure map.

    This map has a true energy axis, the exposure is not combined
    with energy dispersion.

    Parameters
    ----------
    pointing : `~astropy.coordinates.SkyCoord`
        Pointing direction.
    livetime : `~astropy.units.Quantity`
        Livetime.
    aeff : `~gammapy.irf.EffectiveAreaTable2D`
        Effective area.
    geom : `~gammapy.maps.WcsGeom`
        Map geometry (must have an energy axis).
    use_region_center : bool, optional
        For geom as a `~gammapy.maps.RegionGeom`. If True, consider the values at the region center.
        If False, average over the whole region.
        Default is True.

    Returns
    -------
    map : `~gammapy.maps.WcsNDMap`
        Exposure map.
    """
    if isinstance(pointing, FixedPointingInfo):
        origin = pointing.get_icrs(pointing.obstime)
    else:
        origin = pointing

    fov_frame = FoVICRSFrame(origin=origin)

    exposure = project_irf_on_geom(geom, aeff, fov_frame, use_region_center)

    exposure *= u.Quantity(livetime)
    exposure = exposure.to_unit("m2 s")
    exposure.meta.update({"livetime": livetime, "is_pointlike": aeff.is_pointlike})

    return exposure


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
    spectrum : `~gammapy.modeling.models.SpectralModel`, optional
        Spectral model to compute the weights.
        Default is None, which is a power-law with spectral index of 2.

    Returns
    -------
    map_weighted : `~gammapy.maps.Map`
        Weighted image.
    """
    if spectrum is None:
        spectrum = PowerLawSpectralModel(index=2.0)

    # Compute weights vector
    for name in map.geom.axes.names:
        if "energy" in name:
            energy_name = name
    energy_edges = map.geom.axes[energy_name].edges
    weights = spectrum.integral(
        energy_min=energy_edges[:-1], energy_max=energy_edges[1:]
    )
    weights /= weights.sum()
    shape = np.ones(len(map.geom.data_shape))
    shape[0] = -1
    return map * weights.reshape(shape.astype(int))


def make_map_background_irf(
    pointing,
    ontime,
    bkg,
    geom,
    time_start,
    fov_rotation_step=1.0 * u.deg,
    oversampling=None,
    use_region_center=True,
    location=None,
):
    """Compute background map from background IRFs.

    Parameters
    ----------
    pointing : `~gammapy.data.FixedPointingInfo` or `~astropy.coordinates.SkyCoord`
        Observation pointing.

        - If a `~gammapy.data.FixedPointingInfo` is passed, FOV coordinates
          are properly computed.
        - If a `~astropy.coordinates.SkyCoord` is passed, FOV frame rotation
          is not taken into account.

    ontime : `~astropy.units.Quantity`
        Observation ontime. i.e. not corrected for deadtime
        see https://gamma-astro-data-formats.readthedocs.io/en/latest/irfs/full_enclosure/bkg/index.html#notes)  # noqa: E501
    bkg : `~gammapy.irf.Background3D`
        Background rate model.
    geom : `~gammapy.maps.WcsGeom`
        Reference geometry.
    time_start : `~astropy.time.Time`
        Observation start time.
    fov_rotation_step : `~astropy.units.Quantity`
        Maximum rotation error to create sub-time bins if the irf is 3D and in AltAz.
        Default is 1.0 deg.
    oversampling : int, optional
        Oversampling factor in energy, used for the background model evaluation.
        Default is None.
    use_region_center : bool, optional
        For geom as a `~gammapy.maps.RegionGeom`. If True, consider the values at the region center.
        If False, average over the whole region.
        Default is True.
    location : `astropy.coordinates.EarthLocation`, optional
        Observatory location
    Returns
    -------
    background : `~gammapy.maps.WcsNDMap`
        Background predicted counts sky cube in reconstructed energy.
    """
    # TODO:
    #  This implementation can be improved in one way:
    #  Use the pointing table (does not currently exist in CTA files) to
    #  obtain the RA DEC and time for each interval. This then considers that
    #  the pointing might change slightly over the observation duration
    # Compute intermediate time ranges if needed
    times = Time([time_start, time_start + ontime])

    if not bkg.has_offset_axis and bkg.fov_alignment == FoVAlignment.ALTAZ:
        if not isinstance(pointing, FixedPointingInfo):
            raise TypeError(
                "make_map_background_irf requires FixedPointingInfo if "
                "BackgroundIRF.fov_alignement is ALTAZ",
            )
        times = _compute_rotation_time_steps(
            time_start, time_start + ontime, fov_rotation_step, pointing, location
        )
        origin = pointing.get_altaz(times, location)
        fov_frame = FoVAltAzFrame(
            origin=origin, location=origin.location, obstime=times
        )
    else:
        if isinstance(pointing, FixedPointingInfo):
            if pointing.mode == PointingMode.POINTING:
                origin = pointing.fixed_icrs
            else:
                raise NotImplementedError(
                    "Drift pointing mode is not supported for background calculation."
                )
        else:
            origin = pointing

        fov_frame = FoVICRSFrame(origin=origin)

    if oversampling is not None:
        geom = geom.upsample(factor=oversampling, axis_name="energy")

    bkg_map = integrate_project_irf_on_geom(geom, bkg, fov_frame, use_region_center)
    bkg_map *= ontime

    if oversampling is not None:
        bkg_map = bkg_map.downsample(factor=oversampling, axis_name="energy")

    return bkg_map.to_unit("")


def make_psf_map(psf, pointing, geom, exposure_map=None):
    """Make a PSF map for a single observation.

    Expected axes : rad and true energy in this specific order.
    The name of the rad MapAxis is expected to be 'rad'.

    Parameters
    ----------
    psf : `~gammapy.irf.PSF3D`
        The PSF IRF.
    pointing : `~astropy.coordinates.SkyCoord`
        The pointing direction.
    geom : `~gammapy.maps.Geom`
        The map geometry to be used. It provides the target geometry.
        rad and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        The associated exposure map.
        Default is None.

    Returns
    -------
    psfmap : `~gammapy.irf.PSFMap`
        The resulting PSF map.
    """
    if isinstance(pointing, FixedPointingInfo):
        origin = pointing.get_icrs(pointing.obstime)
    else:
        origin = pointing

    fov_frame = FoVICRSFrame(origin=origin)

    psf_map = project_irf_on_geom(geom, psf, fov_frame)
    psf_map.normalize(axis_name="rad")
    return PSFMap(psf_map, exposure_map)


def make_edisp_map(edisp, pointing, geom, exposure_map=None, use_region_center=True):
    """Make an edisp map for a single observation.

    Expected axes : migra and true energy in this specific order.
    The name of the migra MapAxis is expected to be 'migra'.

    Parameters
    ----------
    edisp : `~gammapy.irf.EnergyDispersion2D`
        The 2D energy dispersion IRF.
    pointing : `~astropy.coordinates.SkyCoord`
        The pointing direction.
    geom : `~gammapy.maps.Geom`
        The map geometry to be used. It provides the target geometry.
        migra and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        The associated exposure map.
        Default is None.
    use_region_center : bool, optional
        For geom as a `~gammapy.maps.RegionGeom`. If True, consider the values at the region center.
        If False, average over the whole region.
        Default is True.

    Returns
    -------
    edispmap : `~gammapy.irf.EDispMap`
        The resulting energy dispersion map.
    """
    if isinstance(pointing, FixedPointingInfo):
        origin = pointing.get_icrs(pointing.obstime)
    else:
        origin = pointing

    fov_frame = FoVICRSFrame(origin=origin)

    edisp_map = project_irf_on_geom(geom, edisp, fov_frame).to_unit("")
    edisp_map.normalize(axis_name="migra")
    return EDispMap(edisp_map, exposure_map)


def make_edisp_kernel_map(
    edisp, pointing, geom, exposure_map=None, use_region_center=True
):
    """Make an edisp kernel map for a single observation.

    Expected axes : (reco) energy and true energy in this specific order.
    The name of the reco energy MapAxis is expected to be 'energy'.
    The name of the true energy MapAxis is expected to be 'energy_true'.

    Parameters
    ----------
    edisp : `~gammapy.irf.EnergyDispersion2D`
        The 2D energy dispersion IRF.
    pointing : `~astropy.coordinates.SkyCoord`
        The pointing direction.
    geom : `~gammapy.maps.Geom`
        The map geometry to be used. It provides the target geometry.
        energy and true energy axes should be given in this specific order.
    exposure_map : `~gammapy.maps.Map`, optional
        The associated exposure map.
        Default is None.
    use_region_center : bool, optional
        For geom as a `~gammapy.maps.RegionGeom`. If True, consider the values at the region center.
        If False, average over the whole region.
        Default is True.

    Returns
    -------
    edispmap : `~gammapy.irf.EDispKernelMap`
        the resulting EDispKernel map
    """
    # Use EnergyDispersion2D migra axis.
    migra_axis = edisp.axes["migra"]

    # Create temporary EDispMap Geom
    new_geom = geom.to_image().to_cube([migra_axis, geom.axes["energy_true"]])
    new_geom = RegionGeom.create(
        region=geom.region,
        axes=new_geom.axes,
    )
    # CREATE REGIONGEOM WITH NEW AXIS
    edisp_map = make_edisp_map(
        edisp, pointing, new_geom, exposure_map, use_region_center
    )
    if geom.is_unbinned:
        energy_axes = geom.axes["energy"]
    else:
        energy_axes = geom.axes["energy"]
    return edisp_map.to_edisp_kernel_map(energy_axes)


def make_theta_squared_table(
    observations, theta_squared_axis, position, position_off=None, energy_edges=None
):
    """Make theta squared distribution in the same FoV for a list of `~gammapy.data.Observation` objects.

    The ON theta2 profile is computed from a given distribution, on_position.
    By default, the OFF theta2 profile is extracted from a mirror position
    radially symmetric in the FOV to pos_on.

    The ON and OFF regions are assumed to be of the same size, so the normalisation
    factor between both region alpha = 1.

    Parameters
    ----------
    observations: `~gammapy.data.Observations`
        List of observations.
    theta_squared_axis : `~gammapy.maps.MapAxis`
        Axis of edges of the theta2 bin used to compute the distribution.
    position : `~astropy.coordinates.SkyCoord`
        Position from which the on theta^2 distribution is computed.
    position_off : `astropy.coordinates.SkyCoord`
        Position from which the OFF theta^2 distribution is computed.
        Default is reflected position w.r.t. to the pointing position.
    energy_edges : list of `~astropy.units.Quantity`, optional
        Edges of the energy bin where the theta squared distribution
        is evaluated. For now, only one interval is accepted.
        Default is None.

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

    if energy_edges is not None:
        if len(energy_edges) == 2:
            table.meta["Energy_filter"] = energy_edges
        else:
            raise ValueError(
                f"Only  supports one energy interval but {len(energy_edges) - 1} passed."
            )

    for observation in observations:
        events = observation.events
        if energy_edges is not None:
            events = events.select_energy(energy_range=energy_edges)

        event_position = events.radec
        pointing = observation.get_pointing_icrs(observation.tmid)

        separation = position.separation(event_position)
        counts, _ = np.histogram(separation**2, theta_squared_axis.edges)
        table["counts"] += counts

        if create_off:
            # Estimate the position of the mirror position
            pos_angle = pointing.position_angle(position)
            sep_angle = pointing.separation(position)
            position_off = pointing.directional_offset_by(
                pos_angle + Angle(np.pi, "rad"), sep_angle
            )

        # Angular distance of the events from the mirror position
        separation_off = position_off.separation(event_position)

        # Extract the ON and OFF theta2 distribution from the two positions.
        counts_off, _ = np.histogram(separation_off**2, theta_squared_axis.edges)
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


def make_counts_rad_max(geom, rad_max, events):
    """Extract the counts using for the ON region size the values in the `RAD_MAX_2D` table.

    Parameters
    ----------
    geom : `~gammapy.maps.RegionGeom`
        Reference map geometry.
    rad_max : `~gammapy.irf.RadMax2D`
        Rhe RAD_MAX_2D table IRF.
    events : `~gammapy.data.EventList`
        Event list to be used to compute the ON counts.

    Returns
    -------
    counts : `~gammapy.maps.RegionNDMap`
        Counts vs estimated energy extracted from the ON region.
    """
    selected_events = events.select_rad_max(
        rad_max=rad_max, position=geom.region.center
    )

    counts = Map.from_geom(geom=geom)
    counts.fill_events(selected_events)
    return counts


def make_mask_events(geom, rad_max, events):
    """Extract the mask of events using for the ON region size the values in the `RAD_MAX_2D` table.
    Parameters
    ----------
    geom : `~gammapy.maps.RegionGeom`
        Reference map geometry.
    rad_max : `~gammapy.irf.RadMax2D`
        The RAD_MAX_2D table IRF.
    events : `~gammapy.data.EventList`
        Event list.
    Returns
    -------
    events_on_mask : `~numpy.ndarray`
        Mask of events in the ON region.
    """
    events_on_mask = events._mask_rad_max(rad_max=rad_max, position=geom.region.center)
    return events_on_mask


def make_events_off_mask(geom_off, rad_max, events):
    """Extract the mask of events in a list of point regions and given rad max.
    This method does **not** check for overlap of the regions defined by rad_max.
    Parameters
    ----------
    geom_off : `~gammapy.maps.RegionGeom`
        Reference map geometry for the on region.
    rad_max : `~gammapy.irf.RadMax2D`
        The RAD_MAX_2D table IRF.
    events : `~gammapy.data.EventList`
        Event list.
    Returns
    -------
    events_off_mask :`~numpy.ndarray`
        Mask of events in the different OFF regions.
    """
    events_off_mask = []
    for off_region in compound_region_to_regions(geom_off.region):
        events_off_mask.append(
            events._mask_rad_max(rad_max=rad_max, position=off_region.center)
        )
    return np.logical_or.reduce(events_off_mask)


def make_counts_off_rad_max(geom_off, rad_max, events):
    """Extract the OFF counts from a list of point regions and given rad max.

    This method does **not** check for overlap of the regions defined by rad_max.

    Parameters
    ----------
    geom_off : `~gammapy.maps.RegionGeom`
        Reference map geometry for the on region.
    rad_max : `~gammapy.irf.RadMax2D`
        The RAD_MAX_2D table IRF.
    events : `~gammapy.data.EventList`
        Event list to be used to compute the OFF counts.

    Returns
    -------
    counts_off : `~gammapy.maps.RegionNDMap`
        OFF Counts vs estimated energy extracted from the ON region.
    """
    if not geom_off.is_all_point_sky_regions:
        raise ValueError(
            f"Only supports PointSkyRegions, got {geom_off.region} instead"
        )

    counts_off = RegionNDMap.from_geom(geom=geom_off)

    for off_region in compound_region_to_regions(geom_off.region):
        selected_events = events.select_rad_max(
            rad_max=rad_max, position=off_region.center
        )
        counts_off.fill_events(selected_events)

    return counts_off


def make_observation_time_map(observations, geom, offset_max=None):
    """Compute the total observation time on the target geometry for a list of observations.

    Parameters
    ----------
    observations : `~gammapy.data.Observations`
            Observations container containing list of observations.
    geom : `~gammapy.maps.Geom`
            Reference geometry.
    offset_max : `~astropy.units.quantity.Quantity`, optional
        The maximum offset FoV. Default is None.
        If None, it will be taken from the IRFs.

    Returns
    -------
    exposure : `~gammapy.maps.Map`
        Effective livetime.
    """
    geom = geom.to_image()
    stacked = Map.from_geom(geom, unit=u.h)
    for obs in observations:
        if offset_max is None:
            offset_max = guess_instrument_fov(obs)
        coords = geom.get_coord(sparse=True)
        offset = coords.skycoord.separation(obs.get_pointing_icrs(obs.tmid))
        mask = offset < offset_max
        c1 = coords.apply_mask(mask)
        weights = np.ones(c1.shape) * obs.observation_live_time_duration
        stacked.fill_by_coord(coords=c1, weights=weights)
    return stacked


def make_effective_livetime_map(observations, geom, offset_max=None):
    """Compute the acceptance corrected livetime map for a list of observations.

    Parameters
    ----------
    observations : `~gammapy.data.Observations`
        Observations container containing list of observations.
    geom : `~gammapy.maps.Geom`
        Reference geometry.
    offset_max : `~astropy.units.quantity.Quantity`, optional
        The maximum offset FoV. Default is None.

    Returns
    -------
     exposure : `~gammapy.maps.Map`
        Effective livetime.
    """
    livetime = Map.from_geom(geom, unit=u.hr)
    for obs in observations:
        if offset_max is None:
            offset_max = guess_instrument_fov(obs)
        geom_obs = geom.cutout(
            position=obs.get_pointing_icrs(obs.tmid), width=2.0 * offset_max
        )
        coords = geom_obs.get_coord()
        offset = coords.skycoord.separation(obs.get_pointing_icrs(obs.tmid))
        mask = offset < offset_max

        exposure = make_map_exposure_true_energy(
            pointing=geom.center_skydir,
            livetime=obs.observation_live_time_duration,
            aeff=obs.aeff,
            geom=geom_obs,
            use_region_center=True,
        )

        on_axis = obs.aeff.evaluate(
            offset=0.0 * u.deg, energy_true=geom.axes["energy_true"].center
        )
        on_axis = on_axis.reshape((on_axis.shape[0], 1, 1))
        lv_obs = exposure * mask / on_axis
        livetime.stack(lv_obs)
    return livetime


def guess_instrument_fov(obs):
    """
    Guess the camera field of view for the given observation
    from the IRFs. This simply takes the maximum offset of the
    effective area IRF.
    TODO: This logic will break for more complex IRF models.
    A better option would be to compute the offset at which
    the effective area is above 10% of the maximum.

    Parameters
    ----------
    obs : `~gammapy.data.Observation`
        Observation container.

    Returns
    -------
    offset_max : `~astropy.units.quantity.Quantity`
        The maximum offset of the effective area IRF.
    """
    if "aeff" not in obs.available_irfs:
        raise ValueError("No Effective area IRF to infer the FoV from")
    if obs.aeff.is_pointlike:
        raise ValueError("Cannot guess FoV from pointlike IRFs")
    if "offset" not in obs.aeff.axes.names:
        raise ValueError("Offset axis not present!")
    return obs.aeff.axes["offset"].center[-1]


def _get_fov_coord(
    skycoord, fov_frame, use_offset=True, reverse_lon=False, time_resolution=1000 * u.s
):
    """Return coord dict in fov_coord."""
    coords = {}

    if use_offset:
        coords["offset"] = skycoord.separation(fov_frame.origin)
    else:
        sign = -1.0 if reverse_lon else 1.0

        with erfa_astrom.set(ErfaAstromInterpolator(time_resolution)):
            fov_coords = skycoord.transform_to(fov_frame)

        if len(fov_frame.shape) == 1:
            coords["fov_lon"] = np.moveaxis(fov_coords.fov_lon, -1, 0)
            coords["fov_lat"] = np.moveaxis(fov_coords.fov_lat, -1, 0)
        else:
            coords["fov_lon"] = sign * fov_coords.fov_lon
            coords["fov_lat"] = fov_coords.fov_lat

    return coords


def project_irf_on_geom(geom, irf, fov_frame, use_region_center=True):
    """Evaluate and project an IRF on a given `~gammapy.maps.Geom` object according to a given FoV Frame.

    When ``geom`` is a `~gammapy.maps.RegionGeom`, the IRF is evaluated at the region center when
    ``user_region_center is True``. Otherwise, the IRF is evaluated and averaged over the whole region.

    Parameters
    ----------
    geom : `~gammapy.maps.Geom`
        Geometry to project on. It must follow the required axes of the input IRF.
    irf : `~gammapy.itf.IRF`
        IRF to reproject.
    fov_frame : `~gammapy.utils.coordinate.FoVICRSFrame` or `~gammapy.utils.coordinate.FoVAltAzFrame`
        FoV frame to convert geometry to FoV coordinates.
    use_region_center : bool, optional
        For geom as a `~gammapy.maps.RegionGeom`. If True, consider the values at the region center.
        If False, average over the whole region.
        Default is True.

    Returns
    -------
    map : `~gammapy.maps.Map`
        Map containing the projected IRF.
    """
    if geom.is_unbinned:
        # IF WE WANT TO CONSIDER THE RECO EVENTS OFFSET
        # skycoord = SkyCoord(
        #    ra=geom.axes["events"]["ra"].center,
        #    dec=geom.axes["events"]["dec"].center,
        # )
        # coords = _get_fov_coord(skycoord, fov_frame, irf.has_offset_axis)
        # coords["offset"] = (
        #    np.ones((len(irf.axes["energy_true"].center), 1))
        #    @ np.array([coords["offset"]])
        # ).T.tolist() * coords["offset"].unit

        if not use_region_center:
            image_geom = geom.to_wcs_geom().to_image()
            region_coord, weights = geom.get_wcs_coord_and_weights()
            skycoord = region_coord.skycoord
        else:
            image_geom = geom.to_image()
            skycoord = image_geom.get_coord().skycoord

        coords = _get_fov_coord(skycoord, fov_frame, irf.has_offset_axis)
        non_spatial_axes = set(irf.required_arguments) - set(
            ["offset", "fov_lon", "fov_lat"]
        )

        for axis_name in non_spatial_axes:
            if axis_name == "migra":
                coords[axis_name] = (
                    np.array([geom.axes["energy"].center]).T
                    @ np.array([1 / irf.axes["energy_true"].center])
                ).tolist()
            elif axis_name == "energy_true":
                coords[axis_name] = (
                    np.ones((len(geom.axes["energy"].center), 1))
                    @ np.array([irf.axes["energy_true"].center])
                ).tolist() * irf.axes["energy_true"].unit

        data = irf.evaluate(**coords)

        return Map.from_geom(geom=geom, data=data.value.T, unit=data.unit)

    if not use_region_center:
        image_geom = geom.to_wcs_geom().to_image()
        region_coord, weights = geom.get_wcs_coord_and_weights()
        skycoord = region_coord.skycoord
    else:
        image_geom = geom.to_image()
        skycoord = image_geom.get_coord().skycoord

    coords = _get_fov_coord(skycoord, fov_frame, irf.has_offset_axis)

    non_spatial_axes = set(irf.required_arguments) - set(
        ["offset", "fov_lon", "fov_lat"]
    )

    for axis_name in non_spatial_axes:
        coords[axis_name] = broadcast_axis_values_to_geom(geom, axis_name)

    data = irf.evaluate(**coords)
    if not use_region_center:
        data = np.average(data, axis=-1, weights=weights, keepdims=True)

    return Map.from_geom(geom=geom, data=data.value, unit=data.unit)


def integrate_project_irf_on_geom(geom, irf, fov_frame, use_region_center=True):
    """Integrate and project an IRF on a given `~gammapy.maps.Geom` object according to a given FoV Frame.

    The IRF is integrated in energy and multiplied by the solid angle.

    When ``geom`` is a `~gammapy.maps.RegionGeom`, the IRF is evaluated at the region center when
    ``user_region_center is True``. Otherwise, the IRF is evaluated and averaged over the whole region.

    Parameters
    ----------
    geom : `~gammapy.maps.Geom`
        Geometry to project on. It must follow the required axes of the input IRF.
    irf : `~gammapy.irf.BackgroundIRF`
        IRF to reproject. Typically a background IRF.
    fov_frame : `~gammapy.utils.coordinate.FoVICRSFrame` or `~gammapy.utils.coordinate.FoVAltAzFrame`
        FoV frame to convert geometry to FoV coordinates.
    use_region_center : bool, optional
        For geom as a `~gammapy.maps.RegionGeom`. If True, consider the values at the region center.
        If False, average over the whole region.
        Default is True.

    Returns
    -------
    map : `~gammapy.maps.Map`
        Map containing the projected IRF.
    """
    from scipy.integrate import trapezoid

    if not use_region_center:
        image_geom = geom.to_wcs_geom().to_image()
        region_coord, weights = geom.get_wcs_coord_and_weights()
        skycoord = region_coord.skycoord
    else:
        image_geom = geom.to_image()
        skycoord = image_geom.get_coord().skycoord

    new_geom = geom
    # In case we need to integrate over time
    if len(fov_frame.shape) == 1:
        skycoord = skycoord[..., np.newaxis]

        # Assume ordered times
        time = MapAxis.from_edges(
            (fov_frame.obstime - fov_frame.obstime[0]).to_value("s"),
            unit="s",
            name="time",
        )
        axes = geom.axes

        new_geom = image_geom.to_cube([time, *axes])

    reverse_lon = irf.fov_alignment == "REVERSE_LON_RADEC"
    coords = _get_fov_coord(skycoord, fov_frame, irf.has_offset_axis, reverse_lon)

    non_spatial_axes = set(irf.required_arguments) - set(
        ["offset", "fov_lon", "fov_lat"]
    )

    for axis_name in non_spatial_axes:
        coords[axis_name] = broadcast_axis_values_to_geom(new_geom, axis_name, False)

    data = irf.integrate_log_log(**coords, axis_name="energy")

    if len(fov_frame.shape) == 1:
        time = new_geom.axes["time"]
        ontime = time.bin_width.sum().to_value("s")
        delta = np.reshape(time.edges.to_value("s"), (1, time.nbin + 1, 1, 1))
        data = trapezoid(data, delta, axis=1) / ontime

    if use_region_center:
        data *= image_geom.solid_angle()
    else:
        idx = image_geom.coord_to_idx(region_coord)
        data *= image_geom.solid_angle().T[idx]
        data = np.sum(weights * data, axis=2, keepdims=True)

    return Map.from_geom(geom=geom, data=data.value, unit=data.unit)
