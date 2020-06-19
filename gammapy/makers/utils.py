# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy.coordinates import SkyOffsetFrame
from gammapy.data import FixedPointingInfo
from gammapy.irf import EDispMap, PSFMap
from gammapy.maps import Map, WcsNDMap
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.utils.coordinates import sky_to_fov

__all__ = [
    "make_map_background_irf",
    "make_edisp_map",
    "make_edisp_kernel_map",
    "make_psf_map",
    "make_map_exposure_true_energy",
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
    energy_edges = map.geom.get_axis_by_name("energy_true").edges
    weights = spectrum.integral(
        emin=energy_edges[:-1], emax=energy_edges[1:], intervals=True
    )
    weights /= weights.sum()
    shape = np.ones(len(map.geom.data_shape))
    shape[0] = -1
    return map * weights.reshape(shape.astype(int))


def make_map_background_irf(pointing, ontime, bkg, geom, oversampling=None):
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
        geom = geom.upsample(factor=oversampling, axis="energy")

    map_coord = geom.to_image().get_coord()
    sky_coord = map_coord.skycoord

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

    energies = geom.get_axis_by_name("energy").edges

    bkg_de = bkg.evaluate_integrate(
        fov_lon=fov_lon,
        fov_lat=fov_lat,
        energy_reco=energies[:, np.newaxis, np.newaxis],
    )

    d_omega = geom.to_image().solid_angle()
    data = (bkg_de * d_omega * ontime).to_value("")
    bkg_map = WcsNDMap(geom, data=data)

    if oversampling is not None:
        bkg_map = bkg_map.downsample(factor=oversampling, axis="energy")

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
    psfmap : `~gammapy.cube.PSFMap`
        the resulting PSF map
    """
    energy_axis = geom.get_axis_by_name("energy_true")
    energy = energy_axis.center

    rad_axis = geom.get_axis_by_name("theta")
    rad = rad_axis.center

    # Compute separations with pointing position
    offset = geom.separation(pointing)

    # Compute PSF values
    # TODO: allow broadcasting in PSF3D.evaluate()
    psf_values = psf._interpolate(
        (
            rad[:, np.newaxis, np.newaxis],
            offset,
            energy[:, np.newaxis, np.newaxis, np.newaxis],
        )
    )

    # TODO: this probably does not ensure that probability is properly normalized in the PSFMap
    # Create Map and fill relevant entries
    data = psf_values.to_value("sr-1")
    psfmap = Map.from_geom(geom, data=data, unit="sr-1")
    return PSFMap(psfmap, exposure_map)


def make_edisp_map(edisp, pointing, geom, exposure_map=None):
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

    Returns
    -------
    edispmap : `~gammapy.cube.EDispMap`
        the resulting EDisp map
    """
    energy_axis = geom.get_axis_by_name("energy_true")
    energy = energy_axis.center

    migra_axis = geom.get_axis_by_name("migra")
    migra = migra_axis.center

    # Compute separations with pointing position
    offset = geom.separation(pointing)

    # Compute EDisp values
    edisp_values = edisp.data.evaluate(
        offset=offset,
        energy_true=energy[:, np.newaxis, np.newaxis, np.newaxis],
        migra=migra[:, np.newaxis, np.newaxis],
    )

    # Create Map and fill relevant entries
    data = edisp_values.to_value("")
    edispmap = Map.from_geom(geom, data=data, unit="")
    return EDispMap(edispmap, exposure_map)


def make_edisp_kernel_map(edisp, pointing, geom, exposure_map=None):
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

    Returns
    -------
    edispmap : `~gammapy.cube.EDispKernelMap`
        the resulting EDispKernel map
    """
    # Use EnergyDispersion2D migra axis.
    migra_axis = edisp.data.axis("migra")

    # Create temporary EDispMap Geom
    new_geom = geom.to_image().to_cube(
        [migra_axis, geom.get_axis_by_name("energy_true")]
    )

    edisp_map = make_edisp_map(edisp, pointing, new_geom, exposure_map)

    return edisp_map.to_edisp_kernel_map(geom.get_axis_by_name("energy"))
