# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import SkyCoord, AltAz
from astropy.table import Table
from astropy.time import Time
from regions import PointSkyRegion
from gammapy.data import (
    GTI,
    DataStore,
    EventList,
    FixedPointingInfo,
    Observation,
    observatory_locations,
)
from gammapy.irf import (
    Background2D,
    Background3D,
    EffectiveAreaTable2D,
    EnergyDispersion2D,
)
from gammapy.makers import WobbleRegionsFinder
from gammapy.makers.utils import (
    _map_spectrum_weight,
    guess_instrument_fov,
    make_counts_off_rad_max,
    make_counts_rad_max,
    make_edisp_kernel_map,
    make_effective_livetime_map,
    make_map_background_irf,
    make_map_exposure_true_energy,
    make_observation_time_map,
    make_theta_squared_table,
    project_irf_on_geom,
    integrate_project_irf_on_geom,
)
from gammapy.maps import HpxGeom, MapAxis, RegionGeom, WcsGeom, WcsNDMap
from gammapy.modeling.models import ConstantSpectralModel
from gammapy.utils.coordinates import FoVAltAzFrame
from gammapy.utils.testing import requires_data
from gammapy.utils.time import time_ref_to_dict
from gammapy.utils.coordinates import FoVICRSFrame


@pytest.fixture(scope="session")
def observations():
    """Example observation list for testing."""
    datastore = DataStore.from_dir("$GAMMAPY_DATA/magic/rad_max/data")
    return datastore.get_observations(required_irf="point-like")[0]


@pytest.fixture(scope="session")
def aeff():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    return EffectiveAreaTable2D.read(filename, hdu="EFFECTIVE AREA")


def geom(map_type, ebounds):
    axis = MapAxis.from_edges(ebounds, name="energy_true", unit="TeV", interp="log")
    if map_type == "wcs":
        return WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis])
    elif map_type == "hpx":
        return HpxGeom(256, axes=[axis])
    else:
        raise ValueError()


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            "geom": geom(map_type="wcs", ebounds=[0.1, 1, 10]),
            "shape": (2, 3, 4),
            "sum": 8.103974e08,
        },
        {
            "geom": geom(map_type="wcs", ebounds=[0.1, 10]),
            "shape": (1, 3, 4),
            "sum": 2.387916e08,
        },
        # TODO: make this work for HPX
        # 'HpxGeom' object has no attribute 'separation'
        # {
        #     'geom': geom(map_type='hpx', ebounds=[0.1, 1, 10]),
        #     'shape': '???',
        #     'sum': '???',
        # },
    ],
)
def test_make_map_exposure_true_energy(aeff, pars):
    m = make_map_exposure_true_energy(
        pointing=SkyCoord(2, 1, unit="deg"),
        livetime="42 s",
        aeff=aeff,
        geom=pars["geom"],
    )

    assert m.data.shape == pars["shape"]
    assert m.unit == "m2 s"
    assert_allclose(m.data.sum(), pars["sum"], rtol=1e-5)


def test_map_spectrum_weight():
    axis = MapAxis.from_edges([0.1, 10, 1000], unit="TeV", name="energy_true")
    expo_map = WcsNDMap.create(npix=10, binsz=1, axes=[axis], unit="m2 s")
    expo_map.data += 1
    spectrum = ConstantSpectralModel(const="42 cm-2 s-1 TeV-1")

    weighted_expo = _map_spectrum_weight(expo_map, spectrum)

    assert weighted_expo.data.shape == (2, 10, 10)
    assert weighted_expo.unit == "m2 s"
    assert_allclose(weighted_expo.data.sum(), 100)


@pytest.fixture(scope="session")
def fixed_pointing_info():
    filename = "$GAMMAPY_DATA/cta-1dc/data/baseline/gps/gps_baseline_110380.fits"
    return FixedPointingInfo.read(filename)


@pytest.fixture(scope="session")
def fixed_pointing_info_aligned():
    # Create Fixed Pointing Info aligned between sky and horizon coordinates
    # (removes rotation in FoV and results in predictable solid angles)
    fixed_icrs = SkyCoord(0 * u.deg, 0 * u.deg, frame="icrs")

    return FixedPointingInfo(fixed_icrs=fixed_icrs)


@pytest.fixture(scope="session")
def bkg_3d():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    return Background3D.read(filename, hdu="BACKGROUND")


@pytest.fixture(scope="session")
def bkg_2d():
    offset_axis = MapAxis.from_bounds(0, 4, nbin=10, name="offset", unit="deg")
    energy_axis = MapAxis.from_energy_bounds("0.1 TeV", "10 TeV", nbin=20)
    bkg_2d = Background2D(axes=[energy_axis, offset_axis], unit="s-1 TeV-1 sr-1")
    coords = bkg_2d.axes.get_coord()
    value = np.exp(-0.5 * (coords["offset"] / (2 * u.deg)) ** 2)
    bkg_2d.data = (value * (coords["energy"] / (1 * u.TeV)) ** -2).to_value("")
    return bkg_2d


def bkg_3d_custom(symmetry="constant", fov_align="RADEC"):
    if symmetry == "constant":
        data = np.ones((2, 3, 3))
        sky_edges = [-3, -1, 1, 3] * u.deg
    elif symmetry == "symmetric":
        data = np.ones((2, 3, 3))
        data[:, 1, 1] *= 2
        sky_edges = [-3, -1, 1, 3] * u.deg
    elif symmetry == "hirez_symmetric":
        data = np.ones((3, 3))
        data[1, 1] *= 2
        data = (  # Upscale 3x3 to 9x9 by simply repeating
            data.repeat(3, axis=0).repeat(3, axis=1)[np.newaxis, ...].repeat(2, axis=0)
        )
        sky_edges = np.linspace(-3, 3, 10) * u.deg
    elif symmetry == "asymmetric":
        data = np.indices((3, 3))[1] + 1
        data = np.stack(2 * [data])
        sky_edges = [-3, -1, 1, 3] * u.deg
    else:
        raise ValueError(f"Unknown value for symmetry: {symmetry}")

    energy_axis = MapAxis.from_energy_edges([0.1, 10, 1000] * u.TeV)
    fov_lon_axis = MapAxis.from_edges(sky_edges, name="fov_lon")
    fov_lat_axis = MapAxis.from_edges(sky_edges, name="fov_lat")
    return Background3D(
        axes=[energy_axis, fov_lon_axis, fov_lat_axis],
        data=data,
        unit=u.Unit("s-1 MeV-1 sr-1"),
        interp_kwargs=dict(bounds_error=False, fill_value=None, values_scale="log"),
        fov_alignment=fov_align,
        # allow extrapolation for symmetry tests
    )


def aeff_custom(energy_axis, upscale=3):
    offset = MapAxis.from_bounds(
        -0.25, 2.75, nbin=6 * upscale, unit="deg", name="offset"
    )
    aeff_vals = np.zeros((energy_axis.nbin, offset.nbin))
    Es = energy_axis.center.value
    # Chosen to roughly match HESS areas
    scales = np.repeat(np.array([62, 62, 59, 51, 42, 34]) * 1e4, upscale)
    cores = np.repeat(np.array([0.48, 0.49, 0.52, 0.65, 1.01, 1.61]), upscale)
    ids = list(range(0, offset.nbin))
    for idx, scale, core in zip(ids, scales, cores):
        aeff_vals[:, idx] = (
            scale * core**2 * (1 / (np.sqrt(Es**2 + core**2)) - 1 / core) ** 2
        )

    return EffectiveAreaTable2D([energy_axis, offset], data=aeff_vals, unit="m2")


@requires_data()
def test_map_background_2d(bkg_2d, fixed_pointing_info):
    axis = MapAxis.from_edges([0.1, 1, 10], name="energy", unit="TeV", interp="log")

    obstime = Time("2020-01-01T20:00:00")
    skydir = fixed_pointing_info.get_icrs(obstime).galactic
    geom = WcsGeom.create(
        npix=(3, 3), binsz=4, axes=[axis], skydir=skydir, frame="galactic"
    )

    bkg = make_map_background_irf(
        pointing=skydir,
        ontime="42 s",
        bkg=bkg_2d,
        geom=geom,
        time_start=obstime,
        fov_rotation_step=1.0 * u.deg,
    )

    assert_allclose(bkg.data[:, 1, 1], [1.869025, 0.186903], rtol=1e-5)

    # Check that function works also passing the FixedPointingInfo
    bkg_fpi = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42 s",
        bkg=bkg_2d,
        geom=geom,
        time_start=obstime,
        fov_rotation_step=1.0 * u.deg,
    )
    assert_allclose(bkg.data, bkg_fpi.data, rtol=1e-5)


def make_map_background_irf_with_symmetry(fpi, symmetry="constant"):
    axis = MapAxis.from_edges([0.1, 1, 10], name="energy", unit="TeV", interp="log")
    obstime = Time("2020-01-01T20:00:00")
    return make_map_background_irf(
        pointing=fpi,
        ontime="42 s",
        bkg=bkg_3d_custom(symmetry),
        geom=WcsGeom.create(npix=(3, 3), binsz=4, axes=[axis], skydir=fpi.fixed_icrs),
        time_start=obstime,
        fov_rotation_step=1.0 * u.deg,
    )


def geom(map_type, ebounds, skydir):
    axis = MapAxis.from_edges(ebounds, name="energy", unit="TeV", interp="log")
    if map_type == "wcs":
        return WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis], skydir=skydir)
    elif map_type == "hpx":
        return HpxGeom(256, axes=[axis])
    else:
        raise ValueError()


@requires_data()
@pytest.mark.parametrize(
    "pars",
    [
        {
            "map_type": "wcs",
            "ebounds": [0.1, 1, 10],
            "shape": (2, 3, 4),
            "sum": 1051.960299,
        },
        {
            "map_type": "wcs",
            "ebounds": [0.1, 10],
            "shape": (1, 3, 4),
            "sum": 1051.960299,
        },
        # TODO: make this work for HPX
        # 'HpxGeom' object has no attribute 'separation'
        # {
        #     'geom': geom(map_type='hpx', ebounds=[0.1, 1, 10]),
        #     'shape': '???',
        #     'sum': '???',
        # },
    ],
)
def test_make_map_background_irf(bkg_3d, pars, fixed_pointing_info):
    m = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42 s",
        bkg=bkg_3d,
        geom=geom(
            map_type=pars["map_type"],
            ebounds=pars["ebounds"],
            skydir=fixed_pointing_info.fixed_icrs,
        ),
        time_start=Time("2020-01-01T20:00"),
        fov_rotation_step=1.0 * u.deg,
        oversampling=10,
    )

    assert m.data.shape == pars["shape"]
    assert m.unit == ""
    assert_allclose(m.data.sum(), pars["sum"], rtol=1e-5)


@requires_data()
def test_make_map_background_irf_constant(fixed_pointing_info_aligned):
    m = make_map_background_irf_with_symmetry(
        fpi=fixed_pointing_info_aligned, symmetry="constant"
    )
    for d in m.data:
        assert_allclose(d[1, :], d[1, 0])  # Constant along lon
        assert_allclose(d[0, 1], d[2, 1])  # Symmetric along lat
        with pytest.raises(AssertionError):
            # Not constant along lat due to changes in
            # solid angle (great circle)
            assert_allclose(d[:, 1], d[0, 1])


@requires_data()
def test_make_map_background_irf_sym(fixed_pointing_info_aligned):
    m = make_map_background_irf_with_symmetry(
        fpi=fixed_pointing_info_aligned, symmetry="symmetric"
    )
    for d in m.data:
        assert_allclose(d[1, 0], d[1, 2], rtol=1e-4)  # Symmetric along lon
        assert_allclose(d[0, 1], d[2, 1], rtol=1e-4)  # Symmetric along lat


@requires_data()
def test_make_map_background_irf_asym(fixed_pointing_info_aligned):
    m = make_map_background_irf_with_symmetry(
        fpi=fixed_pointing_info_aligned, symmetry="asymmetric"
    )
    for d in m.data:
        # TODO:
        #  Dimensions of skymap data are [energy, lat, lon] (and is
        #  represented as [lon, lat, energy] in the api, but the bkg irf
        #  dimensions are currently [energy, lon, lat] - Will be changed in
        #  the future (perhaps when IRFs use the skymaps class)
        assert_allclose(d[1, 0], d[1, 2], rtol=1e-4)  # Symmetric along lon
        with pytest.raises(AssertionError):
            assert_allclose(d[0, 1], d[2, 1], rtol=1e-4)  # Symmetric along lat
        assert_allclose(d[0, 1] * 9, d[2, 1], rtol=1e-4)  # Asymmetric along lat


@requires_data()
def test_make_map_background_irf_skycoord(fixed_pointing_info_aligned):
    axis = MapAxis.from_edges([0.1, 1, 10], name="energy", unit="TeV", interp="log")
    position = fixed_pointing_info_aligned.fixed_icrs
    with pytest.raises(TypeError):
        make_map_background_irf(
            pointing=position,
            ontime="42 s",
            bkg=bkg_3d_custom("asymmetric", "ALTAZ"),
            geom=WcsGeom.create(npix=(3, 3), binsz=4, axes=[axis], skydir=position),
            time_start=Time("2020-01-01T20:00:00"),
            fov_rotation_step=1.0 * u.deg,
        )


@requires_data()
def test_make_map_background_irf_altaz_align(fixed_pointing_info):
    def _get_geom(pnt_info, time):
        axis = MapAxis.from_edges([0.1, 1, 10], name="energy", unit="TeV", interp="log")

        return WcsGeom.create(
            npix=(10, 10),
            binsz=0.1,
            axes=[axis],
            skydir=pnt_info.get_icrs(time),
        )

    obstime = Time("2020-01-01T20:00:00")
    location = observatory_locations["ctao_south"]

    map_long_altaz = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42000 s",
        bkg=bkg_3d_custom("asymmetric", "ALTAZ"),
        geom=_get_geom(fixed_pointing_info, obstime),
        time_start=obstime,
        fov_rotation_step=20.0 * u.deg,
        location=location,
    )

    map_short_altaz = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42 s",
        bkg=bkg_3d_custom("asymmetric", "ALTAZ"),
        geom=_get_geom(fixed_pointing_info, obstime),
        time_start=obstime + "20979 s",
        fov_rotation_step=20.0 * u.deg,
        location=location,
    )
    map_long_radec = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42000 s",
        bkg=bkg_3d_custom("asymmetric", "RADEC"),
        geom=_get_geom(fixed_pointing_info, obstime),
        time_start=obstime,
        fov_rotation_step=20.0 * u.deg,
        location=location,
    )
    map_short_altaz_norotation = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42 s",
        bkg=bkg_3d_custom("asymmetric", "ALTAZ"),
        geom=_get_geom(fixed_pointing_info, obstime),
        time_start=obstime - 21 * u.s,
        fov_rotation_step=360.0 * u.deg,
        location=location,
    )
    map_altaz_long_norotation = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="4200 s",
        bkg=bkg_3d_custom("asymmetric", "ALTAZ"),
        geom=_get_geom(fixed_pointing_info, obstime),
        time_start=obstime - 2100 * u.s,
        fov_rotation_step=360.0 * u.deg,
        location=location,
    )
    # Check that background normalisations are consistent
    assert_allclose(np.sum(map_long_altaz.data), np.sum(map_long_radec.data), rtol=1e-2)
    assert np.isclose(
        np.sum(map_long_altaz.data), 1000 * np.sum(map_short_altaz.data), rtol=1e-2
    )
    # Check that results differ when considering short and long observations with
    # AltAz aligned IRFs
    assert_allclose(
        map_long_altaz.data[0, 0, :4], [252123, 250654, 249086, 247564.0], rtol=1e-2
    )
    assert_allclose(
        map_short_altaz.data[0, 0, :4],
        [260.9476, 258.2620, 255.6040, 252.973375],
        rtol=1e-5,
    )

    # Check that results differ when considering RaDec or AltAz aligned IRFs
    assert_allclose(
        map_long_radec.data[0, 0, :4], [197029, 197029, 197029, 197029.0], rtol=1e-5
    )

    # Check that results are independent of the observation duration with AltAz
    # aligned IRFs when the FoV rotation is ignored
    assert_allclose(
        map_altaz_long_norotation.data,
        map_short_altaz_norotation.data * 100,
        rtol=1e-2,
    )


def test_make_edisp_kernel_map():
    migra = MapAxis.from_edges(np.linspace(0.5, 1.5, 50), unit="", name="migra")
    etrue = MapAxis.from_energy_bounds(0.5, 2, 6, unit="TeV", name="energy_true")
    offset = MapAxis.from_edges(np.linspace(0.0, 2.0, 3), unit="deg", name="offset")
    ereco = MapAxis.from_energy_bounds(0.5, 2, 3, unit="TeV", name="energy")

    edisp = EnergyDispersion2D.from_gauss(
        energy_axis_true=etrue, migra_axis=migra, bias=0, sigma=0.01, offset_axis=offset
    )

    geom = WcsGeom.create(10, binsz=0.5, axes=[ereco, etrue])
    pointing = SkyCoord(0, 0, frame="icrs", unit="deg")
    edispmap = make_edisp_kernel_map(edisp, pointing, geom)

    kernel = edispmap.get_edisp_kernel(position=pointing)
    assert_allclose(kernel.pdf_matrix[:, 0], (1.0, 1.0, 0.0, 0.0, 0.0, 0.0), atol=1e-14)
    assert_allclose(kernel.pdf_matrix[:, 1], (0.0, 0.0, 1.0, 1.0, 0.0, 0.0), atol=1e-14)
    assert_allclose(kernel.pdf_matrix[:, 2], (0.0, 0.0, 0.0, 0.0, 1.0, 1.0), atol=1e-14)


@requires_data()
def test_make_counts_rad_max(observations):
    pos = SkyCoord(083.6331144560900, +22.0144871383400, unit="deg", frame="icrs")
    on_region = PointSkyRegion(pos)
    energy_axis = MapAxis.from_energy_bounds(
        0.05, 100, nbin=6, unit="TeV", name="energy"
    )
    geom = RegionGeom.create(region=on_region, axes=[energy_axis])
    counts = make_counts_rad_max(geom, observations.rad_max, observations.events)

    assert_allclose(np.squeeze(counts.data), np.array([547, 188, 52, 8, 0, 0]))


@requires_data()
def test_make_counts_off_rad_max(observations):
    pos = SkyCoord(83.6331, +22.0145, unit="deg", frame="icrs")
    on_region = PointSkyRegion(pos)
    energy_axis = MapAxis.from_energy_bounds(
        0.05, 100, nbin=6, unit="TeV", name="energy"
    )

    region_finder = WobbleRegionsFinder(n_off_regions=3)
    region_off, wcs = region_finder.run(on_region, pos)
    geom_off = RegionGeom.from_regions(regions=region_off, axes=[energy_axis], wcs=wcs)

    counts_off = make_counts_off_rad_max(
        geom_off=geom_off, rad_max=observations.rad_max, events=observations.events
    )

    assert_allclose(np.squeeze(counts_off.data), np.array([1641, 564, 156, 24, 0, 0]))


class TestTheta2Table:
    def setup_class(self):
        self.observations = []
        for sign in [-1, 1]:
            events = Table()
            events["RA"] = [0.0, 0.0, 0.0, 0.0, 10.0] * u.deg
            events["DEC"] = sign * ([0.0, 0.05, 0.9, 10.0, 10.0] * u.deg)
            events["ENERGY"] = [1.0, 1.0, 1.5, 1.5, 10.0] * u.TeV
            events["OFFSET"] = [0.1, 0.1, 0.5, 1.0, 1.5] * u.deg
            events["TIME"] = Time("2025-01-01") + [0.1, 0.2, 0.3, 0.4, 0.5] * u.s

            obs_info = dict(
                OBS_ID=0,
                DEADC=1,
                GEOLON=16.500222222222224,
                GEOLAT=-23.271777777777775,
                ALTITUDE=1834.9999999997833,
            )

            meta = time_ref_to_dict("2010-01-01")
            obs_info.update(meta)
            events.meta.update(obs_info)
            gti = GTI.create(
                start=[1] * u.s,
                stop=[3] * u.s,
                reference_time=Time("2010-01-01", scale="tt"),
            )
            pointing = FixedPointingInfo(
                fixed_icrs=SkyCoord(0 * u.deg, sign * 0.5 * u.deg),
            )

            self.observations.append(
                Observation(
                    events=EventList(events),
                    gti=gti,
                    pointing=pointing,
                )
            )

    def test_make_theta_squared_table(self):
        # pointing position: (0,0.5) degree in ra/dec
        # On theta2 distribution compute from (0,0) in ra/dec.
        # OFF theta2 distribution from the mirror position at (0,1) in ra/dec.
        position = SkyCoord(ra=0, dec=0, unit="deg", frame="icrs")
        axis = MapAxis.from_bounds(0, 0.2, nbin=4, interp="lin", unit="deg2")
        theta2_table = make_theta_squared_table(
            observations=[self.observations[0]],
            position=position,
            theta_squared_axis=axis,
        )
        theta2_lo = [0, 0.05, 0.1, 0.15]
        theta2_hi = [0.05, 0.1, 0.15, 0.2]
        on_counts = [2, 0, 0, 0]
        off_counts = [1, 0, 0, 0]
        acceptance = [1, 1, 1, 1]
        acceptance_off = [1, 1, 1, 1]
        alpha = [1, 1, 1, 1]
        assert len(theta2_table) == 4
        assert theta2_table["theta2_min"].unit == "deg2"
        assert_allclose(theta2_table["theta2_min"], theta2_lo)
        assert_allclose(theta2_table["theta2_max"], theta2_hi)
        assert_allclose(theta2_table["counts"], on_counts)
        assert_allclose(theta2_table["counts_off"], off_counts)
        assert_allclose(theta2_table["acceptance"], acceptance)
        assert_allclose(theta2_table["acceptance_off"], acceptance_off)
        assert_allclose(theta2_table["alpha"], alpha)
        assert_allclose(theta2_table.meta["ON_RA"], 0 * u.deg)
        assert_allclose(theta2_table.meta["ON_DEC"], 0 * u.deg)

        # Taking the off position as the on one
        off_position = position
        theta2_table2 = make_theta_squared_table(
            observations=[self.observations[0]],
            position=position,
            theta_squared_axis=axis,
            position_off=off_position,
        )

        assert_allclose(theta2_table2["counts_off"], theta2_table["counts"])

        # Test for two observations, here identical
        theta2_table_two_obs = make_theta_squared_table(
            observations=self.observations,
            position=position,
            theta_squared_axis=axis,
        )
        on_counts_two_obs = [4, 0, 0, 0]
        off_counts_two_obs = [2, 0, 0, 0]
        acceptance_two_obs = [2, 2, 2, 2]
        acceptance_off_two_obs = [2, 2, 2, 2]
        alpha_two_obs = [1, 1, 1, 1]
        assert_allclose(theta2_table_two_obs["counts"], on_counts_two_obs)
        assert_allclose(theta2_table_two_obs["counts_off"], off_counts_two_obs)
        assert_allclose(theta2_table_two_obs["acceptance"], acceptance_two_obs)
        assert_allclose(theta2_table_two_obs["acceptance_off"], acceptance_off_two_obs)
        assert_allclose(theta2_table["alpha"], alpha_two_obs)

        # Test for energy selection
        axis = MapAxis.from_bounds(0, 1, nbin=4, interp="lin", unit="deg2")
        theta2_table = make_theta_squared_table(
            observations=[self.observations[0]],
            position=position,
            theta_squared_axis=axis,
            energy_edges=[1.2, 11] * u.TeV,
        )
        on_counts = [0, 0, 0, 1]
        off_counts = [1, 0, 0, 0]
        acceptance = [1, 1, 1, 1]
        acceptance_off = [1, 1, 1, 1]
        alpha = [1, 1, 1, 1]
        assert_allclose(theta2_table["counts"], on_counts)
        assert_allclose(theta2_table["counts_off"], off_counts)
        assert_allclose(theta2_table["acceptance"], acceptance)
        assert_allclose(theta2_table["acceptance_off"], acceptance_off)
        assert_allclose(theta2_table["alpha"], alpha)
        assert_allclose(theta2_table.meta["Energy_filter"], [1.2, 11] * u.TeV)

        with pytest.raises(ValueError):
            make_theta_squared_table(
                observations=[self.observations[0]],
                position=position,
                theta_squared_axis=axis,
                energy_edges=[1.2, 11, 20] * u.TeV,
            )


@requires_data()
def test_guess_instrument_fov(observations):
    with pytest.raises(ValueError):
        guess_instrument_fov(observations)

    ds = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs_hess = ds.obs(23523)

    assert_allclose(guess_instrument_fov(obs_hess), 2.5 * u.deg)

    obs_no_aeff = obs_hess.copy(in_memory=True, aeff=None)
    with pytest.raises(ValueError):
        guess_instrument_fov(obs_no_aeff)


@requires_data()
def test_make_observation_time_map():
    ds = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs_id = ds.obs_table["OBS_ID"][ds.obs_table["OBJECT"] == "MSH 15-5-02"][:3]
    observations = ds.get_observations(obs_id)
    source_pos = SkyCoord(228.32, -59.08, unit="deg")
    geom = WcsGeom.create(
        skydir=source_pos,
        binsz=0.02,
        width=(6, 6),
        frame="icrs",
        proj="CAR",
    )
    obs_time = make_observation_time_map(observations, geom, offset_max=2.5 * u.deg)
    obs_time_center = obs_time.get_by_coord(source_pos)
    assert_allclose(obs_time_center, 1.2847, rtol=1e-3)
    assert obs_time.unit == u.hr


@requires_data()
def test_make_effective_livetime_map():
    ds = DataStore.from_dir("$GAMMAPY_DATA/hess-dl3-dr1")
    obs_id = ds.obs_table["OBS_ID"][ds.obs_table["OBJECT"] == "MSH 15-5-02"][:3]
    observations = ds.get_observations(obs_id)
    source_pos = SkyCoord(228.32, -59.08, unit="deg")
    offset_pos = SkyCoord(322.00, 0.1, unit="deg", frame="galactic")

    energy_axis_true = MapAxis.from_energy_bounds(
        10 * u.GeV, 1 * u.TeV, nbin=2, name="energy_true"
    )
    geom = WcsGeom.create(
        skydir=source_pos,
        binsz=0.02,
        width=(6, 6),
        frame="galactic",
        proj="CAR",
        axes=[energy_axis_true],
    )
    obs_time = make_effective_livetime_map(observations, geom, offset_max=2.5 * u.deg)
    obs_time_center = obs_time.get_by_coord((source_pos, energy_axis_true.center))
    assert_allclose(obs_time_center, [0, 1.2847], rtol=1e-3)

    obs_time_offset = obs_time.get_by_coord((offset_pos, energy_axis_true.center))
    assert_allclose(obs_time_offset, [0, 0.242814], rtol=1e-3)

    assert obs_time.unit == u.hr


def test_project_irf_on_geom():
    location = observatory_locations.get("ctao_north")
    crab = SkyCoord(83.63333333, 22.01444444, unit="deg", frame="icrs")

    # Test projection to geom aligned with the FOV binning and orientation
    obstime = Time("2025-01-01T00:00:00")
    origin = crab.transform_to(AltAz(location=location, obstime=obstime))
    fov_frame = FoVAltAzFrame(origin=origin, location=location, obstime=obstime)

    axis = MapAxis.from_energy_bounds(
        energy_min=0.1, energy_max=10, nbin=2, name="energy_true", unit="TeV"
    )
    sky_geom = WcsGeom.create(
        npix=(3, 3), binsz=0.5, axes=[axis], skydir=crab, proj="TAN"
    )
    aeff = aeff_custom(axis)
    sky_irf = project_irf_on_geom(sky_geom, aeff, fov_frame)

    ref0 = aeff.evaluate(offset=0 * u.deg).flatten().value
    ref1 = aeff.evaluate(offset=sky_geom.separation(crab)[0, 1]).flatten().value

    assert_allclose(ref0, sky_irf.data[:, 1, 1])
    assert_allclose(ref1, sky_irf.data[:, 0, 1])
    assert_allclose(ref1, sky_irf.data[:, 1, 0])

    obs_times = obstime - np.linspace(0, 2.15, 2) * u.minute

    fov_frame = FoVAltAzFrame(origin=origin, location=location, obstime=obs_times)
    aeff = aeff_custom(axis, upscale=5)
    sky_irf = project_irf_on_geom(sky_geom, aeff, fov_frame)

    ref3 = (ref0 + ref1) / 2
    assert_allclose(ref3, sky_irf.data[:, 1, 1])


def test_integrate_project_irf_on_geom():
    location = observatory_locations.get("ctao_north")
    crab = SkyCoord(83.63333333, 22.01444444, unit="deg", frame="icrs")

    # Test projection to geom aligned with the FOV binning and orientation
    obstime = Time("2025-01-01T00:04:00")
    origin = crab.transform_to(AltAz(location=location, obstime=obstime))
    axis = MapAxis.from_edges([0.1, 1.1, 11.1], name="energy", unit="TeV", interp="log")
    sky_geom = WcsGeom.create(
        npix=(3, 3), binsz=2, axes=[axis], skydir=crab, proj="TAN"
    )

    fov_frame = FoVAltAzFrame(origin=origin, location=location, obstime=obstime)
    bkg_irf = bkg_3d_custom(symmetry="asymmetric")
    bkg_sky = integrate_project_irf_on_geom(sky_geom, bkg_irf, fov_frame)

    ref1 = (
        1
        * u.TeV
        * bkg_irf.evaluate(fov_lon=0 * u.deg, fov_lat=0 * u.deg, energy=1 * u.TeV)
        * sky_geom.solid_angle()[0, 1, 1]
    )
    ref2 = (
        10
        * u.TeV
        * bkg_irf.evaluate(fov_lon=0 * u.deg, fov_lat=2 * u.deg, energy=1 * u.TeV)
        * sky_geom.solid_angle()[0, 1, 2]
    )

    assert_allclose(bkg_sky.data[0, 1, 1], ref1.value, rtol=1e-9)
    assert_allclose(bkg_sky.data[1, 2, 1], ref2.value, rtol=1e-3)

    assert bkg_sky.unit.is_equivalent(1 / u.s)

    bkg_irf = bkg_3d_custom(symmetry="hirez_symmetric")
    bkg_sky = integrate_project_irf_on_geom(sky_geom, bkg_irf, fov_frame)
    ref3 = (  # center center pixel, value = 2
        1
        * u.TeV
        * bkg_irf.evaluate(fov_lon=0 * u.deg, fov_lat=0 * u.deg, energy=1 * u.TeV)
    ).value
    ref4 = (  # right edge center pixel, value = 1
        1
        * u.TeV
        * bkg_irf.evaluate(fov_lon=0 * u.deg, fov_lat=2 * u.deg, energy=1 * u.TeV)
    ).value
    omega11 = sky_geom.solid_angle()[0, 1, 1].value  # center center pixel
    omega12 = sky_geom.solid_angle()[0, 1, 2].value  # right edge center pixel
    omega21 = sky_geom.solid_angle()[0, 2, 1].value  # bottom edge center pixel
    assert_allclose(bkg_sky.data[0, 1, 1], ref3 * omega11, rtol=1e-9)
    assert_allclose(bkg_sky.data[0, 1, 2], ref4 * omega12, rtol=1e-9)
    # Times chosen such that at second step the IRF FoV has moved
    # over one full geom pixel
    obs_times = obstime - np.linspace(0, 8.6, 2) * u.minute
    fov_timedep = FoVAltAzFrame(origin=origin, location=location, obstime=obs_times)
    bkg_sky = integrate_project_irf_on_geom(sky_geom, bkg_irf, fov_timedep)

    assert_allclose(
        (ref3 + ref4) / 2 * omega11,
        bkg_sky.data[0, 1, 1],
    )
    assert_allclose(
        (ref3 + ref4) / 2 * omega12,
        bkg_sky.data[0, 1, 2],
    )
    with pytest.raises(AssertionError):
        assert_allclose(
            (ref3 + ref4) / 2 * omega21,
            bkg_sky.data[0, 2, 1],
        )
    assert_allclose(
        (ref4 + ref4) / 2 * omega21,
        bkg_sky.data[0, 2, 1],
    )


@requires_data()
def test_project_irf(aeff):
    ebounds = [0.1, 1, 10]
    axis = MapAxis.from_edges(ebounds, name="energy_true", unit="TeV", interp="log")
    geom = WcsGeom.create(npix=(4, 3), binsz=2, axes=[axis])
    pointing = SkyCoord(2, 1, unit="deg")
    fov_frame = FoVICRSFrame(origin=pointing)
    proj_irf_geom = project_irf_on_geom(geom, aeff, fov_frame)

    assert geom.data_shape == proj_irf_geom.data.shape
    assert_allclose(
        proj_irf_geom.data[:, 1, 1], [373700.92608812, 2477509.59635073], rtol=1e-5
    )
    assert proj_irf_geom.geom.center_skydir == geom.center_skydir


@requires_data()
def test_integrate_project_irf(bkg_2d, fixed_pointing_info):
    axis = MapAxis.from_edges([0.1, 1, 10], name="energy", unit="TeV", interp="log")
    obstime = Time("2020-01-01T20:00:00")
    skydir = fixed_pointing_info.get_icrs(obstime).galactic
    geom = WcsGeom.create(
        npix=(3, 3), binsz=4, axes=[axis], skydir=skydir, frame="galactic"
    )
    fov_frame = FoVICRSFrame(origin=skydir)
    int_proj_irf_geom = integrate_project_irf_on_geom(geom, bkg_2d, fov_frame)

    assert geom.data_shape == int_proj_irf_geom.data.shape
    assert_allclose(int_proj_irf_geom.data[:, 1, 1], [0.0445006, 0.00445006], rtol=1e-5)
    assert int_proj_irf_geom.geom.center_skydir == geom.center_skydir
