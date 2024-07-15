# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from regions import PointSkyRegion
from gammapy.data import GTI, DataStore, EventList, FixedPointingInfo, Observation
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
)
from gammapy.maps import HpxGeom, MapAxis, RegionGeom, WcsGeom, WcsNDMap
from gammapy.modeling.models import ConstantSpectralModel
from gammapy.utils.testing import requires_data
from gammapy.utils.time import time_ref_to_dict


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
    time_start = Time("2000-09-21 11:55:00")
    time_stop = Time("2000-09-12 12:05:00")
    location = EarthLocation(lat=90 * u.deg, lon=0 * u.deg)
    fixed_icrs = SkyCoord(0 * u.deg, 0 * u.deg, frame="icrs")

    return FixedPointingInfo(
        fixed_icrs=fixed_icrs,
        location=location,
        time_start=time_start,
        time_stop=time_stop,
    )


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
    elif symmetry == "symmetric":
        data = np.ones((2, 3, 3))
        data[:, 1, 1] *= 2
    elif symmetry == "asymmetric":
        data = np.indices((3, 3))[1] + 1
        data = np.stack(2 * [data])
    else:
        raise ValueError(f"Unknown value for symmetry: {symmetry}")

    energy_axis = MapAxis.from_energy_edges([0.1, 10, 1000] * u.TeV)
    fov_lon_axis = MapAxis.from_edges([-3, -1, 1, 3] * u.deg, name="fov_lon")
    fov_lat_axis = MapAxis.from_edges([-3, -1, 1, 3] * u.deg, name="fov_lat")

    return Background3D(
        axes=[energy_axis, fov_lon_axis, fov_lat_axis],
        data=data,
        unit=u.Unit("s-1 MeV-1 sr-1"),
        interp_kwargs=dict(bounds_error=False, fill_value=None, values_scale="log"),
        fov_alignment=fov_align,
        # allow extrapolation for symmetry tests
    )


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
    )

    assert_allclose(bkg.data[:, 1, 1], [1.869025, 0.186903], rtol=1e-5)

    # Check that function works also passing the FixedPointingInfo
    bkg_fpi = make_map_background_irf(
        pointing=fixed_pointing_info,
        ontime="42 s",
        bkg=bkg_2d,
        geom=geom,
        obstime=obstime,
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
        obstime=obstime,
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
        oversampling=10,
        obstime=Time("2020-01-01T20:00"),
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
    geome = RegionGeom.create(region=on_region, axes=[energy_axis])
    counts = make_counts_rad_max(geome, observations.rad_max, observations.events)

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
            events["TIME"] = [0.1, 0.2, 0.3, 0.4, 0.5] * u.s

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
