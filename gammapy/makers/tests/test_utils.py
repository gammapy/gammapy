# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from gammapy.data import GTI, EventList, FixedPointingInfo, Observation
from gammapy.irf import Background3D, EffectiveAreaTable2D, EnergyDispersion2D
from gammapy.makers.utils import (
    _map_spectrum_weight,
    make_edisp_kernel_map,
    make_map_background_irf,
    make_map_exposure_true_energy,
    make_theta_squared_table,
)
from gammapy.maps import HpxGeom, MapAxis, WcsGeom, WcsNDMap
from gammapy.modeling.models import ConstantSpectralModel
from gammapy.utils.testing import requires_data
from gammapy.utils.time import time_ref_to_dict


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
def fixed_pointing_info_aligned(fixed_pointing_info):
    # Create Fixed Pointing Info aligined between sky and horizon coordinates
    # (removes rotation in FoV and results in predictable solid angles)
    origin = SkyCoord(
        0,
        0,
        unit="deg",
        frame="icrs",
        location=EarthLocation(lat=90 * u.deg, lon=0 * u.deg),
        obstime=Time("2000-9-21 12:00:00"),
    )
    fpi = fixed_pointing_info
    meta = fpi.meta.copy()
    meta["RA_PNT"] = origin.icrs.ra
    meta["DEC_PNT"] = origin.icrs.dec
    meta["GEOLON"] = origin.location.lon
    meta["GEOLAT"] = origin.location.lat
    meta["ALTITUDE"] = origin.location.height
    time_start = origin.obstime.datetime - fpi.time_ref.datetime
    meta["TSTART"] = time_start.total_seconds()
    meta["TSTOP"] = meta["TSTART"] + 60
    return FixedPointingInfo(meta)


@pytest.fixture(scope="session")
def bkg_3d():
    filename = (
        "$GAMMAPY_DATA/cta-1dc/caldb/data/cta/1dc/bcf/South_z20_50h/irf_file.fits"
    )
    return Background3D.read(filename, hdu="BACKGROUND")


def bkg_3d_custom(symmetry="constant"):
    if symmetry == "constant":
        data = np.ones((2, 3, 3)) * u.Unit("s-1 MeV-1 sr-1")
    elif symmetry == "symmetric":
        data = np.ones((2, 3, 3)) * u.Unit("s-1 MeV-1 sr-1")
        data[:, 1, 1] *= 2
    elif symmetry == "asymmetric":
        data = np.indices((3, 3))[1] + 1
        data = np.stack(2 * [data]) * u.Unit("s-1 MeV-1 sr-1")
    else:
        raise ValueError(f"Unkown value for symmetry: {symmetry}")

    energy_axis = MapAxis.from_energy_edges([0.1, 10, 1000] * u.TeV)
    fov_lon_axis = MapAxis.from_edges([-3, -1, 1, 3] * u.deg, name="fov_lon")
    fov_lat_axis = MapAxis.from_edges([-3, -1, 1, 3] * u.deg, name="fov_lat")
    return Background3D(
        energy_axis=energy_axis,
        fov_lat_axis=fov_lat_axis,
        fov_lon_axis=fov_lon_axis,
        data=data,
    )


def make_map_background_irf_with_symmetry(fpi, symmetry="constant"):
    axis = MapAxis.from_edges([0.1, 1, 10], name="energy", unit="TeV", interp="log")
    return make_map_background_irf(
        pointing=fpi,
        ontime="42 s",
        bkg=bkg_3d_custom(symmetry),
        geom=WcsGeom.create(npix=(3, 3), binsz=4, axes=[axis], skydir=fpi.radec),
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
            "sum": 1050.930197,
        },
        {
            "map_type": "wcs",
            "ebounds": [0.1, 10],
            "shape": (1, 3, 4),
            "sum": 1050.9301972,
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
            skydir=fixed_pointing_info.radec,
        ),
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
        #  representated as [lon, lat, energy] in the api, but the bkg irf
        #  dimensions are currently [energy, lon, lat] - Will be changed in
        #  the future (perhaps when IRFs use the skymaps class)
        assert_allclose(d[1, 0], d[1, 2], rtol=1e-4)  # Symmetric along lon
        with pytest.raises(AssertionError):
            assert_allclose(d[0, 1], d[2, 1], rtol=1e-4)  # Symmetric along lat
        assert_allclose(d[0, 1] * 9, d[2, 1], rtol=1e-4)  # Asymmetric along lat


def test_make_edisp_kernel_map():
    migra = MapAxis.from_edges(np.linspace(0.5, 1.5, 50), unit="", name="migra")
    etrue = MapAxis.from_energy_bounds(0.5, 2, 6, unit="TeV", name="energy_true")
    offset = MapAxis.from_edges(np.linspace(0.0, 2.0, 3), unit="deg", name="offset")
    ereco = MapAxis.from_energy_bounds(0.5, 2, 3, unit="TeV", name="energy")

    edisp = EnergyDispersion2D.from_gauss(
        etrue.edges, migra.edges, 0, 0.01, offset.edges
    )

    geom = WcsGeom.create(10, binsz=0.5, axes=[ereco, etrue])
    pointing = SkyCoord(0, 0, frame="icrs", unit="deg")
    edispmap = make_edisp_kernel_map(edisp, pointing, geom)

    kernel = edispmap.get_edisp_kernel(pointing)
    assert_allclose(kernel.pdf_matrix[:, 0], (1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
    assert_allclose(kernel.pdf_matrix[:, 1], (0.0, 0.0, 1.0, 1.0, 0.0, 0.0))
    assert_allclose(kernel.pdf_matrix[:, 2], (0.0, 0.0, 0.0, 0.0, 1.0, 1.0))


class TestTheta2Table:
    def setup_class(self):
        table = Table()
        table["RA"] = [0.0, 0.0, 0.0, 0.0, 10.0] * u.deg
        table["DEC"] = [0.0, 0.05, 0.9, 10.0, 10.0] * u.deg
        table["ENERGY"] = [1.0, 1.0, 1.5, 1.5, 10.0] * u.TeV
        table["OFFSET"] = [0.1, 0.1, 0.5, 1.0, 1.5] * u.deg

        table.meta["RA_PNT"] = 0 * u.deg
        table.meta["DEC_PNT"] = 0.5 * u.deg

        meta_obs = dict()
        meta_obs["RA_PNT"] = 0 * u.deg
        meta_obs["DEC_PNT"] = 0.5 * u.deg
        meta_obs["DEADC"] = 1

        meta = time_ref_to_dict("2010-01-01")
        gti_table = Table({"START": [1], "STOP": [3]}, meta=meta)
        gti = GTI(gti_table)

        self.observation = Observation(
            events=EventList(table), obs_info=meta_obs, gti=gti
        )

    def test_make_theta_squared_table(self):
        # pointing position: (0,0.5) degree in ra/dec
        # On theta2 distribution compute from (0,0) in ra/dec.
        # OFF theta2 distribution from the mirror position at (0,1) in ra/dec.
        position = SkyCoord(ra=0, dec=0, unit="deg", frame="icrs")
        axis = MapAxis.from_bounds(0, 0.2, nbin=4, interp="lin", unit="deg2")
        theta2_table = make_theta_squared_table(
            observations=[self.observation], position=position, theta_squared_axis=axis
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
            observations=[self.observation],
            position=position,
            theta_squared_axis=axis,
            position_off=off_position,
        )

        assert_allclose(theta2_table2["counts_off"], theta2_table["counts"])

        # Test for two observations, here identical
        theta2_table_two_obs = make_theta_squared_table(
            observations=[self.observation, self.observation],
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
