# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from gammapy.data import GTI
from gammapy.estimators import FluxMaps
from gammapy.maps import MapAxis, Maps, RegionGeom, TimeMapAxis, WcsNDMap
from gammapy.modeling.models import (
    LogParabolaSpectralModel,
    PointSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import mpl_plot_check


@pytest.fixture(scope="session")
def reference_model():
    return SkyModel(
        spatial_model=PointSpatialModel(), spectral_model=PowerLawSpectralModel(index=2)
    )


@pytest.fixture(scope="session")
def logpar_reference_model():
    logpar = LogParabolaSpectralModel(
        amplitude="2e-12 cm-2s-1TeV-1", alpha=1.5, beta=0.5
    )
    return SkyModel(spatial_model=PointSpatialModel(), spectral_model=logpar)


@pytest.fixture(scope="session")
def wcs_flux_map():
    energy_axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV")

    map_dict = {}

    map_dict["norm"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["norm"].data += 1.0

    map_dict["norm_err"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["norm_err"].data += 0.1

    map_dict["norm_errp"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["norm_errp"].data += 0.2

    map_dict["norm_errn"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["norm_errn"].data += 0.2

    map_dict["norm_ul"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["norm_ul"].data += 2.0

    # Add another map
    map_dict["sqrt_ts"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["sqrt_ts"].data += 1.0

    # Add another map
    map_dict["ts"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["ts"].data[1] += 3.0

    # Add another map
    map_dict["success"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit="", dtype=np.dtype(bool)
    )
    map_dict["success"].data = True
    map_dict["success"].data[0, 0, 1] = False

    return map_dict


@pytest.fixture(scope="session")
def partial_wcs_flux_map():
    energy_axis = MapAxis.from_energy_bounds(0.1, 10, 2, unit="TeV")

    map_dict = {}

    map_dict["norm"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["norm"].data += 1.0

    map_dict["norm_err"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["norm_err"].data += 0.1

    # Add another map
    map_dict["sqrt_ts"] = WcsNDMap.create(
        npix=10, frame="galactic", axes=[energy_axis], unit=""
    )
    map_dict["sqrt_ts"].data += 1.0

    return map_dict


@pytest.fixture(scope="session")
def region_map_flux_estimate():
    axis = MapAxis.from_energy_edges((0.1, 1.0, 10.0), unit="TeV")
    geom = RegionGeom.create("galactic;circle(0, 0, 0.1)", axes=[axis])

    maps = Maps.from_geom(
        geom=geom, names=["norm", "norm_err", "norm_errn", "norm_errp", "norm_ul"]
    )

    maps["norm"].data = np.array([1.0, 1.0])
    maps["norm_err"].data = np.array([0.1, 0.1])
    maps["norm_errn"].data = np.array([0.2, 0.2])
    maps["norm_errp"].data = np.array([0.15, 0.15])
    maps["norm_ul"].data = np.array([2.0, 2.0])
    return maps


@pytest.fixture(scope="session")
def map_flux_estimate():
    axis = MapAxis.from_energy_edges((0.1, 1.0, 10.0), unit="TeV")

    nmap = WcsNDMap.create(npix=5, axes=[axis])

    cols = dict()
    cols["norm"] = nmap.copy(data=1.0)
    cols["norm_err"] = nmap.copy(data=0.1)
    cols["norm_errn"] = nmap.copy(data=0.2)
    cols["norm_errp"] = nmap.copy(data=0.15)
    cols["norm_ul"] = nmap.copy(data=2.0)

    return cols


def test_table_properties(region_map_flux_estimate):
    model = SkyModel(PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2))

    fe = FluxMaps(data=region_map_flux_estimate, reference_model=model)

    assert fe.dnde.unit == u.Unit("cm-2s-1TeV-1")
    assert_allclose(fe.dnde.data.flat, [1e-9, 1e-11])
    assert_allclose(fe.dnde_err.data.flat, [1e-10, 1e-12])
    assert_allclose(fe.dnde_errn.data.flat, [2e-10, 2e-12])
    assert_allclose(fe.dnde_errp.data.flat, [1.5e-10, 1.5e-12])
    assert_allclose(fe.dnde_ul.data.flat, [2e-9, 2e-11])

    assert fe.e2dnde.unit == u.Unit("TeV cm-2s-1")
    assert_allclose(fe.e2dnde.data.flat, [1e-10, 1e-10])

    assert fe.flux.unit == u.Unit("cm-2s-1")
    assert_allclose(fe.flux.data.flat, [9e-10, 9e-11])

    assert fe.eflux.unit == u.Unit("TeV cm-2s-1")
    assert_allclose(fe.eflux.data.flat, [2.302585e-10, 2.302585e-10])


def test_missing_column(region_map_flux_estimate):
    del region_map_flux_estimate["norm_errn"]
    model = SkyModel(PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2))
    fe = FluxMaps(data=region_map_flux_estimate, reference_model=model)

    with pytest.raises(AttributeError):
        fe.dnde_errn


def test_map_properties(map_flux_estimate):
    model = SkyModel(PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2))
    fe = FluxMaps(data=map_flux_estimate, reference_model=model)

    assert fe.dnde.unit == u.Unit("cm-2s-1TeV-1")
    assert_allclose(fe.dnde.quantity.value[:, 2, 2], [1e-9, 1e-11])
    assert_allclose(fe.dnde_err.quantity.value[:, 2, 2], [1e-10, 1e-12])
    assert_allclose(fe.dnde_errn.quantity.value[:, 2, 2], [2e-10, 2e-12])
    assert_allclose(fe.dnde_errp.quantity.value[:, 2, 2], [1.5e-10, 1.5e-12])
    assert_allclose(fe.dnde_ul.quantity.value[:, 2, 2], [2e-9, 2e-11])

    assert fe.e2dnde.unit == u.Unit("TeV cm-2s-1")
    assert_allclose(fe.e2dnde.quantity.value[:, 2, 2], [1e-10, 1e-10])
    assert_allclose(fe.e2dnde_err.quantity.value[:, 2, 2], [1e-11, 1e-11])
    assert_allclose(fe.e2dnde_errn.quantity.value[:, 2, 2], [2e-11, 2e-11])
    assert_allclose(fe.e2dnde_errp.quantity.value[:, 2, 2], [1.5e-11, 1.5e-11])
    assert_allclose(fe.e2dnde_ul.quantity.value[:, 2, 2], [2e-10, 2e-10])

    assert fe.flux.unit == u.Unit("cm-2s-1")
    assert_allclose(fe.flux.quantity.value[:, 2, 2], [9e-10, 9e-11])
    assert_allclose(fe.flux_err.quantity.value[:, 2, 2], [9e-11, 9e-12])
    assert_allclose(fe.flux_errn.quantity.value[:, 2, 2], [1.8e-10, 1.8e-11])
    assert_allclose(fe.flux_errp.quantity.value[:, 2, 2], [1.35e-10, 1.35e-11])
    assert_allclose(fe.flux_ul.quantity.value[:, 2, 2], [1.8e-9, 1.8e-10])

    assert fe.eflux.unit == u.Unit("TeV cm-2s-1")
    assert_allclose(fe.eflux.quantity.value[:, 2, 2], [2.302585e-10, 2.302585e-10])
    assert_allclose(fe.eflux_err.quantity.value[:, 2, 2], [2.302585e-11, 2.302585e-11])
    assert_allclose(fe.eflux_errn.quantity.value[:, 2, 2], [4.60517e-11, 4.60517e-11])
    assert_allclose(
        fe.eflux_errp.quantity.value[:, 2, 2], [3.4538775e-11, 3.4538775e-11]
    )
    assert_allclose(fe.eflux_ul.quantity.value[:, 2, 2], [4.60517e-10, 4.60517e-10])


def test_flux_map_properties(wcs_flux_map, reference_model):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)

    assert_allclose(fluxmap.dnde.data[:, 0, 0], [1e-11, 1e-13])
    assert_allclose(fluxmap.dnde_err.data[:, 0, 0], [1e-12, 1e-14])
    assert_allclose(fluxmap.dnde_err.data[:, 0, 0], [1e-12, 1e-14])
    assert_allclose(fluxmap.dnde_errn.data[:, 0, 0], [2e-12, 2e-14])
    assert_allclose(fluxmap.dnde_errp.data[:, 0, 0], [2e-12, 2e-14])
    assert_allclose(fluxmap.dnde_ul.data[:, 0, 0], [2e-11, 2e-13])

    assert_allclose(fluxmap.flux.data[:, 0, 0], [9e-12, 9e-13])
    assert_allclose(fluxmap.flux_err.data[:, 0, 0], [9e-13, 9e-14])
    assert_allclose(fluxmap.flux_errn.data[:, 0, 0], [18e-13, 18e-14])
    assert_allclose(fluxmap.flux_errp.data[:, 0, 0], [18e-13, 18e-14])
    assert_allclose(fluxmap.flux_ul.data[:, 0, 0], [18e-12, 18e-13])

    assert_allclose(fluxmap.eflux.data[:, 0, 0], [2.302585e-12, 2.302585e-12])
    assert_allclose(fluxmap.eflux_err.data[:, 0, 0], [2.302585e-13, 2.302585e-13])
    assert_allclose(fluxmap.eflux_errp.data[:, 0, 0], [4.60517e-13, 4.60517e-13])
    assert_allclose(fluxmap.eflux_errn.data[:, 0, 0], [4.60517e-13, 4.60517e-13])
    assert_allclose(fluxmap.eflux_ul.data[:, 0, 0], [4.60517e-12, 4.60517e-12])

    assert_allclose(fluxmap.e2dnde.data[:, 0, 0], [1e-12, 1e-12])
    assert_allclose(fluxmap.e2dnde_err.data[:, 0, 0], [1e-13, 1e-13])
    assert_allclose(fluxmap.e2dnde_errn.data[:, 0, 0], [2e-13, 2e-13])
    assert_allclose(fluxmap.e2dnde_errp.data[:, 0, 0], [2e-13, 2e-13])
    assert_allclose(fluxmap.e2dnde_ul.data[:, 0, 0], [2e-12, 2e-12])

    assert_allclose(fluxmap.sqrt_ts.data, 1)
    assert_allclose(fluxmap.ts.data[:, 0, 0], [0, 3])

    assert_allclose(fluxmap.success.data[:, 0, 1], [False, True])
    assert_allclose(fluxmap.flux.data[:, 0, 1], [np.nan, 9e-13])
    assert_allclose(fluxmap.flux_err.data[:, 0, 1], [np.nan, 9e-14])

    assert_allclose(fluxmap.eflux.data[:, 0, 1], [np.nan, 2.30258509e-12])
    assert_allclose(fluxmap.e2dnde_err.data[:, 0, 1], [np.nan, 1e-13])


def test_flux_map_failed_properties(wcs_flux_map, reference_model):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)
    fluxmap.filter_success_nan = False

    assert_allclose(fluxmap.success.data[:, 0, 1], [False, True])
    assert_allclose(fluxmap.flux.data[:, 0, 1], [9.0e-12, 9e-13])
    assert not fluxmap.filter_success_nan


def test_flux_map_str(wcs_flux_map, reference_model):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)

    fm_str = fluxmap.__str__()

    assert "WcsGeom" in fm_str
    assert "errn" in fm_str
    assert "sqrt_ts" in fm_str
    assert "pl" in fm_str
    assert "n_sigma" in fm_str
    assert "n_sigma_ul" in fm_str
    assert "sqrt_ts_threshold" in fm_str


@pytest.mark.parametrize("sed_type", ["likelihood", "dnde", "flux", "eflux", "e2dnde"])
def test_flux_map_read_write(tmp_path, wcs_flux_map, logpar_reference_model, sed_type):
    fluxmap = FluxMaps(wcs_flux_map, logpar_reference_model)

    fluxmap.write(tmp_path / "tmp.fits", sed_type=sed_type, overwrite=True)
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert_allclose(new_fluxmap.norm.data[:, 0, 0], [1, 1])
    assert_allclose(new_fluxmap.norm_err.data[:, 0, 0], [0.1, 0.1])
    assert_allclose(new_fluxmap.norm_errn.data[:, 0, 0], [0.2, 0.2])
    assert_allclose(new_fluxmap.norm_ul.data[:, 0, 0], [2, 2])

    # check model
    assert (
        new_fluxmap.reference_model.spectral_model.tag[0] == "LogParabolaSpectralModel"
    )
    assert new_fluxmap.reference_model.spectral_model.alpha.value == 1.5
    assert new_fluxmap.reference_model.spectral_model.beta.value == 0.5
    assert new_fluxmap.reference_model.spectral_model.amplitude.value == 2e-12

    # check existence and content of additional map
    assert_allclose(new_fluxmap.sqrt_ts.data, 1.0)
    assert_allclose(new_fluxmap.success.data[:, 0, 1], [False, True])
    assert_allclose(new_fluxmap.is_ul.data, True)


@pytest.mark.parametrize("sed_type", ["likelihood", "dnde", "flux", "eflux", "e2dnde"])
def test_partial_flux_map_read_write(
    tmp_path, partial_wcs_flux_map, reference_model, sed_type
):
    fluxmap = FluxMaps(partial_wcs_flux_map, reference_model)

    fluxmap.write(tmp_path / "tmp.fits", sed_type=sed_type, overwrite=True)
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert_allclose(new_fluxmap.norm.data[:, 0, 0], [1, 1])
    assert_allclose(new_fluxmap.norm_err.data[:, 0, 0], [0.1, 0.1])

    # check model
    assert new_fluxmap.reference_model.spectral_model.tag[0] == "PowerLawSpectralModel"
    assert new_fluxmap.reference_model.spectral_model.index.value == 2

    # check existence and content of additional map
    assert_allclose(new_fluxmap._data["sqrt_ts"].data, 1.0)

    # the TS map shouldn't exist
    with pytest.raises(AttributeError):
        new_fluxmap.ts


def test_flux_map_read_write_gti(tmp_path, partial_wcs_flux_map, reference_model):
    start = u.Quantity([1, 2], "min")
    stop = u.Quantity([1.5, 2.5], "min")
    gti = GTI.create(start, stop)

    fluxmap = FluxMaps(partial_wcs_flux_map, reference_model, gti=gti)

    fluxmap.write(tmp_path / "tmp.fits", sed_type="dnde")
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert len(new_fluxmap.gti.table) == 2
    assert_allclose(gti.table["START"], start.to_value("s"))


@pytest.mark.xfail
def test_flux_map_read_write_no_reference_model(tmp_path, wcs_flux_map, caplog):
    fluxmap = FluxMaps(wcs_flux_map)

    fluxmap.write(tmp_path / "tmp.fits")
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert new_fluxmap.reference_model.spectral_model.tag[0] == "PowerLawSpectralModel"
    assert "WARNING" in [_.levelname for _ in caplog.records]
    assert "No reference model set for FluxMaps." in [_.message for _ in caplog.records]


def test_flux_map_read_write_missing_reference_model(
    tmp_path, wcs_flux_map, reference_model
):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)
    fluxmap.write(tmp_path / "tmp.fits")

    hdulist = fits.open(tmp_path / "tmp.fits")
    hdulist[0].header["MODEL"] = "non_existent"

    with pytest.raises(FileNotFoundError):
        _ = FluxMaps.from_hdulist(hdulist)


@pytest.mark.xfail
def test_flux_map_init_no_reference_model(wcs_flux_map, caplog):
    fluxmap = FluxMaps(data=wcs_flux_map)

    assert fluxmap.reference_model.spectral_model.tag[0] == "PowerLawSpectralModel"
    assert fluxmap.reference_model.spatial_model.tag[0] == "PointSpatialModel"
    assert fluxmap.reference_model.spectral_model.index.value == 2

    assert "WARNING" in [_.levelname for _ in caplog.records]
    assert "No reference model set for FluxMaps." in [_.message for _ in caplog.records]


def test_get_flux_point(wcs_flux_map, reference_model):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)
    coord = SkyCoord(0.0, 0.0, unit="deg", frame="galactic")
    fp = fluxmap.get_flux_points(coord)
    table = fp.to_table()

    assert_allclose(table["e_min"], [0.1, 1.0])
    assert_allclose(table["norm"], [1, 1])
    assert_allclose(table["norm_err"], [0.1, 0.1])
    assert_allclose(table["norm_errn"], [0.2, 0.2])
    assert_allclose(table["norm_errp"], [0.2, 0.2])
    assert_allclose(table["norm_ul"], [2, 2])
    assert_allclose(table["sqrt_ts"], [1, 1])
    assert_allclose(table["ts"], [0, 3], atol=1e-15)

    assert_allclose(fp.dnde.data.flat, [1e-11, 1e-13])
    assert fp.dnde.unit == "cm-2s-1TeV-1"

    with mpl_plot_check():
        fp.plot()


def test_get_flux_point_missing_map(wcs_flux_map, reference_model):
    other_data = wcs_flux_map.copy()
    other_data.pop("norm_errn")
    other_data.pop("norm_errp")
    fluxmap = FluxMaps(other_data, reference_model)

    coord = SkyCoord(0.0, 0.0, unit="deg", frame="galactic")
    table = fluxmap.get_flux_points(coord).to_table()
    assert_allclose(table["e_min"], [0.1, 1.0])
    assert_allclose(table["norm"], [1, 1])
    assert_allclose(table["norm_err"], [0.1, 0.1])
    assert_allclose(table["norm_ul"], [2, 2])
    assert "norm_errn" not in table.columns
    assert table["success"].data.dtype == np.dtype(np.bool)


def test_flux_map_from_dict_inconsistent_units(wcs_flux_map, reference_model):
    ref_map = FluxMaps(wcs_flux_map, reference_model)
    map_dict = dict()
    map_dict["eflux"] = ref_map.eflux
    map_dict["eflux"].quantity = map_dict["eflux"].quantity.to("keV/m2/s")
    map_dict["eflux_err"] = ref_map.eflux_err
    map_dict["eflux_err"].quantity = map_dict["eflux_err"].quantity.to("keV/m2/s")

    flux_map = FluxMaps.from_maps(map_dict, "eflux", reference_model)

    assert_allclose(flux_map.norm.data[:, 0, 0], 1.0)
    assert flux_map.norm.unit == ""
    assert_allclose(flux_map.norm_err.data[:, 0, 0], 0.1)
    assert flux_map.norm_err.unit == ""


def test_flux_map_iter_by_axis():
    axis1 = MapAxis.from_energy_edges((0.1, 1.0, 10.0), unit="TeV")
    axis2 = TimeMapAxis.from_time_bounds(
        Time(51544, format="mjd"), Time(51548, format="mjd"), 3
    )
    geom = RegionGeom.create("galactic;circle(0, 0, 0.1)", axes=[axis1, axis2])

    maps = Maps.from_geom(
        geom=geom, names=["norm", "norm_err", "norm_errn", "norm_errp", "norm_ul"]
    )
    val = np.ones(geom.data_shape)

    maps["norm"].data = val
    maps["norm_err"].data = 0.1 * val
    maps["norm_errn"].data = 0.2 * val
    maps["norm_errp"].data = 0.15 * val
    maps["norm_ul"].data = 2.0 * val

    start = u.Quantity([1, 2, 3], "day")
    stop = u.Quantity([1.5, 2.5, 3.9], "day")
    gti = GTI.create(start, stop)
    ref_map = FluxMaps(maps, gti=gti, reference_model=PowerLawSpectralModel())

    split_maps = list(ref_map.iter_by_axis("time"))
    assert len(split_maps) == 3
    assert split_maps[0].available_quantities == ref_map.available_quantities
    assert_allclose(split_maps[0].gti.time_stop.value, 51545.3340, rtol=1e-3)

    assert split_maps[0].reference_model == ref_map.reference_model
