# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from gammapy.data import GTI
from gammapy.maps import MapAxis, WcsNDMap
from gammapy.modeling.models import SkyModel, PowerLawSpectralModel, PointSpatialModel, LogParabolaSpectralModel
from gammapy.estimators import FluxMaps
from gammapy.utils.testing import mpl_plot_check, requires_dependency


@pytest.fixture(scope="session")
def reference_model():
    return SkyModel(spatial_model=PointSpatialModel(), spectral_model=PowerLawSpectralModel(index=2))


@pytest.fixture(scope="session")
def logpar_reference_model():
    logpar = LogParabolaSpectralModel(amplitude="2e-12 cm-2s-1TeV-1", alpha=1.5, beta=0.5)
    return SkyModel(spatial_model=PointSpatialModel(), spectral_model=logpar)


@pytest.fixture(scope="session")
def wcs_flux_map():
    energy_axis = MapAxis.from_energy_bounds(0.1,10, 2, unit='TeV')

    map_dict = {}

    map_dict["norm"]= WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm"].data += 1.0

    map_dict["norm_err"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_err"].data += 0.1

    map_dict["norm_errp"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_errp"].data += 0.2

    map_dict["norm_errn"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_errn"].data += 0.2

    map_dict["norm_ul"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_ul"].data += 2.0

    # Add another map
    map_dict["sqrt_ts"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["sqrt_ts"].data += 1.0

    # Add another map
    map_dict["ts"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["ts"].data[1] += 3.0

    return map_dict


@pytest.fixture(scope="session")
def partial_wcs_flux_map():
    energy_axis = MapAxis.from_energy_bounds(0.1,10, 2, unit='TeV')

    map_dict = {}

    map_dict["norm"]= WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm"].data += 1.0

    map_dict["norm_err"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["norm_err"].data += 0.1

    # Add another map
    map_dict["sqrt_ts"] = WcsNDMap.create(npix=10, frame='galactic', axes=[energy_axis], unit='')
    map_dict["sqrt_ts"].data += 1.0

    return map_dict


def test_flux_map_properties(wcs_flux_map, reference_model):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)

    assert_allclose(fluxmap.dnde.data[:,0,0],[1e-11, 1e-13])
    assert_allclose(fluxmap.dnde_err.data[:,0,0],[1e-12, 1e-14])
    assert_allclose(fluxmap.dnde_err.data[:,0,0],[1e-12, 1e-14])
    assert_allclose(fluxmap.dnde_errn.data[:,0,0],[2e-12, 2e-14])
    assert_allclose(fluxmap.dnde_errp.data[:,0,0],[2e-12, 2e-14])
    assert_allclose(fluxmap.dnde_ul.data[:,0,0],[2e-11, 2e-13])

    assert_allclose(fluxmap.flux.data[:,0,0],[9e-12, 9e-13])
    assert_allclose(fluxmap.flux_err.data[:,0,0],[9e-13, 9e-14])
    assert_allclose(fluxmap.flux_errn.data[:,0,0],[18e-13, 18e-14])
    assert_allclose(fluxmap.flux_errp.data[:,0,0],[18e-13, 18e-14])
    assert_allclose(fluxmap.flux_ul.data[:,0,0],[18e-12, 18e-13])

    assert_allclose(fluxmap.eflux.data[:,0,0],[2.302585e-12, 2.302585e-12])
    assert_allclose(fluxmap.eflux_err.data[:,0,0],[2.302585e-13, 2.302585e-13])
    assert_allclose(fluxmap.eflux_errp.data[:,0,0],[4.60517e-13, 4.60517e-13])
    assert_allclose(fluxmap.eflux_errn.data[:,0,0],[4.60517e-13, 4.60517e-13])
    assert_allclose(fluxmap.eflux_ul.data[:,0,0],[4.60517e-12, 4.60517e-12])

    assert_allclose(fluxmap.e2dnde.data[:, 0, 0], [1e-12, 1e-12])
    assert_allclose(fluxmap.e2dnde_err.data[:, 0, 0], [1e-13, 1e-13])
    assert_allclose(fluxmap.e2dnde_errn.data[:, 0, 0], [2e-13, 2e-13])
    assert_allclose(fluxmap.e2dnde_errp.data[:, 0, 0], [2e-13, 2e-13])
    assert_allclose(fluxmap.e2dnde_ul.data[:, 0, 0], [2e-12, 2e-12])

    assert_allclose(fluxmap.sqrt_ts.data, 1)
    assert_allclose(fluxmap.ts.data[:,0,0], [0, 3])


def test_flux_map_str(wcs_flux_map, reference_model):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)

    fm_str = fluxmap.__str__()

    assert "WcsGeom" in fm_str
    assert "errn" in fm_str
    assert "sqrt_ts" in fm_str
    assert "pl" in fm_str
    assert "n_sigma" in fm_str
    assert "n_sigma_ul" in fm_str
    assert "ts_threshold" in fm_str


@pytest.mark.parametrize("sed_type", ["likelihood", "dnde", "flux", "eflux", "e2dnde"])
def test_flux_map_read_write(tmp_path, wcs_flux_map, logpar_reference_model, sed_type):
    fluxmap = FluxMaps(wcs_flux_map, logpar_reference_model)

    fluxmap.write(tmp_path / "tmp.fits", sed_type=sed_type)
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert_allclose(new_fluxmap.norm.data[:,0,0], [1, 1])
    assert_allclose(new_fluxmap.norm_err.data[:,0,0], [0.1, 0.1])
    assert_allclose(new_fluxmap.norm_errn.data[:,0,0], [0.2, 0.2])
    assert_allclose(new_fluxmap.norm_ul.data[:,0,0], [2, 2])

    # check model
    assert new_fluxmap.reference_model.spectral_model.tag[0] == "LogParabolaSpectralModel"
    assert new_fluxmap.reference_model.spectral_model.alpha.value == 1.5
    assert new_fluxmap.reference_model.spectral_model.beta.value == 0.5
    assert new_fluxmap.reference_model.spectral_model.amplitude.value == 2e-12

    # check existence and content of additional map
    assert_allclose(new_fluxmap.sqrt_ts.data, 1.0)


@pytest.mark.parametrize("sed_type", ["likelihood", "dnde", "flux", "eflux", "e2dnde"])
def test_partial_flux_map_read_write(tmp_path, partial_wcs_flux_map, reference_model, sed_type):
    fluxmap = FluxMaps(partial_wcs_flux_map, reference_model)

    fluxmap.write(tmp_path / "tmp.fits", sed_type=sed_type)
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert_allclose(new_fluxmap.norm.data[:,0,0], [1, 1])
    assert_allclose(new_fluxmap.norm_err.data[:,0,0], [0.1, 0.1])

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

    fluxmap = FluxMaps(partial_wcs_flux_map, reference_model, gti)

    fluxmap.write(tmp_path / "tmp.fits", sed_type='dnde')
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert len(new_fluxmap.gti.table) == 2
    assert_allclose(gti.table["START"], start.to_value("s"))


@pytest.mark.xfail
def test_flux_map_read_write_no_reference_model(tmp_path, wcs_flux_map, caplog):
    fluxmap = FluxMaps(wcs_flux_map)

    fluxmap.write(tmp_path / "tmp.fits")
    new_fluxmap = FluxMaps.read(tmp_path / "tmp.fits")

    assert new_fluxmap.reference_model.spectral_model.tag[0] == "PowerLawSpectralModel"
    assert caplog.records[-1].levelname == "WARNING"
    assert f"No reference model set for FluxMaps." in caplog.records[-1].message


def test_flux_map_read_write_missing_reference_model(tmp_path, wcs_flux_map, reference_model):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)
    fluxmap.write(tmp_path / "tmp.fits")

    hdulist = fits.open(tmp_path / "tmp.fits")
    hdulist[0].header["MODEL"] = "non_existent"

    with pytest.raises(FileNotFoundError):
        new_fluxmap = FluxMaps.from_hdulist(hdulist)


@pytest.mark.xfail
def test_flux_map_init_no_reference_model(wcs_flux_map, caplog):
    fluxmap = FluxMaps(wcs_flux_map)

    assert fluxmap.reference_model.spectral_model.tag[0] == "PowerLawSpectralModel"
    assert fluxmap.reference_model.spatial_model.tag[0] == "PointSpatialModel"
    assert fluxmap.reference_model.spectral_model.index.value == 2

    assert caplog.records[-1].levelname == "WARNING"
    assert f"No reference model set for FluxMaps." in caplog.records[-1].message


@requires_dependency("matplotlib")
def test_get_flux_point(wcs_flux_map, reference_model):
    fluxmap = FluxMaps(wcs_flux_map, reference_model)

    coord = SkyCoord(0., 0., unit="deg", frame="galactic")
    fp = fluxmap.get_flux_points(coord)
    table = fp.to_table()

    assert_allclose(table["e_min"], [0.1, 1.0])
    assert_allclose(table["norm"], [1, 1] )
    assert_allclose(table["norm_err"], [0.1, 0.1] )
    assert_allclose(table["norm_errn"], [0.2, 0.2] )
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

    coord = SkyCoord(0., 0., unit="deg", frame="galactic")
    table = fluxmap.get_flux_points(coord).to_table()

    assert_allclose(table["e_min"], [0.1, 1.0])
    assert_allclose(table["norm"], [1, 1])
    assert_allclose(table["norm_err"], [0.1, 0.1])
    assert_allclose(table["norm_ul"], [2, 2])
    assert "norm_errn" not in table.columns


def test_flux_map_from_dict_inconsistent_units(wcs_flux_map, reference_model):
    ref_map = FluxMaps(wcs_flux_map, reference_model)
    map_dict = dict()
    map_dict["eflux"] = ref_map.eflux
    map_dict["eflux"].quantity = map_dict["eflux"].quantity.to("keV/m2/s")
    map_dict["eflux_err"] = ref_map.eflux_err
    map_dict["eflux_err"].quantity = map_dict["eflux_err"].quantity.to("keV/m2/s")

    flux_map = FluxMaps.from_dict(map_dict, "eflux", reference_model)

    assert_allclose(flux_map.norm.data, 1)
    assert flux_map.norm.unit == ""
    assert_allclose(flux_map.norm_err.data, 0.1)
    assert flux_map.norm_err.unit == ""

