# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
from gammapy.catalog import SourceCatalogGammaCat
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.testing import assert_quantity_allclose, requires_data

SOURCES = [
    {
        "name": "Vela X",
        "str_ref_file": "data/gammacat_vela_x.txt",
        "spec_type": "ecpl",
        "dnde_1TeV": 1.36e-11 * u.Unit("cm-2 s-1 TeV-1"),
        "dnde_1TeV_err": 7.531e-13 * u.Unit("cm-2 s-1 TeV-1"),
        "flux_1TeV": 2.104e-11 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 9.265778680255336e-11 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 24,
        "spatial_model": "GaussianSpatialModel",
        "ra": 128.287003,
        "dec": -45.189999,
    },
    {
        "name": "HESS J1848-018",
        "str_ref_file": "data/gammacat_hess_j1848-018.txt",
        "spec_type": "pl",
        "dnde_1TeV": 3.7e-12 * u.Unit("cm-2 s-1 TeV-1"),
        "dnde_1TeV_err": 4e-13 * u.Unit("cm-2 s-1 TeV-1"),
        "flux_1TeV": 2.056e-12 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 6.235650344765057e-12 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 11,
        "spatial_model": "GaussianSpatialModel",
        "ra": 282.119995,
        "dec": -1.792,
    },
    {
        "name": "HESS J1813-178",
        "str_ref_file": "data/gammacat_hess_j1813-178.txt",
        "spec_type": "pl2",
        "dnde_1TeV": 2.678e-12 * u.Unit("cm-2 s-1 TeV-1"),
        "dnde_1TeV_err": 2.55e-13 * u.Unit("cm-2 s-1 TeV-1"),
        "flux_1TeV": 2.457e-12 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 8.923614018939419e-12 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 13,
        "spatial_model": "GaussianSpatialModel",
        "ra": 273.362915,
        "dec": -17.84889,
    },
]


@pytest.fixture(scope="session")
def gammacat():
    filename = "$GAMMAPY_DATA/catalogs/gammacat/gammacat.fits.gz"
    return SourceCatalogGammaCat(filename=filename)


@requires_data()
class TestSourceCatalogGammaCat:
    def test_source_table(self, gammacat):
        assert gammacat.tag == "gamma-cat"
        assert len(gammacat.table) == 162

    def test_positions(self, gammacat):
        assert len(gammacat.positions) == 162

    def test_w28_alias_names(self, gammacat):
        for name in [
            "W28",
            "HESS J1801-233",
            "W 28",
            "SNR G6.4-0.1",
            "SNR G006.4-00.1",
            "GRO J1801-2320",
        ]:
            assert gammacat[name].row_index == 112


@requires_data()
class TestSourceCatalogObjectGammaCat:
    def test_data(self, gammacat):
        source = gammacat[0]

        assert isinstance(source.data, dict)
        assert source.data["common_name"] == "CTA 1"
        assert_quantity_allclose(source.data["dec"], 72.782997 * u.deg)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_str(self, gammacat, ref):
        actual = str(gammacat[ref["name"]])
        expected = open(get_pkg_data_filename(ref["str_ref_file"])).read()
        assert actual == expected

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_spectral_model(self, gammacat, ref):
        source = gammacat[ref["name"]]
        spectral_model = source.spectral_model()

        assert source.data["spec_type"] == ref["spec_type"]

        e_min, e_max, e_inf = [1, 10, 1e10] * u.TeV

        dne = spectral_model(e_min)
        flux = spectral_model.integral(energy_min=e_min, energy_max=e_inf)
        eflux = spectral_model.energy_flux(energy_min=e_min, energy_max=e_max).to(
            "erg cm-2 s-1"
        )

        print(spectral_model)
        assert_quantity_allclose(dne, ref["dnde_1TeV"], rtol=1e-3)
        assert_quantity_allclose(flux, ref["flux_1TeV"], rtol=1e-3)
        assert_quantity_allclose(eflux, ref["eflux_1_10TeV"], rtol=1e-3)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_spectral_model_err(self, gammacat, ref):
        source = gammacat[ref["name"]]
        spectral_model = source.spectral_model()

        e_min, e_max, e_inf = [1, 10, 1e10] * u.TeV

        dnde, dnde_err = spectral_model.evaluate_error(e_min)

        assert_quantity_allclose(dnde, ref["dnde_1TeV"], rtol=1e-3)
        assert_quantity_allclose(dnde_err, ref["dnde_1TeV_err"], rtol=1e-3)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_flux_points(self, gammacat, ref):
        source = gammacat[ref["name"]]

        flux_points = source.flux_points

        assert flux_points.energy_axis.nbin == ref["n_flux_points"]

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_position(self, gammacat, ref):
        source = gammacat[ref["name"]]

        position = source.position

        assert_allclose(position.ra.deg, ref["ra"], atol=1e-3)
        assert_allclose(position.dec.deg, ref["dec"], atol=1e-3)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_spatial_model(self, gammacat, ref):
        source = gammacat[ref["name"]]

        spatial_model = source.spatial_model()
        assert spatial_model.frame == "galactic"

        # TODO: put better asserts on model properties
        # TODO: add a point and shell source -> separate list of sources for
        # morphology test parametrization?
        assert spatial_model.__class__.__name__ == ref["spatial_model"]

        model = gammacat["HESS J1634-472"].spatial_model()
        pos_err = model.position_error
        scale_r95 = Gauss2DPDF().containment_radius(0.95)
        assert_allclose(pos_err.height.value, 2 * 0.044721 * scale_r95, rtol=1e-4)
        assert_allclose(pos_err.width.value, 2 * 0.044721 * scale_r95, rtol=1e-4)
        assert_allclose(model.position.l.value, pos_err.center.l.value)
        assert_allclose(model.position.b.value, pos_err.center.b.value)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_sky_model(self, gammacat, ref):
        gammacat[ref["name"]].sky_model()
