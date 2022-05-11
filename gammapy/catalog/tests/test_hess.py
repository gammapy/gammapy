# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
from gammapy.catalog import SourceCatalogHGPS, SourceCatalogLargeScaleHGPS
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    PowerLawSpectralModel,
)
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.testing import assert_quantity_allclose, requires_data

SOURCES = [
    {"idx": 33, "name": "HESS J1713-397", "str_ref_file": "data/hess_j1713-397.txt"},
    {"idx": 54, "name": "HESS J1825-137", "str_ref_file": "data/hess_j1825-137.txt"},
    {"idx": 76, "name": "HESS J1930+188", "str_ref_file": "data/hess_j1930+188.txt"},
]


@pytest.fixture(scope="session")
def cat():
    return SourceCatalogHGPS("$GAMMAPY_DATA/catalogs/hgps_catalog_v1.fits.gz")


@requires_data()
class TestSourceCatalogHGPS:
    @staticmethod
    def test_source_table(cat):
        assert cat.tag == "hgps"
        assert len(cat.table) == 78

    @staticmethod
    def test_str(cat):
        assert "name: hgps" in str(cat.__str__())

    @staticmethod
    def test_positions(cat):
        assert len(cat.positions) == 78

    @staticmethod
    def test_table_components(cat):
        assert len(cat.table_components) == 98

    @staticmethod
    def test_table_associations(cat):
        assert len(cat.table_associations) == 223

    @staticmethod
    def test_table_identifications(cat):
        assert len(cat.table_identifications) == 31

    @staticmethod
    def test_gaussian_component(cat):
        # Row index starts at 0, component numbers at 1
        # Thus we expect `HGPSC 084` at row 83
        c = cat.gaussian_component(83)
        assert c.name == "HGPSC 084"

    @staticmethod
    def test_large_scale_component(cat):
        assert isinstance(cat.large_scale_component, SourceCatalogLargeScaleHGPS)

    @staticmethod
    def test_to_models(cat):
        models = cat.to_models(components_status="independent")
        assert len(models) == 96
        models = cat.to_models(components_status="linked")
        assert len(models) == 96
        models = cat.to_models(components_status="merged")
        assert len(models) == 78

        gaussians = models.select(tag="gauss", model_type="spatial")
        assert np.all([m.spatial_model.sigma.value > 0.0 for m in gaussians])


@requires_data()
class TestSourceCatalogObjectHGPS:
    @pytest.fixture(scope="class")
    def source(self, cat):
        return cat["HESS J1843-033"]

    @staticmethod
    def test_basics(source):
        assert source.name == "HESS J1843-033"
        assert source.row_index == 64
        data = source.data
        assert data["Source_Class"] == "Unid"
        assert "SourceCatalogObjectHGPS" in repr(source)

        ss = str(source)
        assert "Source name          : HESS J1843-033" in ss
        assert "Component HGPSC 083:" in ss

    @staticmethod
    @pytest.mark.parametrize("ref", SOURCES)
    def test_str(cat, ref):
        actual = str(cat[ref["idx"]])
        expected = open(get_pkg_data_filename(ref["str_ref_file"])).read()
        assert actual == expected

    @staticmethod
    def test_position(source):
        position = source.position
        assert_allclose(position.ra.deg, 280.95162964)
        assert_allclose(position.dec.deg, -3.55410194)

    @staticmethod
    def test_components(source):
        components = source.components
        assert len(components) == 2
        c = components[1]
        assert c.name == "HGPSC 084"

    @staticmethod
    def test_energy_range(source):
        energy_range = source.energy_range
        assert energy_range.unit == "TeV"
        assert_allclose(energy_range.value, [0.21544346, 61.89658356])

    @staticmethod
    def test_spectral_model_pl(cat):
        source = cat["HESS J1843-033"]

        model = source.spectral_model()

        assert isinstance(model, PowerLawSpectralModel)
        pars = model.parameters
        assert_allclose(pars["amplitude"].value, 9.140179932365378e-13)
        assert_allclose(pars["index"].value, 2.1513476371765137)
        assert_allclose(pars["reference"].value, 1.867810606956482)

    @staticmethod
    def test_spectral_model_ecpl(cat):
        source = cat["HESS J0835-455"]

        model = source.spectral_model()
        assert isinstance(model, ExpCutoffPowerLawSpectralModel)

        pars = model.parameters
        assert_allclose(pars["amplitude"].value, 6.408420542586617e-12)
        assert_allclose(pars["index"].value, 1.3543991614920847)
        assert_allclose(pars["reference"].value, 1.696938754239)
        assert_allclose(pars["lambda_"].value, 0.081517637)

        assert_allclose(pars["amplitude"].error, 3.260472e-13, rtol=1e-3)
        assert_allclose(pars["index"].error, 0.077331, atol=0.001)
        assert_allclose(pars["reference"].error, 0)
        assert_allclose(pars["lambda_"].error, 0.011535, atol=0.001)

        model = source.spectral_model("pl")
        assert isinstance(model, PowerLawSpectralModel)

        pars = model.parameters
        assert_allclose(pars["amplitude"].value, 1.833056926733856e-12)
        assert_allclose(pars["index"].value, 1.8913707)
        assert_allclose(pars["reference"].value, 3.0176312923431396)

        assert_allclose(pars["amplitude"].error, 6.992061e-14, rtol=1e-3)
        assert_allclose(pars["index"].error, 0.028383, atol=0.001)
        assert_allclose(pars["reference"].error, 0)

    @staticmethod
    def test_position_error(cat):
        scale_r95 = Gauss2DPDF().containment_radius(0.95)

        model = cat["HESS J1729-345"].spatial_model()
        pos_err = model.position_error
        assert_allclose(pos_err.angle.value, 0.0)
        assert_allclose(pos_err.height.value, 2 * 0.0414315 * scale_r95, rtol=1e-4)
        assert_allclose(pos_err.width.value, 2 * 0.0344351 * scale_r95, rtol=1e-4)
        assert_allclose(model.position.l.value, pos_err.center.l.value)
        assert_allclose(model.position.b.value, pos_err.center.b.value)

        model = cat["HESS J1858+020"].spatial_model()
        pos_err = model.position_error
        assert_allclose(pos_err.angle.value, 90.0)
        assert_allclose(pos_err.height.value, 2 * 0.0222614 * scale_r95, rtol=1e-4)
        assert_allclose(pos_err.width.value, 2 * 0.0145084 * scale_r95, rtol=1e-4)
        assert_allclose(model.position.l.value, pos_err.center.l.value)
        assert_allclose(model.position.b.value, pos_err.center.b.value)

    @staticmethod
    def test_sky_model_point(cat):
        model = cat["HESS J1826-148"].sky_model()
        p = model.parameters
        assert_allclose(p["amplitude"].value, 9.815771242691063e-13)
        assert_allclose(p["lon_0"].value, 16.882482528686523)
        assert_allclose(p["lat_0"].value, -1.2889292240142822)

    @staticmethod
    def test_sky_model_gaussian(cat):
        model = cat["HESS J1119-614"].sky_model()
        p = model.parameters
        assert_allclose(p["amplitude"].value, 7.959899015960725e-13)
        assert_allclose(p["lon_0"].value, 292.128082)
        assert_allclose(p["lat_0"].value, -0.5332353711128235)
        assert_allclose(p["sigma"].value, 0.09785966575145721)

    @staticmethod
    def test_sky_model_gaussian2(cat):
        models = cat["HESS J1843-033"].components_models()

        p = models[0].parameters
        assert_allclose(p["amplitude"].value, 4.259815e-13, rtol=1e-5)
        assert_allclose(p["lon_0"].value, 29.047216415405273)
        assert_allclose(p["lat_0"].value, 0.24389676749706268)
        assert_allclose(p["sigma"].value, 0.12499100714921951)

        p = models[1].parameters
        assert_allclose(p["amplitude"].value, 4.880365e-13, rtol=1e-5)
        assert_allclose(p["lon_0"].value, 28.77037811279297)
        assert_allclose(p["lat_0"].value, -0.0727819949388504)
        assert_allclose(p["sigma"].value, 0.2294706553220749)

    @staticmethod
    def test_sky_model_gaussian3(cat):
        models = cat["HESS J1825-137"].components_models()

        p = models[0].parameters
        assert_allclose(p["amplitude"].value, 1.8952104218765842e-11)
        assert_allclose(p["lon_0"].value, 16.988601684570312)
        assert_allclose(p["lat_0"].value, -0.4913068115711212)
        assert_allclose(p["sigma"].value, 0.47650089859962463)

        p = models[1].parameters
        assert_allclose(p["amplitude"].value, 4.4639763971527836e-11)
        assert_allclose(p["lon_0"].value, 17.71169090270996)
        assert_allclose(p["lat_0"].value, -0.6598004102706909)
        assert_allclose(p["sigma"].value, 0.3910967707633972)

        p = models[2].parameters
        assert_allclose(p["amplitude"].value, 5.870712920658374e-12)
        assert_allclose(p["lon_0"].value, 17.840524673461914)
        assert_allclose(p["lat_0"].value, -0.7057178020477295)
        assert_allclose(p["sigma"].value, 0.10932201147079468)

    @staticmethod
    def test_sky_model_gaussian_extern(cat):
        # special test for the only extern source with a gaussian morphology
        model = cat["HESS J1801-233"].components_models()
        p = model.parameters
        assert_allclose(p["amplitude"].value, 7.499999970031479e-13)
        assert_allclose(p["lon_0"].value, 6.656888961791992)
        assert_allclose(p["lat_0"].value, -0.267688125371933)
        assert_allclose(p["sigma"].value, 0.17)

    @staticmethod
    def test_sky_model_shell(cat):
        model = cat["Vela Junior"].sky_model()
        p = model.parameters
        assert_allclose(p["amplitude"].value, 3.2163001428830995e-11)
        assert_allclose(p["lon_0"].value, 266.287384)
        assert_allclose(p["lat_0"].value, -1.243260383605957)
        assert_allclose(p["radius"].value, 0.95)
        assert_allclose(p["width"].value, 0.05)

    @staticmethod
    def test_flux_points_meta(cat):
        source = cat["HESS J1843-033"]
        fp = source.flux_points

        assert fp.sqrt_ts_threshold_ul == 1
        assert fp.n_sigma == 1
        assert fp.n_sigma_ul == 2


@requires_data()
class TestSourceCatalogObjectHGPSComponent:
    @pytest.fixture(scope="class")
    def component(self, cat):
        return cat.gaussian_component(83)

    @staticmethod
    def test_repr(component):
        assert "SourceCatalogObjectHGPSComponent" in repr(component)

    @staticmethod
    def test_str(component):
        assert "Component HGPSC 084" in str(component)

    @staticmethod
    def test_name(component):
        assert component.name == "HGPSC 084"

    @staticmethod
    def test_index(component):
        assert component.row_index == 83

    @staticmethod
    def test_spatial_model(component):
        model = component.spatial_model()
        assert model.frame == "galactic"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 28.77037811279297)
        assert_allclose(p["lon_0"].error, 0.058748625218868256)
        assert_allclose(p["lat_0"].value, -0.0727819949388504)
        assert_allclose(p["lat_0"].error, 0.06880396604537964)
        assert_allclose(p["sigma"].value, 0.2294706553220749)
        assert_allclose(p["sigma"].error, 0.04618723690509796)


class TestSourceCatalogLargeScaleHGPS:
    def setup(self):
        table = Table()
        table["GLON"] = [-30, -10, 10, 20] * u.deg
        table["Surface_Brightness"] = [0, 1, 10, 0] * u.Unit("cm-2 s-1 sr-1")
        table["GLAT"] = [-1, 0, 1, 0] * u.deg
        table["Width"] = [0.4, 0.5, 0.3, 1.0] * u.deg
        self.table = table
        self.model = SourceCatalogLargeScaleHGPS(table)

    def test_evaluate(self):
        x = np.linspace(-100, 20, 5)
        y = np.linspace(-2, 2, 7)
        x, y = np.meshgrid(x, y)
        coords = SkyCoord(x, y, unit="deg", frame="galactic")
        image = self.model.evaluate(coords)
        desired = 1.223962643740966 * u.Unit("cm-2 s-1 sr-1")
        assert_quantity_allclose(image.sum(), desired)

    def test_parvals(self):
        glon = Angle(10, unit="deg")
        assert_quantity_allclose(
            self.model.peak_brightness(glon), 10 * u.Unit("cm-2 s-1 sr-1")
        )
        assert_quantity_allclose(self.model.peak_latitude(glon), 1 * u.deg)
        assert_quantity_allclose(self.model.width(glon), 0.3 * u.deg)
