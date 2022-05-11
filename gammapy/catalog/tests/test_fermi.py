# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.time import Time
from astropy.utils.data import get_pkg_data_filename
from gammapy.catalog import (
    SourceCatalog2FHL,
    SourceCatalog3FGL,
    SourceCatalog3FHL,
    SourceCatalog4FGL,
)
from gammapy.modeling.models import (
    ExpCutoffPowerLaw3FGLSpectralModel,
    LogParabolaSpectralModel,
    PowerLaw2SpectralModel,
    PowerLawSpectralModel,
    SuperExpCutoffPowerLaw3FGLSpectralModel,
    SuperExpCutoffPowerLaw4FGLSpectralModel,
)
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.testing import (
    assert_quantity_allclose,
    assert_time_allclose,
    requires_data,
)

SOURCES_4FGL = [
    dict(
        idx=0,
        name="4FGL J0000.3-7355",
        str_ref_file="data/4fgl_J0000.3-7355.txt",
        spec_type=PowerLawSpectralModel,
        dnde=u.Quantity(2.9476e-11, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(5.3318e-12, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=3,
        name="4FGL J0001.5+2113",
        str_ref_file="data/4fgl_J0001.5+2113.txt",
        spec_type=LogParabolaSpectralModel,
        dnde=u.Quantity(2.8545e-8, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(1.3324e-9, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=7,
        name="4FGL J0002.8+6217",
        str_ref_file="data/4fgl_J0002.8+6217.txt",
        spec_type=SuperExpCutoffPowerLaw4FGLSpectralModel,
        dnde=u.Quantity(2.084e-09, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(1.0885e-10, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=2718,
        name="4FGL J1409.1-6121e",
        str_ref_file="data/4fgl_J1409.1-6121e.txt",
        spec_type=LogParabolaSpectralModel,
        dnde=u.Quantity(1.3237202133031811e-12, "cm-2 s-1 MeV-1"),
        dnde_err=u.Quantity(4.513233455580648e-14, "cm-2 s-1 MeV-1"),
    ),
]

SOURCES_3FGL = [
    dict(
        idx=0,
        name="3FGL J0000.1+6545",
        str_ref_file="data/3fgl_J0000.1+6545.txt",
        spec_type=PowerLawSpectralModel,
        dnde=u.Quantity(1.4351261e-9, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(2.1356270e-10, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=4,
        name="3FGL J0001.4+2120",
        str_ref_file="data/3fgl_J0001.4+2120.txt",
        spec_type=LogParabolaSpectralModel,
        dnde=u.Quantity(8.3828599e-10, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(2.6713238e-10, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=55,
        name="3FGL J0023.4+0923",
        str_ref_file="data/3fgl_J0023.4+0923.txt",
        spec_type=ExpCutoffPowerLaw3FGLSpectralModel,
        dnde=u.Quantity(1.8666925e-09, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(2.2068837e-10, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=960,
        name="3FGL J0835.3-4510",
        str_ref_file="data/3fgl_J0835.3-4510.txt",
        spec_type=SuperExpCutoffPowerLaw3FGLSpectralModel,
        dnde=u.Quantity(1.6547128794756733e-06, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(1.6621504e-11, "cm-2 s-1 MeV-1"),
    ),
]

SOURCES_2FHL = [
    dict(
        idx=221,
        name="2FHL J1445.1-0329",
        str_ref_file="data/2fhl_j1445.1-0329.txt",
        spec_type=PowerLaw2SpectralModel,
        dnde=u.Quantity(1.065463448091757e-10, "cm-2 s-1 TeV-1"),
        dnde_err=u.Quantity(4.9691205387540815e-11, "cm-2 s-1 TeV-1"),
    ),
    dict(
        idx=134,
        name="2FHL J0822.6-4250e",
        str_ref_file="data/2fhl_j0822.6-4250e.txt",
        spec_type=LogParabolaSpectralModel,
        dnde=u.Quantity(2.46548351696472e-10, "cm-2 s-1 TeV-1"),
        dnde_err=u.Quantity(9.771755529198772e-11, "cm-2 s-1 TeV-1"),
    ),
]

SOURCES_3FHL = [
    dict(
        idx=352,
        name="3FHL J0534.5+2201",
        spec_type=PowerLawSpectralModel,
        dnde=u.Quantity(6.3848912826152664e-12, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(2.679593524691324e-13, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=1442,
        name="3FHL J2158.8-3013",
        spec_type=LogParabolaSpectralModel,
        dnde=u.Quantity(2.056998292908196e-12, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(4.219030630302381e-13, "cm-2 s-1 GeV-1"),
    ),
]


@requires_data()
def test_4FGL_DR3():
    cat = SourceCatalog4FGL("$GAMMAPY_DATA/catalogs/fermi/gll_psc_v28.fit.gz")
    source = cat["4FGL J0534.5+2200"]
    model = source.spectral_model()
    fp = source.flux_points
    not_ul = ~fp.is_ul.data.squeeze()
    fp_dnde = fp.dnde.quantity.squeeze()[not_ul]
    model_dnde = model(fp.energy_ref[not_ul])
    assert_quantity_allclose(model_dnde, fp_dnde, rtol=0.07)


@requires_data()
class TestFermi4FGLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog4FGL("$GAMMAPY_DATA/catalogs/fermi/gll_psc_v20.fit.gz")
        cls.source_name = "4FGL J0534.5+2200"
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_row_index(self):
        assert self.source.row_index == 995

    @pytest.mark.parametrize("ref", SOURCES_4FGL, ids=lambda _: _["name"])
    def test_str(self, ref):
        actual = str(self.cat[ref["idx"]])
        expected = open(get_pkg_data_filename(ref["str_ref_file"])).read()
        assert actual == expected

    @pytest.mark.parametrize("ref", SOURCES_4FGL, ids=lambda _: _["name"])
    def test_spectral_model(self, ref):
        model = self.cat[ref["idx"]].spectral_model()

        e_ref = model.reference.quantity
        dnde, dnde_err = model.evaluate_error(e_ref)
        assert isinstance(model, ref["spec_type"])
        assert_quantity_allclose(dnde, ref["dnde"], rtol=1e-4)
        assert_quantity_allclose(dnde_err, ref["dnde_err"], rtol=1e-4)

    def test_spatial_model(self):
        model = self.cat["4FGL J0000.3-7355"].spatial_model()
        assert "PointSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 0.0983)
        assert_allclose(p["lat_0"].value, -73.921997)
        pos_err = model.position_error
        assert_allclose(pos_err.angle.value, -62.7)
        assert_allclose(0.5 * pos_err.height.value, 0.0525, rtol=1e-4)
        assert_allclose(0.5 * pos_err.width.value, 0.051, rtol=1e-4)
        assert_allclose(model.position.ra.value, pos_err.center.ra.value)
        assert_allclose(model.position.dec.value, pos_err.center.dec.value)

        model = self.cat["4FGL J1409.1-6121e"].spatial_model()
        assert "DiskSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 212.294006)
        assert_allclose(p["lat_0"].value, -61.353001)
        assert_allclose(p["r_0"].value, 0.7331369519233704)

        model = self.cat["4FGL J0617.2+2234e"].spatial_model()
        assert "GaussianSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 94.309998)
        assert_allclose(p["lat_0"].value, 22.58)
        assert_allclose(p["sigma"].value, 0.27)

        model = self.cat["4FGL J1443.0-6227e"].spatial_model()
        assert "TemplateSpatialModel" in model.tag
        assert model.frame == "fk5"
        assert model.normalize

    @pytest.mark.parametrize("ref", SOURCES_4FGL, ids=lambda _: _["name"])
    def test_sky_model(self, ref):
        self.cat[ref["idx"]].sky_model

    def test_flux_points(self):
        flux_points = self.source.flux_points

        assert flux_points.norm.geom.axes["energy"].nbin == 7
        assert flux_points.norm_ul

        desired = [
            2.2378458e-06,
            1.4318283e-06,
            5.4776939e-07,
            1.2769708e-07,
            2.5820052e-08,
            2.3897000e-09,
            7.1766204e-11,
        ]
        assert_allclose(flux_points.flux.data.flat, desired, rtol=1e-5)

    def test_flux_points_meta(self):
        source = self.cat["4FGL J0000.3-7355"]
        fp = source.flux_points

        assert_allclose(fp.sqrt_ts_threshold_ul, 1)
        assert_allclose(fp.n_sigma, 1)
        assert_allclose(fp.n_sigma_ul, 2)

    def test_flux_points_ul(self):
        source = self.cat["4FGL J0000.3-7355"]
        flux_points = source.flux_points

        desired = [
            4.13504750e-08,
            3.80519616e-09,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            7.99699456e-12,
        ]
        assert_allclose(flux_points.flux_ul.data.flat, desired, rtol=1e-5)

    def test_lightcurve_dr1(self):
        lc = self.source.lightcurve(interval="1-year")
        table = lc.to_table(format="lightcurve", sed_type="flux")

        assert len(table) == 8
        assert table.colnames == [
            "time_min",
            "time_max",
            "e_ref",
            "e_min",
            "e_max",
            "flux",
            "flux_errp",
            "flux_errn",
            "flux_ul",
            "ts",
            "sqrt_ts",
            "is_ul",
        ]
        axis = lc.geom.axes["time"]
        expected = Time(54682.6552835, format="mjd", scale="utc")
        assert_time_allclose(axis.time_min[0].utc, expected)

        expected = Time(55045.30090278, format="mjd", scale="utc")
        assert_time_allclose(axis.time_max[0].utc, expected)

        assert table["flux"].unit == "cm-2 s-1"
        assert_allclose(table["flux"][0], 2.2122326e-06, rtol=1e-3)

        assert table["flux_errp"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errp"][0], 2.3099371e-08, rtol=1e-3)

        assert table["flux_errn"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errn"][0], 2.3099371e-08, rtol=1e-3)

        table = self.source.lightcurve(interval="2-month").to_table(
            format="lightcurve", sed_type="flux"
        )
        assert len(table) == 48  # (12 month/year / 2month) * 8 years

        assert table["flux"].unit == "cm-2 s-1"
        assert_allclose(table["flux"][0], 2.238483e-6, rtol=1e-3)

        assert table["flux_errp"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errp"][0], 4.437058e-8, rtol=1e-3)

        assert table["flux_errn"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errn"][0], 4.437058e-8, rtol=1e-3)

    def test_lightcurve_dr2(self):
        dr2 = SourceCatalog4FGL("$GAMMAPY_DATA/catalogs/fermi/gll_psc_v27.fit.gz")
        source_dr2 = dr2[self.source_name]
        table = source_dr2.lightcurve(interval="1-year").to_table(
            format="lightcurve", sed_type="flux"
        )

        assert table["flux"].unit == "cm-2 s-1"
        assert_allclose(table["flux"][0], 2.196788e-6, rtol=1e-3)

        assert table["flux_errp"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errp"][0], 2.312938e-8, rtol=1e-3)

        assert table["flux_errn"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errn"][0], 2.312938e-8, rtol=1e-3)

        with pytest.raises(ValueError):
            source_dr2.lightcurve(interval="2-month")


@requires_data()
class TestFermi3FGLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FGL()
        # Use 3FGL J0534.5+2201 (Crab) as a test source
        cls.source_name = "3FGL J0534.5+2201"
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_row_index(self):
        assert self.source.row_index == 621

    def test_data(self):
        assert_allclose(self.source.data["Signif_Avg"], 30.669872283935547)

    def test_position(self):
        position = self.source.position
        assert_allclose(position.ra.deg, 83.637199, atol=1e-3)
        assert_allclose(position.dec.deg, 22.024099, atol=1e-3)

    @pytest.mark.parametrize("ref", SOURCES_3FGL, ids=lambda _: _["name"])
    def test_str(self, ref):
        actual = str(self.cat[ref["idx"]])
        expected = open(get_pkg_data_filename(ref["str_ref_file"])).read()
        assert actual == expected

    @pytest.mark.parametrize("ref", SOURCES_3FGL, ids=lambda _: _["name"])
    def test_spectral_model(self, ref):
        model = self.cat[ref["idx"]].spectral_model()

        dnde, dnde_err = model.evaluate_error(1 * u.GeV)

        assert isinstance(model, ref["spec_type"])
        assert_quantity_allclose(dnde, ref["dnde"])
        assert_quantity_allclose(dnde_err, ref["dnde_err"], rtol=1e-3)

    def test_spatial_model(self):
        model = self.cat[0].spatial_model()
        assert "PointSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 0.0377)
        assert_allclose(p["lat_0"].value, 65.751701)

        model = self.cat[122].spatial_model()
        assert "GaussianSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 14.75)
        assert_allclose(p["lat_0"].value, -72.699997)
        assert_allclose(p["sigma"].value, 1.35)

        model = self.cat[955].spatial_model()
        assert "DiskSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 128.287201)
        assert_allclose(p["lat_0"].value, -45.190102)
        assert_allclose(p["r_0"].value, 0.91)

        model = self.cat[602].spatial_model()
        assert "TemplateSpatialModel" in model.tag
        assert model.frame == "fk5"
        assert model.normalize

        model = self.cat["3FGL J0000.2-3738"].spatial_model()
        pos_err = model.position_error
        assert_allclose(pos_err.angle.value, -88.55)
        assert_allclose(0.5 * pos_err.height.value, 0.0731, rtol=1e-4)
        assert_allclose(0.5 * pos_err.width.value, 0.0676, rtol=1e-4)
        assert_allclose(model.position.ra.value, pos_err.center.ra.value)
        assert_allclose(model.position.dec.value, pos_err.center.dec.value)

    @pytest.mark.parametrize("ref", SOURCES_3FGL, ids=lambda _: _["name"])
    def test_sky_model(self, ref):
        self.cat[ref["idx"]].sky_model()

    def test_flux_points(self):
        flux_points = self.source.flux_points

        assert flux_points.energy_axis.nbin == 5
        assert flux_points.norm_ul

        desired = [1.645888e-06, 5.445407e-07, 1.255338e-07, 2.545524e-08, 2.263189e-09]
        assert_allclose(flux_points.flux.data.flat, desired, rtol=1e-5)

    def test_flux_points_ul(self):
        source = self.cat["3FGL J0000.2-3738"]
        flux_points = source.flux_points

        desired = [4.096391e-09, 6.680059e-10, np.nan, np.nan, np.nan]
        assert_allclose(flux_points.flux_ul.data.flat, desired, rtol=1e-5)

    def test_lightcurve(self):
        lc = self.source.lightcurve()
        table = lc.to_table(format="lightcurve", sed_type="flux")

        assert len(table) == 48
        assert table.colnames == [
            "time_min",
            "time_max",
            "e_ref",
            "e_min",
            "e_max",
            "flux",
            "flux_errp",
            "flux_errn",
            "flux_ul",
            "is_ul",
        ]

        expected = Time(54680.02313657408, format="mjd", scale="utc")
        axis = lc.geom.axes["time"]
        assert_time_allclose(axis.time_min[0].utc, expected)

        expected = Time(54710.46295139, format="mjd", scale="utc")
        assert_time_allclose(axis.time_max[0].utc, expected)

        assert table["flux"].unit == "cm-2 s-1"
        assert_allclose(table["flux"][0], 2.384e-06, rtol=1e-3)

        assert table["flux_errp"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errp"][0], 8.071e-08, rtol=1e-3)

        assert table["flux_errn"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errn"][0], 8.071e-08, rtol=1e-3)

    def test_crab_alias(self):
        for name in [
            "Crab",
            "3FGL J0534.5+2201",
            "1FHL J0534.5+2201",
            "PSR J0534+2200",
        ]:
            assert self.cat[name].row_index == 621


@requires_data()
class TestFermi2FHLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog2FHL()
        # Use 2FHL J0534.5+2201 (Crab) as a test source
        cls.source_name = "2FHL J0534.5+2201"
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_position(self):
        position = self.source.position
        assert_allclose(position.ra.deg, 83.634102, atol=1e-3)
        assert_allclose(position.dec.deg, 22.0215, atol=1e-3)

    @pytest.mark.parametrize("ref", SOURCES_2FHL, ids=lambda _: _["name"])
    def test_str(self, ref):
        actual = str(self.cat[ref["idx"]])
        expected = open(get_pkg_data_filename(ref["str_ref_file"])).read()
        assert actual == expected

    def test_spectral_model(self):
        model = self.source.spectral_model()
        energy = u.Quantity(100, "GeV")
        desired = u.Quantity(6.8700477298e-12, "cm-2 GeV-1 s-1")
        assert_quantity_allclose(model(energy), desired)

    def test_flux_points(self):
        # test flux point on  PKS 2155-304
        src = self.cat["PKS 2155-304"]
        flux_points = src.flux_points
        actual = flux_points.flux.quantity[:, 0, 0]
        desired = [2.866363e-10, 6.118736e-11, 3.257970e-16] * u.Unit("cm-2 s-1")
        assert_quantity_allclose(actual, desired)

        actual = flux_points.flux_ul.quantity[:, 0, 0]
        desired = [np.nan, np.nan, 1.294092e-11] * u.Unit("cm-2 s-1")
        assert_quantity_allclose(actual, desired, rtol=1e-3)

    def test_spatial_model(self):
        model = self.cat[221].spatial_model()
        assert "PointSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 221.281998, rtol=1e-5)
        assert_allclose(p["lat_0"].value, -3.4943, rtol=1e-5)

        model = self.cat["2FHL J1304.5-4353"].spatial_model()
        pos_err = model.position_error
        scale = Gauss2DPDF().containment_radius(0.95) / Gauss2DPDF().containment_radius(
            0.68
        )
        assert_allclose(pos_err.height.value, 2 * 0.041987 * scale, rtol=1e-4)
        assert_allclose(pos_err.width.value, 2 * 0.041987 * scale, rtol=1e-4)
        assert_allclose(model.position.ra.value, pos_err.center.ra.value)
        assert_allclose(model.position.dec.value, pos_err.center.dec.value)

        model = self.cat[97].spatial_model()
        assert "GaussianSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 94.309998, rtol=1e-5)
        assert_allclose(p["lat_0"].value, 22.58, rtol=1e-5)
        assert_allclose(p["sigma"].value, 0.27)

        model = self.cat[134].spatial_model()
        assert "DiskSpatialModel" in model.tag
        assert model.frame == "icrs"
        p = model.parameters
        assert_allclose(p["lon_0"].value, 125.660004, rtol=1e-5)
        assert_allclose(p["lat_0"].value, -42.84, rtol=1e-5)
        assert_allclose(p["r_0"].value, 0.37)

        model = self.cat[256].spatial_model()
        assert "TemplateSpatialModel" in model.tag
        assert model.frame == "fk5"
        assert model.normalize
        # TODO: have to check the extended template used for RX J1713,
        # for now I guess it's the same than for 3FGL
        # and added a copy with the name given by 2FHL in gammapy-extra


@requires_data()
class TestFermi3FHLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FHL()
        # Use 3FHL J0534.5+2201 (Crab) as a test source
        cls.source_name = "3FHL J0534.5+2201"
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_row_index(self):
        assert self.source.row_index == 352

    def test_data(self):
        assert_allclose(self.source.data["Signif_Avg"], 168.64082)

    def test_str(self):
        actual = str(self.cat["3FHL J2301.9+5855e"])  # an extended source
        expected = open(get_pkg_data_filename("data/3fhl_j2301.9+5855e.txt")).read()
        assert actual == expected

    def test_position(self):
        position = self.source.position
        assert_allclose(position.ra.deg, 83.634834, atol=1e-3)
        assert_allclose(position.dec.deg, 22.019203, atol=1e-3)

    @pytest.mark.parametrize("ref", SOURCES_3FHL, ids=lambda _: _["name"])
    def test_spectral_model(self, ref):
        model = self.cat[ref["idx"]].spectral_model()

        dnde, dnde_err = model.evaluate_error(100 * u.GeV)

        assert isinstance(model, ref["spec_type"])
        assert_quantity_allclose(dnde, ref["dnde"])
        assert_quantity_allclose(dnde_err, ref["dnde_err"], rtol=1e-3)

    @pytest.mark.parametrize("ref", SOURCES_3FHL, ids=lambda _: _["name"])
    def test_spatial_model(self, ref):
        model = self.cat[ref["idx"]].spatial_model()
        assert model.frame == "icrs"

        model = self.cat["3FHL J0002.1-6728"].spatial_model()
        pos_err = model.position_error
        assert_allclose(0.5 * pos_err.height.value, 0.035713, rtol=1e-4)
        assert_allclose(0.5 * pos_err.width.value, 0.035713, rtol=1e-4)
        assert_allclose(model.position.ra.value, pos_err.center.ra.value)
        assert_allclose(model.position.dec.value, pos_err.center.dec.value)

    @pytest.mark.parametrize("ref", SOURCES_3FHL, ids=lambda _: _["name"])
    def test_sky_model(self, ref):
        self.cat[ref["idx"]].sky_model()

    def test_flux_points(self):
        flux_points = self.source.flux_points

        assert flux_points.energy_axis.nbin == 5
        assert flux_points.norm_ul

        desired = [5.169889e-09, 2.245024e-09, 9.243175e-10, 2.758956e-10, 6.684021e-11]
        assert_allclose(flux_points.flux.data[:, 0, 0], desired, rtol=1e-3)

    def test_crab_alias(self):
        for name in ["Crab Nebula", "3FHL J0534.5+2201", "3FGL J0534.5+2201i"]:
            assert self.cat[name].row_index == 352


@requires_data()
class TestSourceCatalog3FGL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FGL()

    def test_main_table(self):
        assert len(self.cat.table) == 3034

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 25


@requires_data()
class TestSourceCatalog2FHL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog2FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 360

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 25

    def test_crab_alias(self):
        for name in ["Crab", "3FGL J0534.5+2201i", "1FHL J0534.5+2201"]:
            assert self.cat[name].row_index == 85


@requires_data()
class TestSourceCatalog3FHL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 1556

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 55

    def test_to_models(self):
        mask = self.cat.table["GLAT"].quantity > 80 * u.deg
        subcat = self.cat[mask]
        models = subcat.to_models()
        assert len(models) == 17
