# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
from astropy import units as u
from astropy.utils.data import get_pkg_data_filename
from gammapy.catalog import (
    GammaCatResource,
    GammaCatResourceIndex,
    SourceCatalogGammaCat,
)
from gammapy.utils.gauss import Gauss2DPDF
from gammapy.utils.testing import (
    assert_quantity_allclose,
    requires_data,
    requires_dependency,
)

SOURCES = [
    {
        "name": "Vela X",
        "str_ref_file": "data/gammacat_vela_x.txt",
        "spec_type": "ecpl",
        "dnde_1TeV": 1.36e-11 * u.Unit("cm-2 s-1 TeV-1"),
        "dnde_1TeV_err": 7.531e-13 * u.Unit("cm-2 s-1 TeV-1"),
        "flux_1TeV": 2.104e-11 * u.Unit("cm-2 s-1"),
        "flux_1TeV_err": 1.973e-12 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 9.265778680255336e-11 * u.Unit("erg cm-2 s-1"),
        "eflux_1_10TeV_err": 9.590978299538194e-12 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 24,
        "is_pointlike": False,
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
        "flux_1TeV_err": 3.187e-13 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 6.235650344765057e-12 * u.Unit("erg cm-2 s-1"),
        "eflux_1_10TeV_err": 1.2210315515569183e-12 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 11,
        "is_pointlike": False,
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
        "flux_1TeV_err": 3.692e-13 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 8.923614018939419e-12 * u.Unit("erg cm-2 s-1"),
        "eflux_1_10TeV_err": 1.4613807070890267e-12 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 13,
        "is_pointlike": False,
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
        assert gammacat.name == "gamma-cat"
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
            assert gammacat[name].index == 112

    def test_sort_table(self, gammacat):
        name = "HESS J1848-018"
        sort_keys = ["ra", "dec", "reference_id"]
        for sort_key in sort_keys:
            # this test modifies the catalog, so we make a copy
            cat = gammacat.copy()
            cat.table.sort(sort_key)
            assert cat[name].name == name

    def test_to_sky_models(self, gammacat):
        models = gammacat.to_sky_models()

        assert len(models) == 74
        assert models[0].name == "CTA 1"
        assert_allclose(models[0].spectral_model.parameters["index"].value, 2.2)


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

    def test_data_python_dict(self, gammacat):
        source = gammacat[0]
        data = source._data_python_dict
        assert isinstance(data["ra"], float)
        assert data["ra"] == 1.649999976158142
        assert isinstance(data["sed_e_min"], list)
        assert isinstance(data["sed_e_min"][0], float)
        assert_allclose(data["sed_e_min"][0], 0.5600000023841858)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_spectral_model(self, gammacat, ref):
        source = gammacat[ref["name"]]
        spectral_model = source.spectral_model()

        assert source.data["spec_type"] == ref["spec_type"]

        e_min, e_max, e_inf = [1, 10, 1e10] * u.TeV

        dne = spectral_model(e_min)
        flux = spectral_model.integral(emin=e_min, emax=e_inf)
        eflux = spectral_model.energy_flux(emin=e_min, emax=e_max).to("erg cm-2 s-1")

        assert_quantity_allclose(dne, ref["dnde_1TeV"], rtol=1e-3)
        assert_quantity_allclose(flux, ref["flux_1TeV"], rtol=1e-3)
        assert_quantity_allclose(eflux, ref["eflux_1_10TeV"], rtol=1e-3)

    @requires_dependency("uncertainties")
    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_spectral_model_err(self, gammacat, ref):
        source = gammacat[ref["name"]]
        spectral_model = source.spectral_model()

        e_min, e_max, e_inf = [1, 10, 1e10] * u.TeV

        dnde, dnde_err = spectral_model.evaluate_error(e_min)
        flux, flux_err = spectral_model.integral_error(emin=e_min, emax=e_inf)
        eflux, eflux_err = spectral_model.energy_flux_error(emin=e_min, emax=e_max).to(
            "erg cm-2 s-1"
        )

        assert_quantity_allclose(dnde, ref["dnde_1TeV"], rtol=1e-3)
        assert_quantity_allclose(flux, ref["flux_1TeV"], rtol=1e-3)
        assert_quantity_allclose(eflux, ref["eflux_1_10TeV"], rtol=1e-3)

        assert_quantity_allclose(dnde_err, ref["dnde_1TeV_err"], rtol=1e-3)
        assert_quantity_allclose(flux_err, ref["flux_1TeV_err"], rtol=1e-3)
        assert_quantity_allclose(eflux_err, ref["eflux_1_10TeV_err"], rtol=1e-3)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_flux_points(self, gammacat, ref):
        source = gammacat[ref["name"]]

        flux_points = source.flux_points

        assert len(flux_points.table) == ref["n_flux_points"]

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
        # TODO: add a point and shell source -> separate list of sources for morphology test parametrization?
        assert spatial_model.__class__.__name__ == ref["spatial_model"]

        assert source.is_pointlike == ref["is_pointlike"]

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


class TestGammaCatResource:
    def setup(self):
        self.resource = GammaCatResource(
            source_id=42, reference_id="2010A&A...516A..62A", file_id=2
        )
        self.global_id = "42|2010A&A...516A..62A|2|none"

    def test_global_id(self):
        assert self.resource.global_id == self.global_id

    def test_eq(self):
        resource1 = self.resource
        resource2 = GammaCatResource(source_id=42, reference_id="2010A&A...516A..62A")

        assert resource1 == resource1
        assert resource1 != resource2

    def test_lt(self):
        resource = GammaCatResource(
            source_id=42, reference_id="2010A&A...516A..62A", file_id=2
        )

        assert not resource < resource

        assert resource < GammaCatResource(
            source_id=43, reference_id="2010A&A...516A..62A", file_id=2
        )
        assert resource < GammaCatResource(
            source_id=42, reference_id="2010A&A...516A..62B", file_id=2
        )
        assert resource < GammaCatResource(
            source_id=42, reference_id="2010A&A...516A..62A", file_id=3
        )

        assert resource > GammaCatResource(
            source_id=41, reference_id="2010A&A...516A..62A", file_id=2
        )

    def test_repr(self):
        expected = (
            "GammaCatResource(source_id=42, reference_id='2010A&A...516A..62A', "
            "file_id=2, type='none', location='none')"
        )
        assert repr(self.resource) == expected

    def test_to_dict(self):
        expected = {
            "source_id": 42,
            "reference_id": "2010A&A...516A..62A",
            "file_id": 2,
            "type": "none",
            "location": "none",
        }
        assert self.resource.to_dict() == expected

    def test_dict_roundtrip(self):
        actual = GammaCatResource.from_dict(self.resource.to_dict())
        assert actual == self.resource


class TestGammaCatResourceIndex:
    def setup(self):
        self.resource_index = GammaCatResourceIndex(
            [
                GammaCatResource(source_id=99, reference_id="2014ApJ...780..168A"),
                GammaCatResource(
                    source_id=42,
                    reference_id="2010A&A...516A..62A",
                    file_id=2,
                    type="sed",
                ),
                GammaCatResource(
                    source_id=42, reference_id="2010A&A...516A..62A", file_id=1
                ),
            ]
        )

    def test_repr(self):
        assert repr(self.resource_index) == "GammaCatResourceIndex(n_resources=3)"

    def test_eq(self):
        resource_index1 = self.resource_index
        resource_index2 = GammaCatResourceIndex(resource_index1.resources[:-1])

        assert resource_index1 == resource_index1
        assert resource_index1 != resource_index2

    def test_unique_source_ids(self):
        expected = [42, 99]
        assert self.resource_index.unique_source_ids == expected

    def test_unique_reference_ids(self):
        expected = ["2010A&A...516A..62A", "2014ApJ...780..168A"]
        assert self.resource_index.unique_reference_ids == expected

    def test_global_ids(self):
        expected = [
            "99|2014ApJ...780..168A|-1|none",
            "42|2010A&A...516A..62A|2|sed",
            "42|2010A&A...516A..62A|1|none",
        ]
        assert self.resource_index.global_ids == expected

    def test_sort(self):
        expected = [
            "42|2010A&A...516A..62A|1|none",
            "42|2010A&A...516A..62A|2|sed",
            "99|2014ApJ...780..168A|-1|none",
        ]
        assert self.resource_index.sort().global_ids == expected

    def test_to_list(self):
        result = self.resource_index.to_list()
        assert isinstance(result, list)
        assert len(result) == 3

    def test_list_roundtrip(self):
        data = self.resource_index.to_list()
        actual = GammaCatResourceIndex.from_list(data)
        assert actual == self.resource_index

    def test_to_table(self):
        table = self.resource_index.to_table()
        assert len(table) == 3
        assert table.colnames == [
            "source_id",
            "reference_id",
            "file_id",
            "type",
            "location",
        ]

    def test_table_roundtrip(self):
        table = self.resource_index.to_table()
        actual = GammaCatResourceIndex.from_table(table)
        assert actual == self.resource_index

    @requires_dependency("pandas")
    def test_to_pandas(self):
        df = self.resource_index.to_pandas()
        df2 = df.query("source_id == 42")
        assert len(df2) == 2

    @requires_dependency("pandas")
    def test_pandas_roundtrip(self):
        df = self.resource_index.to_pandas()
        actual = GammaCatResourceIndex.from_pandas(df)
        assert actual == self.resource_index

    @requires_dependency("pandas")
    def test_query(self):
        resource_index = self.resource_index.query('type == "sed" and source_id == 42')
        assert len(resource_index.resources) == 1
        assert resource_index.resources[0].global_id == "42|2010A&A...516A..62A|2|sed"
