# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import OrderedDict
from numpy.testing import assert_allclose
import pytest
from astropy import units as u
from ...utils.testing import requires_data, requires_dependency
from ...utils.testing import assert_quantity_allclose
from ..gammacat import SourceCatalogGammaCat
from ..gammacat import GammaCatResource, GammaCatResourceIndex

SOURCES = [
    {
        "name": "Vela X",
        "spec_type": "ecpl",
        "dnde_1TeV": 1.36e-11 * u.Unit("cm-2 s-1 TeV-1"),
        "dnde_1TeV_err": 7.531e-13 * u.Unit("cm-2 s-1 TeV-1"),
        "flux_1TeV": 2.104e-11 * u.Unit("cm-2 s-1"),
        "flux_1TeV_err": 1.973e-12 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 9.265778680255336e-11 * u.Unit("erg cm-2 s-1"),
        "eflux_1_10TeV_err": 9.590978299538194e-12 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 24,
        "is_pointlike": False,
        "spatial_model": "SkyGaussian",
    },
    {
        "name": "HESS J1848-018",
        "spec_type": "pl",
        "dnde_1TeV": 3.7e-12 * u.Unit("cm-2 s-1 TeV-1"),
        "dnde_1TeV_err": 4e-13 * u.Unit("cm-2 s-1 TeV-1"),
        "flux_1TeV": 2.056e-12 * u.Unit("cm-2 s-1"),
        "flux_1TeV_err": 3.187e-13 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 6.235650344765057e-12 * u.Unit("erg cm-2 s-1"),
        "eflux_1_10TeV_err": 1.2210315515569183e-12 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 11,
        "is_pointlike": False,
        "spatial_model": "SkyGaussian",
    },
    {
        "name": "HESS J1813-178",
        "spec_type": "pl2",
        "dnde_1TeV": 2.678e-12 * u.Unit("cm-2 s-1 TeV-1"),
        "dnde_1TeV_err": 2.55e-13 * u.Unit("cm-2 s-1 TeV-1"),
        "flux_1TeV": 2.457e-12 * u.Unit("cm-2 s-1"),
        "flux_1TeV_err": 3.692e-13 * u.Unit("cm-2 s-1"),
        "eflux_1_10TeV": 8.923614018939419e-12 * u.Unit("erg cm-2 s-1"),
        "eflux_1_10TeV_err": 1.4613807070890267e-12 * u.Unit("erg cm-2 s-1"),
        "n_flux_points": 13,
        "is_pointlike": False,
        "spatial_model": "SkyGaussian",
    },
]


@pytest.fixture(scope="session")
def gammacat():
    filename = "$GAMMAPY_EXTRA/datasets/catalogs/gammacat/gammacat.fits.gz"
    return SourceCatalogGammaCat(filename=filename)


@requires_data("gammapy-extra")
@requires_data("gamma-cat")
class TestSourceCatalogGammaCat:
    def test_source_table(self, gammacat):
        assert gammacat.name == "gamma-cat"
        assert len(gammacat.table) == 162

    def test_w28_alias_names(self, gammacat):
        names = [
            "W28",
            "HESS J1801-233",
            "W 28",
            "SNR G6.4-0.1",
            "SNR G006.4-00.1",
            "GRO J1801-2320",
        ]
        for name in names:
            assert str(gammacat[name]) == str(gammacat["W28"])

    def test_sort_table(self):
        name = "HESS J1848-018"
        sort_keys = ["ra", "dec", "reference_id"]
        for sort_key in sort_keys:
            # this test modifies the catalog, so we make a copy
            cat = gammacat()
            cat.table.sort(sort_key)
            assert cat[name].name == name

    def test_to_sky_models(self, gammacat):
        sources = gammacat.to_sky_models()
        source = sources.skymodels[0]

        assert len(sources.skymodels) == 74
        assert source.name == "CTA 1"
        assert_allclose(source.spectral_model.parameters["index"].value, 2.2)


@requires_data("gammapy-extra")
@requires_data("gamma-cat")
class TestSourceCatalogObjectGammaCat:
    def test_data(self, gammacat):
        source = gammacat[0]

        assert isinstance(source.data, OrderedDict)
        assert source.data["common_name"] == "CTA 1"
        assert_quantity_allclose(source.data["dec"], 72.782997 * u.deg)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_str(self, gammacat, ref):
        ss = str(gammacat[ref["name"]])
        assert ss == SOURCES_STR[ref["name"]]

    def test_data_python_dict(self, gammacat):
        source = gammacat[0]
        data = source._data_python_dict
        assert type(data["ra"]) == float
        assert data["ra"] == 1.649999976158142
        assert type(data["sed_e_min"]) == list
        assert type(data["sed_e_min"][0]) == float
        assert_allclose(data["sed_e_min"][0], 0.5600000023841858)

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_spectral_model(self, gammacat, ref):
        source = gammacat[ref["name"]]
        spectral_model = source.spectral_model

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
        spectral_model = source.spectral_model

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
    def test_spatial_model(self, gammacat, ref):
        source = gammacat[ref["name"]]

        spatial_model = source.spatial_model

        # TODO: put better asserts on model properties
        # TODO: add a point and shell source -> separate list of sources for morphology test parametrization?
        assert spatial_model.__class__.__name__ == ref["spatial_model"]

        assert source.is_pointlike == ref["is_pointlike"]

    @pytest.mark.parametrize("ref", SOURCES, ids=lambda _: _["name"])
    def test_sky_model(self, gammacat, ref):
        gammacat[ref["name"]].sky_model


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
        expected = OrderedDict(
            [
                ("source_id", 42),
                ("reference_id", "2010A&A...516A..62A"),
                ("file_id", 2),
                ("type", "none"),
                ("location", "none"),
            ]
        )
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


SOURCES_STR = {}

SOURCES_STR[
    "Vela X"
] = """
*** Basic info ***

Catalog row index (zero-based) : 36
Common name     : Vela X
Other names     : HESS J0835-455
Location        : gal
Class           : pwn

TeVCat ID       : 86
TeVCat 2 ID     : yVoFOS
TeVCat name     : TeV J0835-456

TGeVCat ID      : 37
TGeVCat name    : TeV J0835-4536

Discoverer      : hess
Discovery date  : 2006-03
Seen by         : hess
Reference       : 2012A&A...548A..38A

*** Position info ***

SIMBAD:
RA                   : 128.287 deg
DEC                  : -45.190 deg
GLON                 : 263.332 deg
GLAT                 : -3.106 deg

Measurement:
RA                   : 128.750 deg
DEC                  : -45.600 deg
GLON                 : 263.856 deg
GLAT                 : -3.089 deg
Position error       : nan deg

*** Morphology info ***

Morphology model type     : gauss
Sigma                     : 0.480 deg
Sigma error               : 0.030 deg
Sigma2                    : 0.360 deg
Sigma2 error              : 0.030 deg
Position angle            : 41.000 deg
Position angle error      : 7.000 deg
Position angle frame      : radec

*** Spectral info ***

Significance    : 27.900
Livetime        : 53.100 h

Spectrum type   : ecpl
norm            : 1.46e-11 +- 8e-13 (stat) +- 3e-12 (sys) cm-2 s-1 TeV-1
index           : 1.32 +- 0.06 (stat) +- 0.12 (sys)
e_cut           : 14.0 +- 1.6 (stat) +- 2.6 (stat) TeV
reference       : 1.0 TeV

Energy range         : (0.75, nan) TeV
theta                : 1.2 deg


Derived fluxes:
Spectral model norm (1 TeV)    : 1.36e-11 +- 7.53e-13 (stat) cm-2 s-1 TeV-1
Integrated flux (>1 TeV)       : 2.1e-11 +- 1.97e-12 (stat) cm-2 s-1
Integrated flux (>1 TeV)       : 101.425 +- 9.511 (% Crab)
Integrated flux (1-10 TeV)     : 9.27e-11 +- 9.59e-12 (stat) erg cm-2 s-1

*** Spectral points ***

SED reference id          : 2012A&A...548A..38A
Number of spectral points : 24
Number of upper limits    : 0

e_ref        dnde         dnde_errn       dnde_errp   
 TeV   1 / (cm2 s TeV) 1 / (cm2 s TeV) 1 / (cm2 s TeV)
------ --------------- --------------- ---------------
 0.719       1.055e-11       3.284e-12       3.280e-12
 0.868       1.304e-11       2.130e-12       2.130e-12
 1.051       9.211e-12       1.401e-12       1.399e-12
 1.274       8.515e-12       9.580e-13       9.610e-13
 1.546       5.378e-12       7.070e-13       7.090e-13
 1.877       4.455e-12       5.050e-13       5.070e-13
 2.275       3.754e-12       3.300e-13       3.340e-13
 2.759       2.418e-12       2.680e-13       2.700e-13
 3.352       1.605e-12       1.800e-13       1.830e-13
 4.078       1.445e-12       1.260e-13       1.290e-13
 4.956       9.240e-13       9.490e-14       9.700e-14
 6.008       7.348e-13       6.470e-14       6.710e-14
 7.271       3.863e-13       4.540e-14       4.700e-14
 8.795       3.579e-13       3.570e-14       3.750e-14
10.650       1.696e-13       2.490e-14       2.590e-14
12.910       1.549e-13       2.060e-14       2.160e-14
15.650       6.695e-14       1.134e-14       1.230e-14
18.880       2.105e-14       1.390e-14       1.320e-14
22.620       3.279e-14       6.830e-15       7.510e-15
26.870       3.026e-14       5.910e-15       6.660e-15
31.610       1.861e-14       4.380e-15       5.120e-15
36.970       5.653e-15       2.169e-15       2.917e-15
43.080       3.479e-15       1.641e-15       2.410e-15
52.370       1.002e-15       8.327e-16       1.615e-15
"""

SOURCES_STR[
    "HESS J1848-018"
] = """
*** Basic info ***

Catalog row index (zero-based) : 134
Common name     : HESS J1848-018
Other names     : HESS J1848-018,1HWC J1849-017c,WR121a,W43
Location        : gal
Class           : unid

TeVCat ID       : 187
TeVCat 2 ID     : hcE3Ou
TeVCat name     : TeV J1848-017

TGeVCat ID      : 128
TGeVCat name    : TeV J1848-0147

Discoverer      : hess
Discovery date  : 2008-07
Seen by         : hess
Reference       : 2008AIPC.1085..372C

*** Position info ***

SIMBAD:
RA                   : 282.120 deg
DEC                  : -1.792 deg
GLON                 : 31.000 deg
GLAT                 : -0.159 deg

Measurement:
RA                   : 282.121 deg
DEC                  : -1.792 deg
GLON                 : 31.000 deg
GLAT                 : -0.160 deg
Position error       : nan deg

*** Morphology info ***

Morphology model type     : gauss
Sigma                     : 0.320 deg
Sigma error               : 0.020 deg
Sigma2                    : nan deg
Sigma2 error              : nan deg
Position angle            : nan deg
Position angle error      : nan deg
Position angle frame      : 

*** Spectral info ***

Significance    : 9.000
Livetime        : 50.000 h

Spectrum type   : pl
norm            : 3.7e-12 +- 4e-13 (stat) +- 7e-13 (sys) cm-2 s-1 TeV-1
index           : 2.8 +- 0.2 (stat) +- 0.2 (sys)
reference       : 1.0 TeV

Energy range         : (0.9, 12.0) TeV
theta                : 0.2 deg


Derived fluxes:
Spectral model norm (1 TeV)    : 3.7e-12 +- 4e-13 (stat) cm-2 s-1 TeV-1
Integrated flux (>1 TeV)       : 2.06e-12 +- 3.19e-13 (stat) cm-2 s-1
Integrated flux (>1 TeV)       : 9.909 +- 1.536 (% Crab)
Integrated flux (1-10 TeV)     : 6.24e-12 +- 1.22e-12 (stat) erg cm-2 s-1

*** Spectral points ***

SED reference id          : 2008AIPC.1085..372C
Number of spectral points : 11
Number of upper limits    : 0

e_ref        dnde         dnde_errn       dnde_errp   
 TeV   1 / (cm2 s TeV) 1 / (cm2 s TeV) 1 / (cm2 s TeV)
------ --------------- --------------- ---------------
 0.624       9.942e-12       3.301e-12       3.265e-12
 0.878       6.815e-12       1.042e-12       1.029e-12
 1.284       1.707e-12       3.889e-13       3.826e-13
 1.881       5.027e-13       1.566e-13       1.533e-13
 2.754       3.266e-13       7.526e-14       7.323e-14
 4.033       8.183e-14       3.609e-14       3.503e-14
 5.905       2.979e-14       1.981e-14       1.921e-14
 8.648       4.022e-15       9.068e-15       8.729e-15
12.663      -6.647e-15       3.786e-15       3.675e-15
18.542       3.735e-15       2.009e-15       1.786e-15
27.173      -5.317e-16       9.236e-16       8.568e-16
"""

SOURCES_STR[
    "HESS J1813-178"
] = """
*** Basic info ***

Catalog row index (zero-based) : 118
Common name     : HESS J1813-178
Other names     : HESS J1813-178,G12.82-0.02,PSR J1813-1749,CXOU J181335.1-174957,IGR J18135-1751,W33
Location        : gal
Class           : pwn

TeVCat ID       : 114
TeVCat 2 ID     : Unhlxa
TeVCat name     : TeV J1813-178

TGeVCat ID      : 116
TGeVCat name    : TeV J1813-1750

Discoverer      : hess
Discovery date  : 2005-03
Seen by         : hess,magic
Reference       : 2006ApJ...636..777A

*** Position info ***

SIMBAD:
RA                   : 273.363 deg
DEC                  : -17.849 deg
GLON                 : 12.787 deg
GLAT                 : 0.000 deg

Measurement:
RA                   : 273.408 deg
DEC                  : -17.842 deg
GLON                 : 12.813 deg
GLAT                 : -0.034 deg
Position error       : 0.005 deg

*** Morphology info ***

Morphology model type     : gauss
Sigma                     : 0.036 deg
Sigma error               : 0.006 deg
Sigma2                    : nan deg
Sigma2 error              : nan deg
Position angle            : nan deg
Position angle error      : nan deg
Position angle frame      : 

*** Spectral info ***

Significance    : 13.500
Livetime        : 9.700 h

Spectrum type   : pl2
flux            : 1.42e-11 +- 1.1e-12 (stat) +- 3e-13 (sys) cm-2 s-1
index           : 2.09 +- 0.08 (stat) +- 0.2 (sys)
e_min           : 0.2 TeV
e_max           : nan TeV

Energy range         : (nan, nan) TeV
theta                : 0.15 deg


Derived fluxes:
Spectral model norm (1 TeV)    : 2.68e-12 +- 2.55e-13 (stat) cm-2 s-1 TeV-1
Integrated flux (>1 TeV)       : 2.46e-12 +- 3.69e-13 (stat) cm-2 s-1
Integrated flux (>1 TeV)       : 11.844 +- 1.780 (% Crab)
Integrated flux (1-10 TeV)     : 8.92e-12 +- 1.46e-12 (stat) erg cm-2 s-1

*** Spectral points ***

SED reference id          : 2006ApJ...636..777A
Number of spectral points : 13
Number of upper limits    : 0

e_ref        dnde         dnde_errn       dnde_errp   
 TeV   1 / (cm2 s TeV) 1 / (cm2 s TeV) 1 / (cm2 s TeV)
------ --------------- --------------- ---------------
 0.323       2.736e-11       5.690e-12       5.971e-12
 0.427       1.554e-11       3.356e-12       3.559e-12
 0.574       8.142e-12       1.603e-12       1.716e-12
 0.777       4.567e-12       9.319e-13       1.007e-12
 1.023       2.669e-12       5.586e-13       6.110e-13
 1.373       1.518e-12       3.378e-13       3.721e-13
 1.841       7.966e-13       2.166e-13       2.426e-13
 2.476       3.570e-13       1.135e-13       1.295e-13
 3.159       3.321e-13       8.757e-14       1.012e-13
 4.414       1.934e-13       5.764e-14       6.857e-14
 5.560       4.461e-14       2.130e-14       2.844e-14
10.765       1.318e-14       6.056e-15       1.085e-14
22.052       1.372e-14       6.128e-15       1.178e-14
"""
