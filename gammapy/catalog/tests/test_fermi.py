# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy import units as u
from ...utils.testing import assert_quantity_allclose
from ...utils.testing import requires_data, requires_dependency
from ...spectrum.models import (
    PowerLaw,
    LogParabola,
    ExponentialCutoffPowerLaw3FGL,
    PLSuperExpCutoff3FGL,
)
from .. import (
    SourceCatalog3FGL,
    SourceCatalog2FHL,
    SourceCatalog1FHL,
    SourceCatalog3FHL,
)

SOURCES_3FGL = [
    dict(
        idx=0,
        spec_type=PowerLaw,
        dnde=u.Quantity(1.4351261e-9, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(2.1356270e-10, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=4,
        spec_type=LogParabola,
        dnde=u.Quantity(8.3828599e-10, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(2.6713238e-10, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=55,
        spec_type=ExponentialCutoffPowerLaw3FGL,
        dnde=u.Quantity(1.8666925e-09, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(2.2068837e-10, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=960,
        spec_type=PLSuperExpCutoff3FGL,
        dnde=u.Quantity(1.6547128794756733e-06, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(1.6621504e-11, "cm-2 s-1 MeV-1"),
    ),
]

SOURCES_3FHL = [
    dict(
        idx=352,
        spec_type=PowerLaw,
        dnde=u.Quantity(6.3848912826152664e-12, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(2.679593524691324e-13, "cm-2 s-1 GeV-1"),
    ),
    dict(
        idx=1442,
        spec_type=LogParabola,
        dnde=u.Quantity(2.056998292908196e-12, "cm-2 s-1 GeV-1"),
        dnde_err=u.Quantity(4.219030630302381e-13, "cm-2 s-1 GeV-1"),
    ),
]


@requires_data("gammapy-extra")
class TestFermi3FGLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FGL()
        # Use 3FGL J0534.5+2201 (Crab) as a test source
        cls.source_name = "3FGL J0534.5+2201"
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_index(self):
        assert self.source.index == 621

    def test_data(self):
        assert_allclose(self.source.data["Signif_Avg"], 30.669872283935547)

    def test_str(self):
        ss = str(self.source)
        assert "Source name          : 3FGL J0534.5+2201" in ss
        assert "RA                   : 83.637 deg" in ss
        assert "Detection significance (100 MeV - 300 GeV)    : 30.670" in ss
        assert (
            "Integral flux (1 - 100 GeV)                   : 1.57e-07 +- 1.08e-09 cm-2 s-1"
            in ss
        )

    @pytest.mark.parametrize("ref", SOURCES_3FGL, ids=lambda _: _["name"])
    def test_str_all(self, ref):
        ss = str(self.cat[ref["idx"]])
        # TODO: put better assert on content. Maybe like for gamma-cat?
        assert "Source name" in ss

    def test_data_python_dict(self):
        data = self.source._data_python_dict
        assert type(data["RAJ2000"]) == float
        assert data["RAJ2000"] == 83.63719940185547
        assert type(data["Unc_Flux100_300"]) == list
        assert type(data["Unc_Flux100_300"][0]) == float
        assert_allclose(data["Unc_Flux100_300"][0], -1.44535601265261e-08)

    @requires_dependency("uncertainties")
    @pytest.mark.parametrize("ref", SOURCES_3FGL, ids=lambda _: _["name"])
    def test_spectral_model(self, ref):
        model = self.cat[ref["idx"]].spectral_model

        dnde, dnde_err = model.evaluate_error(1 * u.GeV)

        assert isinstance(model, ref["spec_type"])
        assert_quantity_allclose(dnde, ref["dnde"])
        assert_quantity_allclose(dnde_err, ref["dnde_err"])

    @pytest.mark.parametrize("ref", SOURCES_3FGL, ids=lambda _: _["name"])
    def test_spatial_model(self, ref):
        self.cat[ref["idx"]].spatial_model

    @pytest.mark.parametrize("ref", SOURCES_3FGL, ids=lambda _: _["name"])
    def test_sky_model(self, ref):
        self.cat[ref["idx"]].sky_model

    def test_flux_points(self):
        flux_points = self.source.flux_points

        assert len(flux_points.table) == 5
        assert "flux_ul" in flux_points.table.colnames

        desired = [8.174943e-03, 7.676263e-04, 6.119782e-05, 3.350906e-06, 1.308784e-08]
        assert_allclose(flux_points.table["dnde"].data, desired, rtol=1e-5)

    def test_flux_points_ul(self):
        source = self.cat["3FGL J0000.2-3738"]
        flux_points = source.flux_points

        desired = [4.096391e-09, 6.680059e-10, np.nan, np.nan, np.nan]
        assert_allclose(flux_points.table["flux_ul"].data, desired, rtol=1e-5)

    def test_lightcurve(self):
        lc = self.source.lightcurve
        table = lc.table

        assert len(table) == 48
        assert table.colnames == [
            "time_min",
            "time_max",
            "flux",
            "flux_errp",
            "flux_errn",
        ]

        assert lc.time_min[0].fits == "2008-08-02T00:33:19.000(UTC)"
        assert lc.time_max[0].fits == "2008-09-01T10:31:04.625(UTC)"

        assert table["flux"].unit == "cm-2 s-1"
        assert_allclose(table["flux"][0], 2.384e-06, rtol=1e-3)

        assert table["flux_errp"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errp"][0], 8.071e-08, rtol=1e-3)

        assert table["flux_errn"].unit == "cm-2 s-1"
        assert_allclose(table["flux_errn"][0], 8.071e-08, rtol=1e-3)

    @pytest.mark.parametrize(
        "name",
        [
            "Crab",
            "3FGL J0534.5+2201",
            "1FHL J0534.5+2201",
            "2FGL J0534.5+2201",
            "PSR J0534+2200",
            "0FGL J0534.6+2201",
        ],
    )
    def test_crab_alias(self, name):
        assert str(self.cat["Crab"]) == str(self.cat[name])


@requires_data("gammapy-extra")
class TestFermi1FHLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog1FHL()
        # Use 1FHL J0534.5+2201 (Crab) as a test source
        cls.source_name = "1FHL J0534.5+2201"
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_spectral_model(self):
        model = self.source.spectral_model
        energy = u.Quantity(100, "GeV")
        desired = u.Quantity(4.7717464131e-12, "cm-2 GeV-1 s-1")
        assert_quantity_allclose(model(energy), desired)

    def test_flux_points(self):
        # test flux point on  PKS 2155-304
        src = self.cat["1FHL J0153.1+7515"]
        flux_points = src.flux_points
        actual = flux_points.table["flux"]
        desired = [5.523017e-11, 0, 0] * u.Unit("cm-2 s-1")
        assert_quantity_allclose(actual, desired)

        actual = flux_points.table["flux_ul"]
        desired = [np.nan, 4.163177e-11, 2.599397e-11] * u.Unit("cm-2 s-1")
        assert_quantity_allclose(actual, desired, rtol=1e-5)


@requires_data("gammapy-extra")
class TestFermi2FHLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog2FHL()
        # Use 2FHL J0534.5+2201 (Crab) as a test source
        cls.source_name = "2FHL J0534.5+2201"
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_spectral_model(self):
        model = self.source.spectral_model
        energy = u.Quantity(100, "GeV")
        desired = u.Quantity(6.8700477298e-12, "cm-2 GeV-1 s-1")
        assert_quantity_allclose(model(energy), desired)

    def test_flux_points(self):
        # test flux point on  PKS 2155-304
        src = self.cat["PKS 2155-304"]
        flux_points = src.flux_points
        actual = flux_points.table["flux"]
        desired = [2.866363e-10, 6.118736e-11, 3.257970e-16] * u.Unit("cm-2 s-1")
        assert_quantity_allclose(actual, desired)

        actual = flux_points.table["flux_ul"]
        desired = [np.nan, np.nan, 1.294092e-11] * u.Unit("cm-2 s-1")
        assert_quantity_allclose(actual, desired, rtol=1E-3)


@requires_data("gammapy-extra")
class TestFermi3FHLObject:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FHL()
        # Use 3FHL J0534.5+2201 (Crab) as a test source
        cls.source_name = "3FHL J0534.5+2201"
        cls.source = cls.cat[cls.source_name]

    def test_name(self):
        assert self.source.name == self.source_name

    def test_index(self):
        assert self.source.index == 352

    def test_data(self):
        assert_allclose(self.source.data["Signif_Avg"], 168.64082)

    def test_str(self):
        source = self.cat["3FHL J2301.9+5855e"]  # Picking an extended source
        ss = str(source)
        assert "Source name          : 3FHL J2301.9+5855e" in ss
        assert "RA                   : 345.494 deg" in ss
        assert "Significance (10 GeV - 2 TeV)    : 7.974" in ss
        assert "Integral flux (10 GeV - 1 TeV)   : 1.46e-10 +- 2.57e-11 cm-2 s-1" in ss
        assert "Model form       : Disk" in ss

    def test_data_python_dict(self):
        data = self.source._data_python_dict
        assert type(data["RAJ2000"]) == float
        assert data["RAJ2000"] == 83.63483428955078
        assert type(data["Flux_Band"]) == list
        assert type(data["Flux_Band"][0]) == float
        assert_allclose(data["Flux_Band"][0], 5.1698894054652555e-09)

    @requires_dependency("uncertainties")
    @pytest.mark.parametrize("ref", SOURCES_3FHL, ids=lambda _: _["name"])
    def test_spectral_model(self, ref):
        model = self.cat[ref["idx"]].spectral_model

        dnde, dnde_err = model.evaluate_error(100 * u.GeV)

        assert isinstance(model, ref["spec_type"])
        assert_quantity_allclose(dnde, ref["dnde"])
        assert_quantity_allclose(dnde_err, ref["dnde_err"])

    @pytest.mark.parametrize("ref", SOURCES_3FHL, ids=lambda _: _["name"])
    def test_spatial_model(self, ref):
        self.cat[ref["idx"]].spatial_model

    @pytest.mark.parametrize("ref", SOURCES_3FHL, ids=lambda _: _["name"])
    def test_sky_model(self, ref):
        self.cat[ref["idx"]].sky_model

    def test_flux_points(self):
        flux_points = self.source.flux_points

        assert len(flux_points.table) == 5
        assert "flux_ul" in flux_points.table.colnames

        desired = [5.124e-07, 7.370e-08, 9.044e-09, 7.681e-10, 4.307e-11]
        assert_allclose(flux_points.table["dnde"].data, desired, rtol=1e-3)

    @pytest.mark.parametrize(
        "name", ["Crab Nebula", "3FHL J0534.5+2201", "3FGL J0534.5+2201i"]
    )
    def test_crab_alias(self, name):
        assert str(self.cat["Crab Nebula"]) == str(self.cat[name])


@requires_data("gammapy-extra")
class TestSourceCatalog3FGL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FGL()

    def test_main_table(self):
        assert len(self.cat.table) == 3034

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 25

    def test_select_source_classes(self):
        selection = self.cat.select_source_class("galactic")
        assert len(selection.table) == 101

        selection = self.cat.select_source_class("extra-galactic")
        assert len(selection.table) == 1684

        selection = self.cat.select_source_class("unassociated")
        assert len(selection.table) == 1010

        selection = self.cat.select_source_class("ALL")
        assert len(selection.table) == 239

        selection = self.cat.select_source_class("PSR")
        assert len(selection.table) == 143


@requires_data("gammapy-extra")
class TestSourceCatalog1FHL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog1FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 514

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 18

    @pytest.mark.parametrize(
        "name", ["Crab", "1FHL J0534.5+2201", "2FGL J0534.5+2201", "PSR J0534+2200"]
    )
    def test_crab_alias(self, name):
        assert str(self.cat["Crab"]) == str(self.cat[name])


@requires_data("gammapy-extra")
class TestSourceCatalog2FHL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog2FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 360

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 25

    @pytest.mark.parametrize(
        "name", ["Crab", "3FGL J0534.5+2201i", "1FHL J0534.5+2201", "TeV J0534+2200"]
    )
    def test_crab_alias(self, name):
        assert str(self.cat["Crab"]) == str(self.cat[name])


@requires_data("gammapy-extra")
class TestSourceCatalog3FHL:
    @classmethod
    def setup_class(cls):
        cls.cat = SourceCatalog3FHL()

    def test_main_table(self):
        assert len(self.cat.table) == 1556

    def test_extended_sources(self):
        table = self.cat.extended_sources_table
        assert len(table) == 55

    def test_select_source_classes(self):
        selection = self.cat.select_source_class("galactic")
        assert len(selection.table) == 44

        selection = self.cat.select_source_class("extra-galactic")
        assert len(selection.table) == 1177

        selection = self.cat.select_source_class("unassociated")
        assert len(selection.table) == 177

        selection = self.cat.select_source_class("ALL")
        assert len(selection.table) == 135

        selection = self.cat.select_source_class("PSR")
        assert len(selection.table) == 53
