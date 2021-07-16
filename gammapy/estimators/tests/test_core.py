# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Table
import astropy.units as u
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.maps import MapAxis, WcsNDMap, RegionNDMap, RegionGeom, Maps
from gammapy.estimators.core import FluxEstimate
from gammapy.estimators import ESTIMATOR_REGISTRY


def test_estimator_registry():
    assert "Estimator" in str(ESTIMATOR_REGISTRY)


@pytest.fixture(scope="session")
def region_map_flux_estimate():
    axis = MapAxis.from_energy_edges((0.1, 1.0, 10.0), unit="TeV")
    geom = RegionGeom.create(
        "galactic;circle(0, 0, 0.1)", axes=[axis]
    )

    maps = Maps.from_geom(
        geom=geom,
        names=["norm", "norm_err", "norm_errn", "norm_errp", "norm_ul"]
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


class TestFluxEstimate:
    def test_table_properties(self, region_map_flux_estimate):
        model = PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2)
        fe = FluxEstimate(data=region_map_flux_estimate, reference_spectral_model=model)

        print(fe.available_quantities)
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

    def test_missing_column(self, region_map_flux_estimate):
        del region_map_flux_estimate["norm_errn"]
        model = PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2)
        fe = FluxEstimate(data=region_map_flux_estimate, reference_spectral_model=model)

        with pytest.raises(AttributeError):
            fe.dnde_errn

    def test_map_properties(self, map_flux_estimate):
        model = PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2)
        fe = FluxEstimate(data=map_flux_estimate, reference_spectral_model=model)

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
        assert_allclose(
            fe.eflux_err.quantity.value[:, 2, 2], [2.302585e-11, 2.302585e-11]
        )
        assert_allclose(
            fe.eflux_errn.quantity.value[:, 2, 2], [4.60517e-11, 4.60517e-11]
        )
        assert_allclose(
            fe.eflux_errp.quantity.value[:, 2, 2], [3.4538775e-11, 3.4538775e-11]
        )
        assert_allclose(fe.eflux_ul.quantity.value[:, 2, 2], [4.60517e-10, 4.60517e-10])
