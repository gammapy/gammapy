# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from astropy.table import Table
import astropy.units as u
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.maps import MapAxis, WcsNDMap
from gammapy.estimators.core import FluxEstimate
from gammapy.estimators import ESTIMATOR_REGISTRY


def test_estimator_registry():
    assert "Estimator" in str(ESTIMATOR_REGISTRY)


@pytest.fixture(scope="session")
def table_flux_estimate():
    axis = MapAxis.from_energy_edges((0.1, 1.0, 10.0), unit="TeV")

    cols = dict()
    cols["norm"] = np.array([1.0, 1.0])
    cols["norm_err"] = np.array([0.1, 0.1])
    cols["norm_errn"] = np.array([0.2, 0.2])
    cols["norm_errp"] = np.array([0.15, 0.15])
    cols["norm_ul"] = np.array([2.0, 2.0])
    cols["e_min"] = axis.edges[:-1]
    cols["e_max"] = axis.edges[1:]

    table = Table(cols, names=cols.keys())

    return table


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
    def test_table_properties(self, table_flux_estimate):
        model = PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2)
        fe = FluxEstimate(data=table_flux_estimate, reference_spectral_model=model)

        assert fe.dnde.unit == u.Unit("cm-2s-1TeV-1")
        assert_allclose(fe.dnde.value, [1e-9, 1e-11])
        assert_allclose(fe.dnde_err.value, [1e-10, 1e-12])
        assert_allclose(fe.dnde_errn.value, [2e-10, 2e-12])
        assert_allclose(fe.dnde_errp.value, [1.5e-10, 1.5e-12])
        assert_allclose(fe.dnde_ul.value, [2e-9, 2e-11])

        assert fe.e2dnde.unit == u.Unit("TeV cm-2s-1")
        assert_allclose(fe.e2dnde.value, [1e-10, 1e-10])

        assert fe.flux.unit == u.Unit("cm-2s-1")
        assert_allclose(fe.flux.value, [9e-10, 9e-11])

        assert fe.eflux.unit == u.Unit("TeV cm-2s-1")
        assert_allclose(fe.eflux.value, [2.302585e-10, 2.302585e-10])

    def test_missing_column(self, table_flux_estimate):
        table_flux_estimate.remove_column("norm_errn")
        model = PowerLawSpectralModel(amplitude="1e-10 cm-2s-1TeV-1", index=2)
        fe = FluxEstimate(data=table_flux_estimate, reference_spectral_model=model)

        with pytest.raises(KeyError):
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
