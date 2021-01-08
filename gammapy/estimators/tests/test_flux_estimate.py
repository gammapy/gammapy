# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from astropy.table import Table
import astropy.units as u
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.maps import MapAxis
from gammapy.estimators.flux_estimate import FluxEstimate

@pytest.fixture(scope="session")
def table_flux_estimate():
    axis = MapAxis.from_energy_edges((0.1, 1., 10.), unit='TeV')

    cols = dict()
    cols['norm'] = np.array([1.0, 1.0])
    cols['norm_err'] = np.array([0.1, 0.1])
    cols['norm_errn'] = np.array([0.2, 0.2])
    cols['norm_errp'] = np.array([0.15, 0.15])
    cols['norm_ul'] = np.array([2.0, 2.0])
    cols['e_min'] = axis.edges[:-1]
    cols['e_max'] = axis.edges[1:]

    table = Table(cols, names=cols.keys())

    print(axis.center)
    print(table)
    return table


class TestTableFluxEstimate:
    def test_properties(self, table_flux_estimate):
        model = PowerLawSpectralModel(amplitude='1e-10 cm-2s-1TeV-1', index=2)
        fe = FluxEstimate(data=table_flux_estimate, spectral_model=model)

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
        model = PowerLawSpectralModel(amplitude='1e-10 cm-2s-1TeV-1', index=2)
        fe = FluxEstimate(data=table_flux_estimate, spectral_model=model)

        with pytest.raises(KeyError):
            fe.dnde_errn
