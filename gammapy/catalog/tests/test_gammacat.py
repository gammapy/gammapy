# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy.tests.helper import assert_quantity_allclose, pytest
from astropy import units as u
from ...utils.testing import requires_data, requires_dependency
from ..gammacat import SourceCatalogGammaCat, SourceCatalogObjectGammaCat
from ...utils.energy import Energy


SOURCES = ['Vela X', 'HESS J1848-018', 'HESS J1813-178']
DESIRED_SM = [ {'flux_at_1TeV': 1.36e-11 * u.Unit('1 / (cm2 TeV s)'),
                'flux_above_1TeV': 2.104e-11 * u.Unit('1 / (cm2 s)'),
                'eflux_1_10TeV': 5.783e-11 * u.Unit('TeV / (cm2 s)')},

               {'flux_at_1TeV': 3.7e-12 * u.Unit('1 / (cm2 TeV s)'),
                'flux_above_1TeV': 2.056e-12 * u.Unit('1 / (cm2 s)'),
                'eflux_1_10TeV': 3.892e-12 * u.Unit('TeV / (cm2 s)')},

               {'flux_at_1TeV': 2.678e-12 * u.Unit('1 / (cm2 TeV s)'),
                'flux_above_1TeV': 2.457e-12 * u.Unit('1 / (cm2 s)'),
                'eflux_1_10TeV': 5.5697e-12 * u.Unit('TeV / (cm2 s)')}]

DESIRED_FP = [{'N': 24},
              {'N': 11},
              {'N': 13}]

DESIRED_BF = [{'energy_sum': 40.8695 * u.TeV,
               'flux_lo_sum': 3.965e-11 * u.Unit('1 / (cm2 s TeV)'),
               'flux_hi_sum': 4.555e-11 * u.Unit('1 / (cm2 s TeV)')},

              {'energy_sum': 40.8695 * u.TeV,
               'flux_lo_sum': 6.2880e-12 * u.Unit('1 / (cm2 s TeV)'),
               'flux_hi_sum': 8.168e-12 * u.Unit('1 / (cm2 s TeV)')},

              {'energy_sum': 40.8695 * u.TeV,
               'flux_lo_sum': 5.691e-12 * u.Unit('1 / (cm2 s TeV)'),
               'flux_hi_sum': 7.181e-12 * u.Unit('1 / (cm2 s TeV)')}]


@requires_data('gamma-cat')
class TestSourceCatalogGammaCat:
    def setup(self):
        self.cat = SourceCatalogGammaCat()

    def test_source_table(self):
        assert self.cat.name == 'gamma-cat'
        assert len(self.cat.table) == 162


@requires_data('gamma-cat')
class TestSourceCatalogObjectGammaCat:
    def setup(self):
        self.cat = SourceCatalogGammaCat()

    @pytest.mark.parametrize(['name', 'desired'], zip(SOURCES, DESIRED_SM))
    def test_spectral_model(self, name, desired):
        source = self.cat[name]
        spectral_model = source.spectral_model

        emin, emax = [1, 10] * u.TeV
        einf = 1E10 * u.TeV
        flux_at_1TeV = spectral_model(emin)
        flux_above_1TeV = spectral_model.integral(emin=emin, emax=einf)
        eflux_1_10TeV = spectral_model.energy_flux(emin=emin, emax=emax)

        assert_quantity_allclose(flux_at_1TeV, desired['flux_at_1TeV'], rtol=1E-3)
        assert_quantity_allclose(flux_above_1TeV, desired['flux_above_1TeV'], rtol=1E-3)
        assert_quantity_allclose(eflux_1_10TeV, desired['eflux_1_10TeV'], rtol=1E-3)

    @pytest.mark.parametrize(['name', 'desired'], zip(SOURCES, DESIRED_FP))
    def test_flux_points(self, name, desired):
        source = self.cat[name]

        assert name == source.name
        flux_points = source.flux_points
        assert len(flux_points) == desired['N']

    @requires_dependency('uncertainties')
    @pytest.mark.parametrize(['name', 'desired'], zip(SOURCES, DESIRED_BF))
    def test_butterfly(self, name, desired):
        source = self.cat[name]
        emin, emax = [1, 10] * u.TeV
        energies = Energy.equal_log_spacing(emin, emax, 10)

        butterfly = source.spectrum.butterfly(energies)

        assert_quantity_allclose(butterfly['energy'].sum(), desired['energy_sum'], rtol=1E-3)
        assert_quantity_allclose(butterfly['flux_lo'].sum(), desired['flux_lo_sum'], rtol=1E-3)
        assert_quantity_allclose(butterfly['flux_hi'].sum(), desired['flux_hi_sum'], rtol=1E-3)
