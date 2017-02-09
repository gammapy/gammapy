# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import itertools
import numpy as np
from astropy.units import Quantity
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest, assert_quantity_allclose
from astropy.table import Table
import astropy.units as u
from ...utils.testing import requires_dependency, requires_data
from ..flux_point import (_e_ref_lafferty, _dnde_from_flux,
                          compute_flux_points_dnde,
                          FluxPointEstimator, FluxPoints)
from ..flux_point import SEDLikelihoodProfile
from ...spectrum.powerlaw import power_law_evaluate, power_law_integral_flux
from ...spectrum import SpectrumObservation, SpectrumEnergyGroupMaker
from ...spectrum.models import PowerLaw, SpectralModel
from ...utils.modeling import ParameterList

E_REF_METHODS = ['table', 'lafferty', 'log_center']
indices = [0, 1, 2, 3]

FLUX_POINTS_FILES = ['diff_flux_points.ecsv',
                     'diff_flux_points.fits',
                     'flux_points.ecsv',
                     'flux_points.fits']

class LWTestModel(SpectralModel):
    parameters = ParameterList([])

    @staticmethod
    def evaluate(x):
        return 1E4 * np.exp(-6 * x)

    def integral(self, xmin, xmax):
        return - 1. / 6 * 1E4 * (np.exp(-6 * xmax) - np.exp(-6 * xmin))

    def inverse(self, y):
        return - 1. / 6 * np.log(y * 1E-4)


class XSqrTestModel(SpectralModel):
    parameters = ParameterList([]) 

    @staticmethod
    def evaluate(x):
        return x ** 2

    def integral(self, xmin, xmax):
        return 1. / 3 * (xmax ** 3 - xmin ** 2)

    def inverse(self, y):
        return np.sqrt(y)


class ExpTestModel(SpectralModel):
    parameters = ParameterList([]) 

    @staticmethod
    def evaluate(x):
        return np.exp(x * u.Unit('1 / TeV'))

    def integral(self, xmin, xmax):
        return np.exp(xmax * u.Unit('1 / TeV')) - np.exp(xmin * u.Unit('1 / TeV'))

    def inverse(self, y):
        return np.log(y * u.TeV) * u.TeV


def test_e_ref_lafferty():
    """
    Tests Lafferty & Wyatt x-point method.

    Using input function g(x) = 10^4 exp(-6x) against
    check values from paper Lafferty & Wyatt. Nucl. Instr. and Meth. in Phys.
    Res. A 355 (1995) 541-547, p. 542 Table 1
    """
    # These are the results from the paper
    desired = np.array([0.048, 0.190, 0.428, 0.762])

    model = LWTestModel()
    e_min = np.array([0.0, 0.1, 0.3, 0.6])
    e_max = np.array([0.1, 0.3, 0.6, 1.0])
    actual = _e_ref_lafferty(model, e_min, e_max)
    assert_allclose(actual, desired, atol=1e-3)


@requires_dependency('scipy')
def test_dnde_from_flux():
    """Tests y-value normalization adjustment method.
    """
    e_min = np.array([10, 20, 30, 40])
    e_max = np.array([20, 30, 40, 50])
    flux = np.array([42, 52, 62, 72])  # 'True' integral flux in this test bin

    # Get values
    model = XSqrTestModel()
    e_ref = _e_ref_lafferty(model, e_min, e_max)
    dnde = _dnde_from_flux(flux, model, e_ref, e_min, e_max)

    # Set up test case comparison
    dnde_model = model(e_ref)

    # Test comparison result
    desired = model.integral(e_min, e_max)
    # Test output result
    actual = flux * (dnde_model / dnde)
    # Compare
    assert_allclose(actual, desired, rtol=1e-6)


@requires_dependency('scipy')
@pytest.mark.parametrize('method', E_REF_METHODS)
def test_compute_flux_points_dnde_exp(method):
    """
    Tests against analytical result or result from gammapy.spectrum.powerlaw.
    """
    model = ExpTestModel()

    e_min = [1.0, 10.0] * u.TeV
    e_max = [10.0, 100.0] * u.TeV
    spectral_index = 2.0

    table = Table()
    table.meta['SED_TYPE'] = 'flux'
    table['e_min'] = e_min
    table['e_max'] = e_max

    flux = model.integral(e_min, e_max)
    table['flux'] = flux

    if method == 'log_center':
        e_ref = np.sqrt(e_min * e_max)
    elif method == 'table':
        e_ref = [2.0, 20.0] * u.TeV
        table['e_ref'] = e_ref
    elif method == 'lafferty':
        e_ref = _e_ref_lafferty(model, e_min, e_max)

    result = compute_flux_points_dnde(FluxPoints(table), model, method)

    # Test energy
    actual = result.e_ref
    assert_quantity_allclose(actual, e_ref, rtol=1e-8)

    # Test flux
    actual = result.table['dnde'].quantity
    desired = model(e_ref)
    assert_quantity_allclose(actual, desired, rtol=1e-8)


@pytest.mark.xfail(reason='Cannot freeze parameters at the moment')
@requires_data('gammapy-extra')
@requires_dependency('sherpa')
@requires_dependency('scipy')
class TestFluxEstimator:
    def setup(self):
        self.model = PowerLaw(
            index=Quantity(2, ''),
            amplitude=Quantity(1e-11, 'm-2 s-1 TeV-1'),
            reference=Quantity(1, 'TeV'),
        )

        # TODO: simulate known spectrum instead of using this example:
        filename = '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits'
        self.obs = SpectrumObservation.read(filename)
        self.seg = SpectrumEnergyGroupMaker(obs=self.obs)
        ebounds = [0.3, 1.001, 3, 10.001, 30] * u.TeV
        self.seg.compute_range_safe()
        self.seg.compute_groups_fixed(ebounds=ebounds)

        self.groups = self.seg.groups

    def test_with_power_law(self):
        # import logging
        # logging.basicConfig(level=logging.DEBUG)

        fpe = FluxPointEstimator(
            obs=self.obs,
            groups=self.groups,
            model=self.model,
        )

        assert 'FluxPointEstimator' in str(fpe)

        fpe.compute_points()
        flux_points = fpe.flux_points
        flux_points.table.pprint()
        flux_points.table.info()

        actual = flux_points.table['dnde'].quantity[2]
        desired = Quantity(5.737510858664804e-09, 'm-2 s-1 TeV-1')
        assert_quantity_allclose(actual, desired, rtol=1e-3)

        actual = flux_points.table['dnde_err'].quantity[2]
        desired = Quantity(9.904468386098078e-10, 'm-2 s-1 TeV-1')
        assert_quantity_allclose(actual, desired, rtol=1e-3)

    def test_with_ecpl(self):
        # TODO: implement
        assert True


@requires_data('gammapy-extra')
class TestSEDLikelihoodProfile:
    def setup(self):
        self.sed = SEDLikelihoodProfile.read('$GAMMAPY_EXTRA/datasets/spectrum/'
                                             'llsed_hights.fits')

    def test_basics(self):
        # print(self.sed)
        assert 'SEDLikelihoodProfile' in str(self.sed)

    @requires_dependency('matplotlib')
    def test_plot(self):
        ax = self.sed.plot()


@pytest.fixture(params=FLUX_POINTS_FILES)
def flux_points(request):
    path = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/' + request.param
    return FluxPoints.read(path)

@requires_dependency('yaml')
@requires_data('gammapy-extra')
class TestFluxPoints:

    @requires_dependency('matplotlib')
    def test_plot(self, flux_points):
        import matplotlib.pyplot as plt
        ax = plt.gca()
        flux_points.plot(ax=ax)

    def test_info(self, flux_points):
        info = str(flux_points)
        assert flux_points.sed_type in info

    def test_e_ref(self, flux_points):
        actual = flux_points.e_ref
        if flux_points.sed_type == 'dnde':
            pass
        elif flux_points.sed_type == 'flux':
            desired = np.sqrt(flux_points.e_min * flux_points.e_max)
            assert_quantity_allclose(actual, desired)

    def test_e_min(self, flux_points):
        if flux_points.sed_type == 'dnde':
            pass
        elif flux_points.sed_type == 'flux':
            actual = flux_points.e_min
            desired = 299530.9757217623 * u.MeV
            assert_quantity_allclose(actual.sum(), desired)

    def test_e_max(self, flux_points):
        if flux_points.sed_type == 'dnde':
            pass
        elif flux_points.sed_type == 'flux':
            actual = flux_points.e_max
            desired = 399430.975721694 * u.MeV
            assert_quantity_allclose(actual.sum(), desired)

    def test_write_fits(self, tmpdir, flux_points):
        filename = tmpdir / 'flux_points.fits'
        flux_points.write(filename)
        actual = FluxPoints.read(filename)
        assert str(flux_points) == str(actual)

    def test_write_ecsv(self, tmpdir, flux_points):
        filename = tmpdir / 'flux_points.ecsv'
        flux_points.write(filename)
        actual = FluxPoints.read(filename)
        assert str(flux_points) == str(actual)


@requires_data('gammapy-extra')
def test_compute_flux_points_dnde():
    """
    Test compute_flux_points_dnde on reference spectra.
    """
    path = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/'
    flux_points = FluxPoints.read(path + 'flux_points.fits')
    desired_fp = FluxPoints.read(path + 'diff_flux_points.fits')

    # TODO: verify index=2.2, but it seems to give reasonable values
    model = PowerLaw(2.2 * u.Unit(''), 1E-12 * u.Unit('cm-2 s-1 TeV-1'), 1 * u.TeV)
    actual_fp = compute_flux_points_dnde(flux_points, model=model, method='log_center')

    for column in ['dnde', 'dnde_err', 'dnde_ul']:
        actual = actual_fp.table[column].quantity
        desired = desired_fp.table[column].quantity
        assert_quantity_allclose(actual, desired, rtol=1E-12)
