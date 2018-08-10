# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from numpy.testing import assert_allclose
from astropy.units import Quantity
import pytest
from astropy.table import Table
import astropy.units as u
from ...catalog.fermi import SourceCatalog3FGL
from ...utils.testing import requires_dependency, requires_data, assert_quantity_allclose
from ...utils.modeling import ParameterList
from ..results import SpectrumResult
from ..fit import SpectrumFit
from ..observation import SpectrumObservation
from ..energy_group import SpectrumEnergyGroupMaker
from ..models import PowerLaw, SpectralModel, ExponentialCutoffPowerLaw
from ..flux_point import FluxPoints, FluxPointProfiles, FluxPointFitter, FluxPointEstimator

FLUX_POINTS_FILES = [
    'diff_flux_points.ecsv',
    'diff_flux_points.fits',
    'flux_points.ecsv',
    'flux_points.fits',
]


class LWTestModel(SpectralModel):
    parameters = ParameterList([])

    @staticmethod
    def evaluate(x):
        return 1e4 * np.exp(-6 * x)

    def integral(self, xmin, xmax, **kwargs):
        return - 1. / 6 * 1e4 * (np.exp(-6 * xmax) - np.exp(-6 * xmin))

    def inverse(self, y):
        return - 1. / 6 * np.log(y * 1e-4)


class XSqrTestModel(SpectralModel):
    parameters = ParameterList([])

    @staticmethod
    def evaluate(x):
        return x ** 2

    def integral(self, xmin, xmax, **kwargs):
        return 1. / 3 * (xmax ** 3 - xmin ** 2)

    def inverse(self, y):
        return np.sqrt(y)


class ExpTestModel(SpectralModel):
    parameters = ParameterList([])

    @staticmethod
    def evaluate(x):
        return np.exp(x * u.Unit('1 / TeV'))

    def integral(self, xmin, xmax, **kwargs):
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
    actual = FluxPoints._e_ref_lafferty(model, e_min, e_max)
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
    e_ref = FluxPoints._e_ref_lafferty(model, e_min, e_max)
    dnde = FluxPoints._dnde_from_flux(flux, model, e_ref, e_min, e_max, pwl_approx=False)

    # Set up test case comparison
    dnde_model = model(e_ref)

    # Test comparison result
    desired = model.integral(e_min, e_max)
    # Test output result
    actual = flux * (dnde_model / dnde)
    # Compare
    assert_allclose(actual, desired, rtol=1e-6)


@requires_dependency('scipy')
@pytest.mark.parametrize('method', ['table', 'lafferty', 'log_center'])
def test_compute_flux_points_dnde_exp(method):
    """
    Tests against analytical result or result from gammapy.spectrum.powerlaw.
    """
    model = ExpTestModel()

    e_min = [1.0, 10.0] * u.TeV
    e_max = [10.0, 100.0] * u.TeV

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
        e_ref = FluxPoints._e_ref_lafferty(model, e_min, e_max)

    result = FluxPoints(table).to_sed_type('dnde', model=model, method=method)

    # Test energy
    actual = result.e_ref
    assert_quantity_allclose(actual, e_ref, rtol=1e-8)

    # Test flux
    actual = result.table['dnde'].quantity
    desired = model(e_ref)
    assert_quantity_allclose(actual, desired, rtol=1e-8)


@pytest.fixture(scope='session')
def obs():
    filename = '$GAMMAPY_EXTRA/datasets/hess-crab4_pha/pha_obs23523.fits'
    obs = SpectrumObservation.read(filename)
    return obs

@pytest.fixture(scope='session')
def model():
    model = PowerLaw()
    fit = SpectrumFit(obs(), model)
    fit.fit()
    fit.est_errors()
    return fit.result[0].model

@pytest.fixture(scope='session')
def seg():
    ebounds = [0.3, 1, 30] * u.TeV
    segm = SpectrumEnergyGroupMaker(obs=obs())
    segm.compute_groups_fixed(ebounds=ebounds)
    return segm.groups


@requires_data('gammapy-extra')
@requires_dependency('sherpa')
@requires_dependency('matplotlib')
@requires_dependency('scipy')
class TestFluxPointEstimator:
    def setup(self):
        self.fpe = FluxPointEstimator(
            obs=obs(),
            model=model(),
            groups=seg()
        )

    def test_basic(self):
        assert 'FluxPointEstimator' in str(self.fpe)

    def test_energy_range(self):
        group = self.fpe.groups[1]
        point = self.fpe.compute_flux_point(group)
        fit_range = self.fpe.fit.true_fit_range[0]
        assert_quantity_allclose(fit_range[0], group.energy_min)
        assert_quantity_allclose(fit_range[1], group.energy_max)
        
    def test_values(self):
        self.fpe.compute_points()
        flux_points = self.fpe.flux_points

        actual = flux_points.table['dnde'][0]
        desired = 4.988216856148768e-11
        assert_allclose(actual, desired, rtol=1e-2)

        actual = flux_points.table['dnde_err'][0]
        desired = 6.238263521179057e-12
        assert_allclose(actual, desired, rtol=1e-2)

        actual = flux_points.table['dnde_ul'][0]
        desired = 6.329808685878984e-11
        assert_allclose(actual, desired, rtol=1e-2)

        actual = flux_points.table['dnde_errn'][0]
        desired = 5.931500787551099e-12
        assert_allclose(actual, desired, rtol=1e-2)

        actual = flux_points.table['dnde_errp'][0]
        desired = 6.593114386768349e-12
        assert_allclose(actual, desired, rtol=1e-2)

    def test_spectrum_result(self):
        # TODO: Don't run this again
        self.fpe.compute_points()
        result = SpectrumResult(
            model=self.fpe.model,
            points=self.fpe.flux_points,
        )

        actual = result.flux_point_residuals[0][0]
        desired = -0.3217634756774566
        assert_allclose(actual, desired)

        actual = result.flux_point_residuals[1][0]
        desired = 0.08519973138393112
        assert_allclose(actual, desired)

        result.plot(energy_range=[1, 10] * u.TeV)


@requires_data('gammapy-extra')
class TestFluxPointProfiles:
    def setup(self):
        filename = '$GAMMAPY_EXTRA/datasets/spectrum/llsed_hights.fits'
        self.sed = FluxPointProfiles.read(filename)

    @pytest.mark.xfail(run=False)
    @requires_dependency('matplotlib')
    def test_plot(self):
        self.sed.plot()


@pytest.fixture(params=FLUX_POINTS_FILES, scope='session')
def flux_points(request):
    path = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/' + request.param
    return FluxPoints.read(path)


@requires_dependency('yaml')
@requires_data('gammapy-extra')
class TestFluxPoints:
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

    def test_drop_ul(self, flux_points):
        flux_points = flux_points.drop_ul()
        assert not np.any(flux_points.is_ul)

    def test_stack(self, flux_points):
        stacked = FluxPoints.stack([flux_points, flux_points])
        assert len(stacked.table) == 2 * len(flux_points.table)
        assert stacked.sed_type == flux_points.sed_type

    @requires_dependency('matplotlib')
    def test_plot(self, flux_points):
        flux_points.plot()


@requires_data('gammapy-extra')
def test_compute_flux_points_dnde():
    """
    Test compute_flux_points_dnde on reference spectra.
    """
    path = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/'
    flux_points = FluxPoints.read(path + 'flux_points.fits')
    desired_fp = FluxPoints.read(path + 'diff_flux_points.fits')

    # TODO: verify index=2.2, but it seems to give reasonable values
    model = PowerLaw(2.2 * u.Unit(''), 1e-12 * u.Unit('cm-2 s-1 TeV-1'), 1 * u.TeV)
    actual_fp = flux_points.to_sed_type('dnde', model=model, method='log_center')

    for column in ['dnde', 'dnde_err', 'dnde_ul']:
        actual = actual_fp.table[column].quantity
        desired = desired_fp.table[column].quantity
        assert_quantity_allclose(actual, desired, rtol=1e-12)


@requires_data('gammapy-extra')
def test_compute_flux_points_dnde_fermi():
    """
    Test compute_flux_points_dnde on fermi source.
    """
    fermi_3fgl = SourceCatalog3FGL()
    source = fermi_3fgl['3FGL J0835.3-4510']

    flux_points = source.flux_points.to_sed_type('dnde', model=source.spectral_model,
                                                 method='log_center', pwl_approx=True)

    for column in ['dnde', 'dnde_errn', 'dnde_errp', 'dnde_ul']:
        actual = flux_points.table['e2' + column].quantity
        desired = flux_points.table[column].quantity * flux_points.e_ref ** 2
        assert_quantity_allclose(actual[:-1], desired[:-1], rtol=1e-1)


@requires_data('gammapy-extra')
@requires_dependency('sherpa')
class TestFluxPointFitter:
    def setup(self):
        path = '$GAMMAPY_EXTRA/test_datasets/spectrum/flux_points/diff_flux_points.fits'
        self.flux_points = FluxPoints.read(path)

    def test_fit_pwl(self):
        fitter = FluxPointFitter()
        model = PowerLaw(index=2.3, amplitude='1e-12 cm-2 s-1 TeV-1', reference='1 TeV')
        result = fitter.run(self.flux_points, model)

        index = result['best-fit-model'].parameters['index']
        assert_quantity_allclose(index.quantity, 2.216 * u.Unit(''), rtol=1e-3)
        amplitude = result['best-fit-model'].parameters['amplitude']
        assert_quantity_allclose(amplitude.quantity, 2.1616E-13 * u.Unit('cm-2 s-1 TeV-1'), rtol=1e-3)
        assert_allclose(result['statval'], 25.2059, rtol=1e-3)
        assert_allclose(result['dof'], 22)
