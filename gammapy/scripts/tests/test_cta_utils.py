# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import astropy.units as u
import numpy as np
import pytest
from numpy.testing import assert_allclose
from ...utils.testing import requires_data, requires_dependency
from ...scripts.cta_irf import CTAPerf
from ...scripts.cta_utils import Target, ObservationParameters, CTAObservationSimulation
from ...spectrum.models import PowerLaw, AbsorbedSpectralModel, Absorption


@requires_data('gammapy-extra')
@pytest.fixture(scope='session')
def target():
    name = 'test'

    index = 3. * u.Unit('')
    reference = 1 * u.TeV
    amplitude = 1e-12 * u.Unit('cm-2 s-1 TeV-1')
    pwl = PowerLaw(index=index,
                   reference=reference,
                   amplitude=amplitude)
    redshift = 0.1
    absorption = Absorption.read('$GAMMAPY_EXTRA/datasets/ebl/ebl_dominguez11.fits.gz')
    model = AbsorbedSpectralModel(spectral_model=pwl,
                                  absorption=absorption,
                                  parameter=redshift)
    return Target(name=name, model=model)


@pytest.fixture(scope='session')
def obs_param():
    alpha = 0.2 * u.Unit('')
    livetime = 5. * u.h
    emin = 0.03 * u.TeV
    emax = 5 * u.TeV

    return ObservationParameters(alpha=alpha, livetime=livetime,
                                 emin=emin, emax=emax)


@requires_data('gammapy-extra')
def test_target(target):
    text = str(target)
    assert '*** Target parameters ***' in text
    assert 'Name=test' in text
    assert 'index=3.' in text
    assert 'reference=1' in text
    assert 'amplitude=1e-12' in text
    assert 'redshift=0.1' in text


@requires_data('gammapy-extra')
def test_observation_parameters(obs_param):
    text = str(obs_param)
    assert '*** Observation parameters summary ***' in text
    assert 'alpha=0.2' in text
    assert 'livetime=5.0' in text
    assert 'emin=0.03' in text
    assert 'emax=5.0' in text


@requires_data('gammapy-extra')
@requires_dependency('scipy')
def test_cta_simulation(target, obs_param):
    filename = '$GAMMAPY_EXTRA/datasets/cta/perf_prod2/point_like_non_smoothed/South_5h.fits.gz'

    perf = CTAPerf.read(filename)
    simu = CTAObservationSimulation.simulate_obs(
        perf=perf,
        target=target,
        obs_param=obs_param,
        random_state=0,
    )
    text = str(simu)
    assert '*** Observation summary report ***' in text
    stats = simu.total_stats
    assert_allclose(stats.sigma, 36.51439765644547)
    assert_allclose(stats.excess, 5322.799999999996)
    assert_allclose(stats.background, 15712.200000000004)

    # With reprojection of background rate
    e_reco = np.logspace(np.log10(0.02), np.log10(100), 51) * u.TeV
    perf = CTAPerf.read(filename, e_reco=e_reco)
    simu = CTAObservationSimulation.simulate_obs(
        perf=perf,
        target=target,
        obs_param=obs_param,
        random_state=0,
    )
    stats = simu.total_stats
    assert_allclose(stats.sigma, 40.326731781633505)
    assert_allclose(stats.excess, 5319.200000000001)
    assert_allclose(stats.background, 12519.8)
