# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import copy
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.tests.helper import assert_quantity_allclose
from regions import CircleSkyRegion
from ...utils.scripts import make_path
from ...utils.testing import requires_dependency, requires_data
from ...data import DataStore, Target, ObservationList
from ...datasets import gammapy_extra
from ...image import SkyMask
from ...spectrum import SpectrumExtraction, SpectrumObservation


@pytest.fixture(scope='session')
def obs():
    """An example ObservationList for tests."""
    obs_id = [23523, 23592]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)
    obs = ObservationList([ds.obs(_) for _ in obs_id])
    return obs


@pytest.fixture(scope='session')
def target():
    """An example Target for tests."""
    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.11 deg')
    on_region = CircleSkyRegion(center, radius)
    target = Target(on_region)
    return target


@pytest.fixture(scope='session')
def bkg():
    """An example bkg dict for tests."""
    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = SkyMask.read(exclusion_file)
    bkg = dict(method='reflected', n_min=2, exclusion=excl)
    return bkg


@pytest.fixture(scope='session')
def extraction():
    """An example SpectrumExtraction for tests."""
    # Restrict true energy range covered by HAP exporter
    e_true = np.logspace(-1, 1.9, 70) * u.TeV

    extraction = SpectrumExtraction(target=target(),
                                    obs=obs(),
                                    background=bkg(),
                                    e_true=e_true
                                    )
    return extraction


@requires_dependency('scipy')
@requires_data('gammapy-extra')
class TestSpectrumExtraction:
    @pytest.mark.parametrize("pars, results", [
        (dict(containment_correction=False), dict(n_on=172,
                                                  sigma=24.98,
                                                  aeff=549861.8 * u.m ** 2
                                                  )),
        (dict(containment_correction=True), dict(n_on=172,
                                                 sigma=24.98,
                                                 aeff=412731.8043631101 * u.m ** 2
                                                 ))
    ])
    def test_extract(self, pars, results, target, obs, bkg, tmpdir):
        """Test quantitative output for various configs"""
        extraction = SpectrumExtraction(target=target,
                                        obs=obs,
                                        background=copy.deepcopy(bkg),
                                        **pars)

        # TODO: Improve API
        print(extraction.background)
        extraction.estimate_background(extraction.background)
        extraction.extract_spectrum()
        obs = extraction.observations[0]
        aeff_actual = obs.aeff.data.evaluate(energy=5 * u.TeV)

        # TODO: Introduce assert_stats_allclose
        n_on_actual = obs.total_stats.n_on
        sigma_actual = obs.total_stats.sigma

        assert_quantity_allclose(aeff_actual, results['aeff'], rtol=1e-3)
        assert n_on_actual == results['n_on']
        assert_allclose(sigma_actual, results['sigma'], atol=1e-2)

    def test_run(self, tmpdir, extraction):
        """Test the run method and check if files are written correctly"""
        extraction.run(outdir=tmpdir)
        testobs = SpectrumObservation.read(tmpdir / 'ogip_data' / 'pha_obs23523.fits')
        assert_quantity_allclose(testobs.aeff.data.data,
                                 extraction.observations[0].aeff.data.data)
        assert_quantity_allclose(testobs.on_vector.data.data,
                                 extraction.observations[0].on_vector.data.data)
        assert_quantity_allclose(testobs.on_vector.energy.nodes,
                                 extraction.observations[0].on_vector.energy.nodes)

    def test_define_energy_threshold(self, extraction):
        # TODO: Find better API for this
        extraction.define_energy_threshold(method_lo_threshold="area_max",
                                           percent=10)
        assert_quantity_allclose(extraction.observations[0].lo_threshold,
                                 0.6812920690579611 * u.TeV,
                                 rtol=1e-3)
