# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import absolute_import, division, print_function, unicode_literals
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.tests.helper import assert_quantity_allclose
from ...extern.regions.shapes import CircleSkyRegion
from ...utils.energy import EnergyBounds
from ...utils.scripts import make_path
from ...utils.testing import requires_dependency, requires_data
from ...background import ring_background_estimate
from ...data import DataStore, Target, ObservationList
from ...datasets import gammapy_extra
from ...image import ExclusionMask
from ...spectrum import SpectrumExtraction, SpectrumObservation
import numpy as np

@pytest.mark.parametrize("pars,results", [
    (dict(containment_correction=False), dict(n_on=172,
                                              sigma=28.18,
                                              aeff=549861.8 * u.m ** 2,
                                              ethresh=0.4327 * u.TeV)),
    (dict(containment_correction=True), dict(n_on=172,
                                             sigma=28.18,
                                             aeff=393356.2 * u.m ** 2,
                                             ethresh=0.625 * u.TeV)),
])
@requires_dependency('scipy')
@requires_data('gammapy-extra')
def test_spectrum_extraction(pars, results, tmpdir):

    center = SkyCoord(83.63, 22.01, unit='deg', frame='icrs')
    radius = Angle('0.11 deg')
    on_region = CircleSkyRegion(center, radius)
    target = Target(on_region)

    obs_id = [23523, 23592]
    store = gammapy_extra.filename("datasets/hess-crab4-hd-hap-prod2")
    ds = DataStore.from_dir(store)
    obs = ObservationList([ds.obs(_) for _ in obs_id])

    exclusion_file = gammapy_extra.filename(
        "datasets/exclusion_masks/tevcat_exclusion.fits")
    excl = ExclusionMask.read(exclusion_file)

    bk = dict(method='reflected', n_min=2, exclusion=excl)

    # Restrict energy binning to a range where the aeff interpolation does not
    # give none for HAP test files
    # TODO: set low energies to 0 and extrapolate high energies
    e_true = np.logspace(-1, 1.9, 70) * u.TeV

    ana = SpectrumExtraction(target,
                             obs,
                             bk,
                             e_true = e_true,
                             containment_correction=pars['containment_correction'])

    ana.run(outdir=tmpdir)

    ana.define_energy_threshold(method_lo_threshold="area_max", percent=10)

    assert_quantity_allclose(ana.observations[0].lo_threshold,
                             results['ethresh'], rtol=1e-3)

    assert_quantity_allclose(ana.observations[0].off_vector.lo_threshold,
                             ana.observations[0].on_vector.lo_threshold)

    assert_quantity_allclose(ana.observations[0].aeff.evaluate(
        energy=5 * u.TeV), results['aeff'], rtol=1e-3)
    assert ana.observations[0].total_stats.n_on == results['n_on']
    assert_allclose(ana.observations[1].total_stats.sigma, results['sigma'],
                    atol=1e-2)

    # Write on set of output files to gammapy-extra as input for other tests
    # and check I/O
    if not pars['containment_correction']:
        outdir = gammapy_extra.filename("datasets/hess-crab4_pha")
        ana.observations.write(outdir)

        testobs = SpectrumObservation.read(make_path(outdir) / 'pha_obs23523.fits')
        assert_quantity_allclose(testobs.aeff.data,
                                 ana.observations[0].aeff.data)
        assert_quantity_allclose(testobs.on_vector.data,
                                 ana.observations[0].on_vector.data)
        assert_quantity_allclose(testobs.on_vector.energy.nodes,
                                 ana.observations[0].on_vector.energy.nodes)
