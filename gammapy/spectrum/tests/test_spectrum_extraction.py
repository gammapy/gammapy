# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.tests.helper import pytest
import astropy.units as u
from numpy.testing import assert_allclose
from astropy.tests.helper import assert_quantity_allclose

from ...data import DataStore, ObservationTable, Target, ObservationList
from ...datasets import gammapy_extra
from ...image import ExclusionMask
from ...background import ring_background_estimate
from ...extern.regions.shapes import CircleSkyRegion
from ...spectrum import (
    SpectrumExtraction,
    SpectrumObservation,
    SpectrumObservationList,
    PHACountsSpectrum,
)
from ...utils.energy import EnergyBounds
from ...utils.testing import requires_dependency, requires_data
from ...utils.scripts import read_yaml, make_path


@pytest.mark.parametrize("pars,results",[
    (dict(containment_correction=False),dict(n_on=95,
                                             aeff=549861.8268659255*u.m**2,
                                             ethresh=0.4230466456851681*u.TeV)),
    (dict(containment_correction=True), dict(n_on=95,
                                             aeff=393356.18322397786*u.m**2,
                                             ethresh=0.6005317540449035*u.TeV)),
])
@requires_dependency('scipy')
@requires_data('gammapy-extra')

def test_spectrum_extraction(pars,results,tmpdir):

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

    irad = Angle('0.5 deg')
    orad = Angle('0.6 deg')
    bk = [ring_background_estimate(
        center, radius, irad, orad, _.events) for _ in obs]
    #bk = dict(method='reflected', n_min=2, exclusion=excl)

    bounds = EnergyBounds.equal_log_spacing(1, 10, 40, unit='TeV')
    etrue = EnergyBounds.equal_log_spacing(0.1, 30, 100, unit='TeV')

    ana = SpectrumExtraction(target, obs, bk,
                             e_reco=bounds,
                             e_true=etrue,
                             containment_correction=pars['containment_correction'])

    ana.run(outdir=tmpdir)

    # test methods on SpectrumObservationList
    obslist = ana.observations

    assert len(obslist) == 2
    obs23523 = obslist.obs(23523)

    assert obs23523.on_vector.total_counts.value == results['n_on']
    new_list = [obslist.obs(_) for _ in [23523, 23592]]
    assert new_list[0].obs_id == 23523
    assert new_list[1].obs_id == 23592

    ana.define_ethreshold(method_lo_threshold="AreaMax", percent_area_max=10)
    assert_allclose(ana.observations[0].lo_threshold,results['ethresh'])

    assert_quantity_allclose(obs23523.aeff.evaluate(energy=5*u.TeV),results['aeff'])
    
    # Write on set of output files to gammapy-extra as input for other tests
    # and check I/O
    if not pars['containment_correction']:
        outdir = gammapy_extra.filename("datasets/hess-crab4_pha")
        ana.observations.write(outdir)
        testobs = SpectrumObservation.read(make_path(outdir)/'pha_obs23523.fits') 
        assert str(testobs.aeff) == str(obs23523.aeff) 
        assert str(testobs.on_vector) == str(obs23523.on_vector) 
        assert str(testobs.off_vector) == str(obs23523.off_vector) 
        assert str(testobs.edisp) == str(obs23523.edisp) 
    
@pytest.mark.xfail(reason='This needs some changes to the API')
@requires_data('gammapy-extra')
def test_observation_stacking():
    obs_table_file = gammapy_extra.filename(
        'datasets/hess-crab4_pha/observation_table.fits')

    obs_table = ObservationTable.read(obs_table_file)
    temp = SpectrumObservationList.from_observation_table(obs_table)

    observations = temp.get_obslist_from_ids([23523, 23592])
    spectrum_observation_grouped = SpectrumObservation.stack_observation_list(observations, 0)
    obs0 = observations[0]
    obs1 = observations[1]

    # Test sum on/off vector and alpha group
    sum_on_vector = obs0.on_vector.counts + obs1.on_vector.counts
    sum_off_vector = obs0.off_vector.counts + obs1.off_vector.counts
    alpha_times_off_tot = obs0.alpha * obs0.off_vector.total_counts + obs1.alpha * obs1.off_vector.total_counts
    total_off = obs0.off_vector.total_counts + obs1.off_vector.total_counts

    assert_allclose(spectrum_observation_grouped.on_vector.counts, sum_on_vector)
    assert_allclose(spectrum_observation_grouped.off_vector.counts, sum_off_vector)
    assert_allclose(spectrum_observation_grouped.alpha, alpha_times_off_tot / total_off)

    # Test arf group
    total_time = obs0.meta.livetime + obs1.meta.livetime
    arf_times_livetime = obs0.meta.livetime * obs0.effective_area.data \
                         + obs1.meta.livetime * obs1.effective_area.data
    assert_allclose(spectrum_observation_grouped.effective_area.data, arf_times_livetime / total_time)
    # Test rmf group
    rmf_times_arf_times_livetime = obs0.meta.livetime * obs0.effective_area.data \
                                   * obs0.energy_dispersion.pdf_matrix.T \
                                   + obs1.meta.livetime * obs1.effective_area.data \
                                     * obs1.energy_dispersion.pdf_matrix.T

    inan = np.isnan(rmf_times_arf_times_livetime / arf_times_livetime)
    pdf_expexted = rmf_times_arf_times_livetime / arf_times_livetime
    pdf_expexted[inan] = 0
    assert_allclose(spectrum_observation_grouped.energy_dispersion.pdf_matrix, pdf_expexted.T, atol=1e-6)
